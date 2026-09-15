"""Web UI backend for the FrontierLit Agent demo.

Calls the agent system directly (Orchestrator.run) rather than shelling
out to main.py's CLI entry point — this is what lets the frontend
render the real VerifiedSynthesis (citations, Critic report) and real
per-run telemetry (agent_system.observability.summarize_llm_calls)
instead of a captured stdout blob.

This is a single-operator local demo (see README/portfolio scope), not
a multi-tenant service: no auth, one shared default profile per chat,
in-memory run-progress tracking. Concurrency correctness beyond "one
person clicking around" is deliberately out of scope.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
import sqlite3
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from google.adk.sessions import InMemorySessionService
from pydantic import BaseModel

from agent_system.analyst.agent import Analyst
from agent_system.config import get_settings
from agent_system.critic.agent import Critic
from agent_system.observability import summarize_llm_calls
from agent_system.orchestrator.agent import Orchestrator, ResearchEvidenceUnavailableError
from agent_system.schemas import UserProfile
from agent_system.scout.agent import Scout
from agent_system.synthesizer.agent import Synthesizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("web_ui")

app = FastAPI()

# -------------------------------------------------------------------
# Agent system — built once at process startup and reused across
# requests. Sub-agents hold only a settings reference + a lazily
# constructed Gemini client (no per-request state), so sharing them is
# safe and avoids rebuilding prompts/clients on every message. Only
# the thin Orchestrator wrapper itself is built per request (see
# _run_pipeline) so each run gets its own callback closures and can
# override agentic_replan_enabled independently (the Agent/Workflow
# toggle in the UI).
# -------------------------------------------------------------------
_settings = get_settings()
_scout = Scout(_settings)
_analyst = Analyst(_settings)
_synthesizer = Synthesizer(_settings)
_critic = Critic(_settings)
_session_service = InMemorySessionService()

# chat_id -> {"status": "running"|"done"|"error", "run_id": str,
# "started_at": float, "error": str}. Read by /progress while a
# /messages request for the same chat is in flight on another thread.
_active_runs: dict[int, dict[str, Any]] = {}

STAGE_ORDER = ["intent", "planner", "analyst", "evaluator", "synthesizer", "critic"]
STAGE_LABELS = {
    "intent": "理解意图",
    "planner": "制定检索计划",
    "analyst": "检索并分析文献",
    "evaluator": "评估证据是否充分",
    "synthesizer": "综合生成回答",
    "critic": "事实核查 (Critic)",
}

# -------------------------------------------------------------------
# Database
# -------------------------------------------------------------------
DB_PATH = Path("data/web_ui.sqlite")
DB_PATH.parent.mkdir(parents=True, exist_ok=True)


def get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def _ensure_columns(conn: sqlite3.Connection, table: str, columns: dict[str, str]) -> None:
    existing = {row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
    for name, coltype in columns.items():
        if name not in existing:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {coltype}")


def init_db() -> None:
    with get_db() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS chats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT DEFAULT 'New Chat',
                is_pinned BOOLEAN DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chat_id INTEGER,
                role TEXT,
                content TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(chat_id) REFERENCES chats(id) ON DELETE CASCADE
            )
        """)
        # Additive migration: safe to run against the pre-existing
        # (pre-rewrite) web_ui.sqlite too, not just a fresh DB.
        _ensure_columns(conn, "chats", {"profile_json": "TEXT"})
        _ensure_columns(
            conn,
            "messages",
            {
                "result_json": "TEXT",
                "run_summary_json": "TEXT",
                "is_error": "INTEGER DEFAULT 0",
            },
        )
        conn.commit()


init_db()

# -------------------------------------------------------------------
# Data models
# -------------------------------------------------------------------


class MessageRequest(BaseModel):
    content: str
    agent_mode: bool = True  # True = Agentic (Evaluator + Replan), False = Fixed Workflow baseline


class ProfileRequest(BaseModel):
    interests: list[str]
    role_target: str
    seniority: str


def _default_profile_dict() -> dict[str, Any]:
    return {
        "user_id": "demo_user",
        "interests": ["large language models", "RLHF", "AI safety"],
        "role_target": "researcher",
        "seniority": "PhD student",
        "reading_history": [],
        "feedback_log": [],
    }


def _friendly_error(exc: Exception) -> str:
    text = str(exc)
    name = type(exc).__name__
    if "RESOURCE_EXHAUSTED" in text or " 429" in f" {text}":
        return "遇到 Gemini API 速率限制（429 Too Many Requests），请稍等片刻后重试。"
    if isinstance(exc, ResearchEvidenceUnavailableError):
        return f"没有检索到可用于回答该问题的证据（{text}）。请尝试更换关键词或放宽时间范围。"
    return f"Agent 执行失败：{name}: {text}"


# -------------------------------------------------------------------
# Core logic: run the real pipeline (no CLI shell-out)
# -------------------------------------------------------------------


def _run_pipeline(
    chat_id: int, goal: str, profile: UserProfile, agent_mode: bool
) -> tuple[Any, dict[str, Any]]:
    """Runs synchronously — called via asyncio.to_thread so the event
    loop stays free to serve concurrent /progress polling requests
    while this (potentially 30-90s) pipeline run is in flight."""

    request_settings = dataclasses.replace(_settings, agentic_replan_enabled=agent_mode)
    summary_holder: dict[str, Any] = {}

    def _on_started(run_id: str) -> None:
        _active_runs[chat_id] = {
            "status": "running",
            "run_id": run_id,
            "started_at": time.time(),
        }

    def _on_summary(summary: dict[str, Any]) -> None:
        summary_holder.update(summary)

    orchestrator = Orchestrator(
        request_settings,
        scout=_scout,
        analyst=_analyst,
        synthesizer=_synthesizer,
        critic=_critic,
        session_service=_session_service,
        on_run_started=_on_started,
        on_run_summary=_on_summary,
    )
    verified = orchestrator.run(goal, profile)
    return verified, summary_holder


# -------------------------------------------------------------------
# API Routes
# -------------------------------------------------------------------


@app.get("/", response_class=HTMLResponse)
async def serve_index():
    with open("index.html", encoding="utf-8") as f:
        return f.read()


@app.get("/api/chats")
def get_chats():
    with get_db() as conn:
        chats = conn.execute(
            "SELECT id, title, is_pinned, created_at FROM chats "
            "ORDER BY is_pinned DESC, created_at DESC"
        ).fetchall()
        return [dict(chat) for chat in chats]


@app.post("/api/chats")
def create_chat():
    with get_db() as conn:
        cursor = conn.execute(
            "INSERT INTO chats (profile_json) VALUES (?)",
            (json.dumps(_default_profile_dict(), ensure_ascii=False),),
        )
        conn.commit()
        return {"id": cursor.lastrowid, "title": "New Chat", "is_pinned": 0}


@app.get("/api/chats/{chat_id}")
def get_chat_messages(chat_id: int):
    with get_db() as conn:
        rows = conn.execute(
            "SELECT role, content, result_json, run_summary_json, is_error "
            "FROM messages WHERE chat_id = ? ORDER BY id ASC",
            (chat_id,),
        ).fetchall()
        messages = []
        for row in rows:
            msg: dict[str, Any] = {
                "role": row["role"],
                "content": row["content"],
                "is_error": bool(row["is_error"]),
            }
            if row["result_json"]:
                msg["result"] = json.loads(row["result_json"])
            if row["run_summary_json"]:
                msg["run_summary"] = json.loads(row["run_summary_json"])
            messages.append(msg)
        return messages


@app.delete("/api/chats/{chat_id}")
def delete_chat(chat_id: int):
    with get_db() as conn:
        conn.execute("DELETE FROM chats WHERE id = ?", (chat_id,))
        conn.commit()
    _active_runs.pop(chat_id, None)
    return {"status": "success"}


@app.put("/api/chats/{chat_id}/pin")
def toggle_pin_chat(chat_id: int):
    with get_db() as conn:
        current_status = conn.execute(
            "SELECT is_pinned FROM chats WHERE id = ?", (chat_id,)
        ).fetchone()
        if not current_status:
            raise HTTPException(status_code=404, detail="Chat not found")
        new_status = 0 if current_status[0] else 1
        conn.execute("UPDATE chats SET is_pinned = ? WHERE id = ?", (new_status, chat_id))
        conn.commit()
        return {"is_pinned": new_status}


@app.get("/api/chats/{chat_id}/profile")
def get_profile(chat_id: int):
    with get_db() as conn:
        row = conn.execute(
            "SELECT profile_json FROM chats WHERE id = ?", (chat_id,)
        ).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Chat not found")
        return json.loads(row["profile_json"]) if row["profile_json"] else _default_profile_dict()


@app.put("/api/chats/{chat_id}/profile")
def set_profile(chat_id: int, request: ProfileRequest):
    with get_db() as conn:
        existing = conn.execute(
            "SELECT profile_json FROM chats WHERE id = ?", (chat_id,)
        ).fetchone()
        if not existing:
            raise HTTPException(status_code=404, detail="Chat not found")
        base = (
            json.loads(existing["profile_json"]) if existing["profile_json"] else _default_profile_dict()
        )
        base.update(
            {
                "interests": request.interests,
                "role_target": request.role_target,
                "seniority": request.seniority,
            }
        )
        conn.execute(
            "UPDATE chats SET profile_json = ? WHERE id = ?",
            (json.dumps(base, ensure_ascii=False), chat_id),
        )
        conn.commit()
        return base


@app.get("/api/chats/{chat_id}/progress")
def get_progress(chat_id: int):
    state = _active_runs.get(chat_id)
    if not state:
        return {"status": "idle"}

    status = state.get("status", "running")
    run_id = state.get("run_id", "")
    stages: dict[str, dict[str, Any]] = {
        name: {"label": STAGE_LABELS[name], "calls": 0, "touched": False} for name in STAGE_ORDER
    }
    llm_call_count = 0
    if run_id:
        try:
            summary = summarize_llm_calls(run_id)
        except Exception:
            logger.exception("summarize_llm_calls failed for run_id=%s", run_id)
            summary = {}
        by_component = summary.get("by_component", {}) or {}
        llm_call_count = summary.get("llm_call_count", 0)
        for name in STAGE_ORDER:
            n = int(by_component.get(name, 0))
            stages[name]["calls"] = n
            stages[name]["touched"] = n > 0

    elapsed = time.time() - state.get("started_at", time.time())
    return {
        "status": status,
        "stages": stages,
        "llm_call_count": llm_call_count,
        "elapsed_s": elapsed,
        "error": state.get("error"),
    }


@app.post("/api/chats/{chat_id}/messages")
async def send_message(chat_id: int, request: MessageRequest):
    with get_db() as conn:
        chat_row = conn.execute("SELECT * FROM chats WHERE id = ?", (chat_id,)).fetchone()
        if not chat_row:
            raise HTTPException(status_code=404, detail="Chat not found")

        count = conn.execute(
            "SELECT COUNT(*) FROM messages WHERE chat_id = ?", (chat_id,)
        ).fetchone()[0]
        if count == 0:
            title = request.content[:24] + ("…" if len(request.content) > 24 else "")
            conn.execute("UPDATE chats SET title = ? WHERE id = ?", (title, chat_id))

        conn.execute(
            "INSERT INTO messages (chat_id, role, content) VALUES (?, 'user', ?)",
            (chat_id, request.content),
        )
        conn.commit()

        profile_dict = (
            json.loads(chat_row["profile_json"]) if chat_row["profile_json"] else _default_profile_dict()
        )

    profile = UserProfile.from_dict(profile_dict)

    try:
        verified, summary_holder = await asyncio.to_thread(
            _run_pipeline, chat_id, request.content, profile, request.agent_mode
        )
    except Exception as exc:
        logger.exception("Pipeline run failed for chat_id=%s", chat_id)
        error_text = _friendly_error(exc)
        _active_runs[chat_id] = {"status": "error", "error": error_text}
        with get_db() as conn:
            conn.execute(
                "INSERT INTO messages (chat_id, role, content, is_error) VALUES (?, 'bot', ?, 1)",
                (chat_id, error_text),
            )
            conn.commit()
        return {"role": "bot", "is_error": True, "content": error_text}

    result_dict = verified.to_dict()
    run_summary = dict(summary_holder)
    _active_runs[chat_id] = {"status": "done", "run_id": run_summary.get("run_id", "")}
    title = result_dict.get("draft", {}).get("title") or "Untitled Synthesis"

    with get_db() as conn:
        conn.execute(
            "INSERT INTO messages (chat_id, role, content, result_json, run_summary_json) "
            "VALUES (?, 'bot', ?, ?, ?)",
            (
                chat_id,
                title,
                json.dumps(result_dict, ensure_ascii=False),
                json.dumps(run_summary, ensure_ascii=False),
            ),
        )
        conn.commit()

    return {
        "role": "bot",
        "is_error": False,
        "content": title,
        "result": result_dict,
        "run_summary": run_summary,
    }
