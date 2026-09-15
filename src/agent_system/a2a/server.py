"""A2A server — exposes our agents via the Agent2Agent protocol.

Provides two HTTP surfaces:

1. ``GET /.well-known/agent.json`` — AgentCard discovery (returns
   capability metadata so remote agents know what we can do).
2. ``POST /`` — JSON-RPC 2.0 endpoint accepting ``message/send``
   requests.  The server creates a Task, runs the requested skill
   (currently: literature search via Scout), and returns the result
   as an Artifact.

Usage::

    from agent_system.a2a.server import A2AServer
    server = A2AServer(host="0.0.0.0", port=8080)
    server.run()          # blocking
    # or in tests:
    app = server.app      # ASGI app for httpx.AsyncClient
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import UTC, datetime
from typing import Any

from agent_system.a2a.models import (
    AgentCapability,
    AgentCard,
    AgentSkill,
    Artifact,
    Part,
    TaskState,
)

logger = logging.getLogger(__name__)


def _now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _new_id() -> str:
    return uuid.uuid4().hex[:12]


# ---------------------------------------------------------------------------
# AgentCard
# ---------------------------------------------------------------------------

AGENT_CARD = AgentCard(
    name="FrontierLitAgent",
    description=(
        "Multi-agent system that monitors frontier AI-lab publications "
        "and produces personalised synthesis (digests, trackers, reading plans)."
    ),
    url="http://localhost:8080",
    version="0.1.0",
    capabilities=AgentCapability(streaming=False, push_notifications=False),
    skills=[
        AgentSkill(
            id="literature_search",
            name="Literature Search",
            description="Search arXiv, Semantic Scholar, OpenAlex and lab RSS feeds for recent AI papers/posts.",
            tags=["search", "arxiv", "AI", "literature"],
            examples=[
                "Find recent papers on RLHF",
                "What has Anthropic published this week?",
            ],
        ),
        AgentSkill(
            id="digest",
            name="Weekly Digest",
            description="Produce a weekly digest of frontier AI lab publications.",
            tags=["synthesis", "digest", "summary"],
            examples=["Give me this week's AI digest"],
        ),
    ],
)


# ---------------------------------------------------------------------------
# Task store (in-memory, same pattern as ADK InMemorySessionService)
# ---------------------------------------------------------------------------

_TASKS: dict[str, dict[str, Any]] = {}


def _save_task(task_id: str, data: dict[str, Any]) -> None:
    _TASKS[task_id] = data


def _get_task(task_id: str) -> dict[str, Any] | None:
    return _TASKS.get(task_id)


# ---------------------------------------------------------------------------
# Request handlers
# ---------------------------------------------------------------------------


def handle_agent_card() -> dict[str, Any]:
    """Return the AgentCard for ``/.well-known/agent.json``."""
    return AGENT_CARD.to_dict()


def handle_message_send(params: dict[str, Any]) -> dict[str, Any]:
    """Process a ``message/send`` JSON-RPC request.

    Extracts the user's text from the message parts, creates a Task,
    runs the Scout fetch pipeline, and returns the results as an
    Artifact wrapped in a completed Task.
    """

    # 1. Parse the incoming message
    message = params.get("message", {})
    parts = message.get("parts", [])
    user_text = ""
    for p in parts:
        if "text" in p:
            user_text = p["text"]
            break

    if not user_text:
        return _error_response(-32602, "No text part found in message")

    # 2. Create task
    task_id = _new_id()
    context_id = message.get("contextId", _new_id())

    task_data: dict[str, Any] = {
        "id": task_id,
        "contextId": context_id,
        "status": {
            "state": TaskState.WORKING,
            "timestamp": _now_iso(),
        },
        "artifacts": [],
        "history": [],
    }
    _save_task(task_id, task_data)

    # 3. Execute — run Scout search
    try:
        results = _execute_search(user_text)
        task_data["status"] = {
            "state": TaskState.COMPLETED,
            "timestamp": _now_iso(),
        }
        task_data["artifacts"] = [
            Artifact(
                artifact_id=_new_id(),
                name="search_results",
                description=f"Literature search results for: {user_text}",
                parts=[Part(text=json.dumps(results, ensure_ascii=False, indent=2))],
            ).to_dict()
        ]
    except Exception as exc:
        logger.exception("A2A task %s failed", task_id)
        task_data["status"] = {
            "state": TaskState.FAILED,
            "message": {
                "messageId": _new_id(),
                "role": "agent",
                "parts": [{"text": f"Search failed: {exc}"}],
            },
            "timestamp": _now_iso(),
        }

    _save_task(task_id, task_data)
    return task_data


def handle_get_task(params: dict[str, Any]) -> dict[str, Any]:
    """Process a ``tasks/get`` JSON-RPC request."""

    task_id = params.get("id", "")
    task = _get_task(task_id)
    if task is None:
        return _error_response(-32001, f"Task {task_id} not found")
    return task


# ---------------------------------------------------------------------------
# Search execution (wraps Scout)
# ---------------------------------------------------------------------------


def _execute_search(query: str) -> list[dict[str, str]]:
    """Run a literature search using the Scout module."""

    from agent_system.config import get_settings
    from agent_system.schemas import UserProfile
    from agent_system.scout.agent import Scout

    settings = get_settings()
    scout = Scout(settings)
    profile = UserProfile(
        user_id="a2a_client",
        interests=[query],
        role_target="researcher",
        seniority="unknown",
    )
    plan = scout.plan_sources(query, profile)
    posts = scout.fetch(plan)

    return [
        {
            "post_id": p.post_id,
            "source": p.source,
            "title": p.title,
            "url": p.url,
            "published_at": p.published_at,
            "content_preview": p.content[:300],
        }
        for p in posts
    ]


# ---------------------------------------------------------------------------
# JSON-RPC helpers
# ---------------------------------------------------------------------------


def _error_response(code: int, message: str) -> dict[str, Any]:
    return {"error": {"code": code, "message": message}}


# ---------------------------------------------------------------------------
# FastAPI app factory
# ---------------------------------------------------------------------------


def create_app():
    """Create the FastAPI ASGI app for the A2A server."""

    from fastapi import FastAPI

    app = FastAPI(title="FrontierLitAgent A2A Server", version="0.1.0")

    # JSON-RPC dispatch table
    RPC_METHODS: dict[str, Any] = {
        "message/send": handle_message_send,
        "tasks/get": handle_get_task,
    }

    @app.get("/.well-known/agent.json")
    async def agent_card_endpoint():
        return handle_agent_card()

    @app.post("/")
    async def jsonrpc_endpoint(body: dict[str, Any]):
        jsonrpc_version = body.get("jsonrpc", "2.0")
        method = body.get("method", "")
        params = body.get("params", {})
        req_id = body.get("id")

        if method not in RPC_METHODS:
            return {
                "jsonrpc": jsonrpc_version,
                "id": req_id,
                "error": {"code": -32601, "message": f"Method not found: {method}"},
            }

        result = RPC_METHODS[method](params)

        # Check if the handler returned an error
        if "error" in result:
            return {"jsonrpc": jsonrpc_version, "id": req_id, "error": result["error"]}

        return {"jsonrpc": jsonrpc_version, "id": req_id, "result": result}

    return app


# ---------------------------------------------------------------------------
# Server runner
# ---------------------------------------------------------------------------


class A2AServer:
    """Thin wrapper to start the A2A server."""

    def __init__(self, host: str = "0.0.0.0", port: int = 8080) -> None:
        self.host = host
        self.port = port
        self.app = create_app()

    def run(self) -> None:
        import uvicorn

        logger.info("Starting A2A server on %s:%d", self.host, self.port)
        uvicorn.run(self.app, host=self.host, port=self.port)