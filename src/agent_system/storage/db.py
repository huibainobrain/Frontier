"""SQLite-backed repository for the agent system.

One table per first-class schema. Most rows store a JSON blob of the
underlying dataclass plus a few denormalised columns we want to filter
or sort by. This keeps the schema flexible (no migrations needed when
a dataclass field is added) while still allowing cheap WHERE clauses.

Connections are short-lived (per call). The sqlite file path comes from
:class:`agent_system.config.Settings`; tests can override by passing
``settings=`` explicitly to any function.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from agent_system.config import Settings, get_settings
from agent_system.schemas import (
    CURRENT_ANALYSIS_SCHEMA_VERSION,
    AgentTrace,
    AnalyzedPost,
    CriticReport,
    RawPost,
    VerifiedSynthesis,
    now_iso,
)

logger = logging.getLogger(__name__)


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS raw_posts (
    post_id           TEXT PRIMARY KEY,
    source            TEXT NOT NULL,
    url               TEXT NOT NULL,
    title             TEXT,
    published_at      TEXT,
    content_type      TEXT,
    fetched_at        TEXT,
    raw_html_hash     TEXT,
    payload_json      TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS ix_raw_posts_source ON raw_posts(source);
CREATE INDEX IF NOT EXISTS ix_raw_posts_published_at ON raw_posts(published_at);

CREATE TABLE IF NOT EXISTS analyzed_posts (
    post_id           TEXT PRIMARY KEY,
    category          TEXT,
    confidence        REAL,
    analyzed_at       TEXT,
    payload_json      TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS ix_analyzed_posts_category ON analyzed_posts(category);

CREATE TABLE IF NOT EXISTS syntheses (
    synthesis_id      TEXT PRIMARY KEY,
    synthesis_type    TEXT,
    title             TEXT,
    final             INTEGER,
    revision_count    INTEGER,
    generated_at      TEXT,
    payload_json      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS critic_reports (
    report_id         TEXT PRIMARY KEY,
    synthesis_id      TEXT,
    num_unsupported   INTEGER,
    revision_needed   INTEGER,
    payload_json      TEXT NOT NULL,
    FOREIGN KEY (synthesis_id) REFERENCES syntheses(synthesis_id)
);

CREATE TABLE IF NOT EXISTS user_feedback (
    feedback_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id           TEXT,
    intent            TEXT,
    synthesis_type    TEXT,
    final             INTEGER,
    revision_count    INTEGER,
    logged_at         TEXT,
    payload_json      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS agent_traces (
    trace_id          TEXT PRIMARY KEY,
    agent_name        TEXT,
    model             TEXT,
    started_at        TEXT,
    ended_at          TEXT,
    input_tokens      INTEGER,
    output_tokens     INTEGER,
    cost_estimate     REAL,
    status            TEXT,
    payload_json      TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS ix_agent_traces_agent_name ON agent_traces(agent_name);
"""


def _settings(settings: Settings | None = None) -> Settings:
    return settings if settings is not None else get_settings()


@contextmanager
def _conn(settings: Settings | None = None) -> Iterator[sqlite3.Connection]:
    s = _settings(settings)
    s.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(s.sqlite_path)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db(settings: Settings | None = None) -> None:
    """Create tables. Idempotent — safe to call on every startup."""

    with _conn(settings) as c:
        c.executescript(SCHEMA_SQL)


# ---------------------------------------------------------------------------
# RawPost
# ---------------------------------------------------------------------------


def save_raw_post(post: RawPost, settings: Settings | None = None) -> None:
    init_db(settings)
    with _conn(settings) as c:
        c.execute(
            """INSERT OR REPLACE INTO raw_posts
                 (post_id, source, url, title, published_at, content_type,
                  fetched_at, raw_html_hash, payload_json)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                post.post_id,
                post.source,
                post.url,
                post.title,
                post.published_at,
                post.content_type,
                post.fetched_at,
                post.raw_html_hash,
                json.dumps(post.to_dict()),
            ),
        )


def get_raw_post(
    post_id: str, settings: Settings | None = None
) -> RawPost | None:
    init_db(settings)
    with _conn(settings) as c:
        row = c.execute(
            "SELECT payload_json FROM raw_posts WHERE post_id = ?", (post_id,)
        ).fetchone()
    if row is None:
        return None
    return RawPost.from_dict(json.loads(row["payload_json"]))


def list_raw_posts(
    source: str | None = None,
    limit: int = 100,
    settings: Settings | None = None,
) -> list[RawPost]:
    init_db(settings)
    with _conn(settings) as c:
        if source:
            rows = c.execute(
                "SELECT payload_json FROM raw_posts WHERE source = ? "
                "ORDER BY published_at DESC LIMIT ?",
                (source, limit),
            ).fetchall()
        else:
            rows = c.execute(
                "SELECT payload_json FROM raw_posts "
                "ORDER BY published_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
    return [RawPost.from_dict(json.loads(r["payload_json"])) for r in rows]


# ---------------------------------------------------------------------------
# AnalyzedPost
# ---------------------------------------------------------------------------


def save_analyzed_post(
    post: AnalyzedPost, settings: Settings | None = None
) -> None:
    init_db(settings)
    with _conn(settings) as c:
        c.execute(
            """INSERT OR REPLACE INTO analyzed_posts
                 (post_id, category, confidence, analyzed_at, payload_json)
               VALUES (?, ?, ?, ?, ?)""",
            (
                post.post_id,
                post.category,
                float(post.confidence),
                post.analyzed_at,
                json.dumps(post.to_dict()),
            ),
        )


def get_analyzed_post(
    post_id: str, settings: Settings | None = None
) -> AnalyzedPost | None:
    """Look up a persisted AnalyzedPost, or None if there isn't one *or*
    the stored one predates CURRENT_ANALYSIS_SCHEMA_VERSION.

    This is the single point both the Analyst's cache
    (analyst.agent.Analyst._get_cached) and RAG hydration
    (storage.vectors.search_similar) go through, so treating a stale
    row as "not found" here — rather than in each of those two callers
    separately — makes both self-heal the same way for free: a miss
    triggers a fresh analyze() call, whose result upserts over the old
    row (see analyst.rag.store_analysis), so the row is current from
    then on. No separate migration needed.
    """

    init_db(settings)
    with _conn(settings) as c:
        row = c.execute(
            "SELECT payload_json FROM analyzed_posts WHERE post_id = ?", (post_id,)
        ).fetchone()
    if row is None:
        return None
    post = AnalyzedPost.from_dict(json.loads(row["payload_json"]))
    if post.schema_version < CURRENT_ANALYSIS_SCHEMA_VERSION:
        logger.info(
            "analyzed_posts: post_id=%s has schema_version=%d < current=%d; "
            "treating as stale (cache miss).",
            post_id,
            post.schema_version,
            CURRENT_ANALYSIS_SCHEMA_VERSION,
        )
        return None
    return post


def list_analyzed_posts(
    category: str | None = None,
    min_confidence: float = 0.0,
    limit: int = 100,
    settings: Settings | None = None,
) -> list[AnalyzedPost]:
    init_db(settings)
    sql = "SELECT payload_json FROM analyzed_posts WHERE confidence >= ?"
    params: list[Any] = [min_confidence]
    if category:
        sql += " AND category = ?"
        params.append(category)
    sql += " ORDER BY analyzed_at DESC LIMIT ?"
    params.append(limit)
    with _conn(settings) as c:
        rows = c.execute(sql, params).fetchall()
    return [AnalyzedPost.from_dict(json.loads(r["payload_json"])) for r in rows]


# ---------------------------------------------------------------------------
# Syntheses + critic reports
# ---------------------------------------------------------------------------


def save_synthesis(
    verified: VerifiedSynthesis, settings: Settings | None = None
) -> str:
    """Persist a VerifiedSynthesis (and its critic report). Returns the synthesis_id."""

    init_db(settings)
    synthesis_id = uuid.uuid4().hex
    with _conn(settings) as c:
        c.execute(
            """INSERT INTO syntheses
                 (synthesis_id, synthesis_type, title, final, revision_count,
                  generated_at, payload_json)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                synthesis_id,
                verified.draft.synthesis_type,
                verified.draft.title,
                int(verified.final),
                int(verified.revision_count),
                verified.draft.generated_at,
                json.dumps(verified.to_dict()),
            ),
        )
        report = verified.critic_report
        c.execute(
            """INSERT INTO critic_reports
                 (report_id, synthesis_id, num_unsupported, revision_needed,
                  payload_json)
               VALUES (?, ?, ?, ?, ?)""",
            (
                uuid.uuid4().hex,
                synthesis_id,
                int(report.num_unsupported),
                int(report.revision_needed),
                json.dumps(report.to_dict()),
            ),
        )
    return synthesis_id


def get_synthesis(
    synthesis_id: str, settings: Settings | None = None
) -> VerifiedSynthesis | None:
    init_db(settings)
    with _conn(settings) as c:
        row = c.execute(
            "SELECT payload_json FROM syntheses WHERE synthesis_id = ?",
            (synthesis_id,),
        ).fetchone()
    if row is None:
        return None
    return VerifiedSynthesis.from_dict(json.loads(row["payload_json"]))


def save_critic_report(
    report: CriticReport,
    synthesis_id: str | None = None,
    settings: Settings | None = None,
) -> str:
    init_db(settings)
    report_id = uuid.uuid4().hex
    with _conn(settings) as c:
        c.execute(
            """INSERT INTO critic_reports
                 (report_id, synthesis_id, num_unsupported, revision_needed,
                  payload_json)
               VALUES (?, ?, ?, ?, ?)""",
            (
                report_id,
                synthesis_id,
                int(report.num_unsupported),
                int(report.revision_needed),
                json.dumps(report.to_dict()),
            ),
        )
    return report_id


# ---------------------------------------------------------------------------
# User feedback + agent traces
# ---------------------------------------------------------------------------


def save_user_feedback(
    record: dict, settings: Settings | None = None
) -> int:
    """Persist a freeform feedback dict. Required keys: user_id, intent,
    synthesis_type, final, revision_count, logged_at."""

    init_db(settings)
    with _conn(settings) as c:
        cursor = c.execute(
            """INSERT INTO user_feedback
                 (user_id, intent, synthesis_type, final, revision_count,
                  logged_at, payload_json)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                str(record.get("user_id", "")),
                str(record.get("intent", "")),
                str(record.get("synthesis_type", "")),
                int(bool(record.get("final", False))),
                int(record.get("revision_count", 0)),
                str(record.get("logged_at", now_iso())),
                json.dumps(record),
            ),
        )
        return int(cursor.lastrowid or 0)


def save_agent_trace(
    trace: AgentTrace, settings: Settings | None = None
) -> None:
    init_db(settings)
    with _conn(settings) as c:
        c.execute(
            """INSERT OR REPLACE INTO agent_traces
                 (trace_id, agent_name, model, started_at, ended_at,
                  input_tokens, output_tokens, cost_estimate, status,
                  payload_json)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                trace.trace_id,
                trace.agent_name,
                trace.model,
                trace.started_at,
                trace.ended_at,
                int(trace.input_tokens),
                int(trace.output_tokens),
                float(trace.cost_estimate),
                trace.status,
                json.dumps(trace.to_dict()),
            ),
        )


def list_agent_traces(
    agent_name: str | None = None,
    limit: int = 200,
    settings: Settings | None = None,
) -> list[AgentTrace]:
    init_db(settings)
    with _conn(settings) as c:
        if agent_name:
            rows = c.execute(
                "SELECT payload_json FROM agent_traces WHERE agent_name = ? "
                "ORDER BY started_at DESC LIMIT ?",
                (agent_name, limit),
            ).fetchall()
        else:
            rows = c.execute(
                "SELECT payload_json FROM agent_traces "
                "ORDER BY started_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
    return [AgentTrace.from_dict(json.loads(r["payload_json"])) for r in rows]
