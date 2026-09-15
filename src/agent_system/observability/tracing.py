"""Observability: token tracking, traces, optional Langfuse.

Three primitives that wrap LLM-shaped work and write a one-line JSON
record to ``data/token_log.jsonl`` (path comes from
:class:`agent_system.config.Settings`):

* ``@traced(agent_name)`` — decorator that times a function call
  (sync or async) and writes one ``AgentTrace``-shaped record on
  completion.
* ``with trace_span(name):`` — context manager for finer-grained
  spans (e.g. wrapping a sub-step inside ``Orchestrator.run``).
* ``record_llm_call(...)`` — explicit sink the LLM caller invokes
  with token counts when it has them. Use this from agent code
  immediately after a Gemini call; the decorators don't know the
  token cost on their own.

If Langfuse keys are configured the same record is also pushed to the
Langfuse cloud (best-effort — failures here are logged, not raised).

``current_budget()`` aggregates the JSONL file into a small dict the
demo notebook can render.

Usage::

    from agent_system.observability.tracing import traced, trace_span

    @traced("analyst.analyze")
    def analyze(self, post): ...

    with trace_span("scout.fetch"):
        ...
"""

from __future__ import annotations

import asyncio
import contextlib
import functools
import json
import logging
import time
import uuid
from collections.abc import Callable, Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _token_log_path() -> Path:
    """Resolved lazily so tests can monkey-patch Settings before import."""

    from agent_system.config import get_settings

    return get_settings().token_log_path


def _append_record(record: dict) -> None:
    try:
        path = _token_log_path()
    except Exception:
        # Settings unavailable (e.g. in tests that bypass env loading) — skip
        # the local file but don't break callers.
        logger.debug("token log path unavailable; skipping append")
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")
    except Exception:
        logger.exception("failed to append to token log at %s", path)


def _maybe_push_langfuse(record: dict) -> None:
    try:
        from agent_system.config import get_settings

        settings = get_settings()
    except Exception:
        return
    if not settings.has_langfuse:
        return
    try:
        from langfuse import Langfuse  # type: ignore

        client = Langfuse(
            public_key=settings.langfuse_public_key,
            secret_key=settings.langfuse_secret_key,
            host=settings.langfuse_host,
        )
        client.trace(
            name=record.get("agent_name", "agent"),
            input=record.get("input"),
            output=record.get("output"),
            metadata=record,
        )
    except Exception:
        logger.exception("Langfuse push failed; continuing with local log only")


def _emit(record: dict) -> None:
    _append_record(record)
    _maybe_push_langfuse(record)


def _build_record(
    *,
    agent_name: str,
    started_at: str,
    duration_ms: int,
    status: str,
    kind: str,
    input_tokens: int = 0,
    output_tokens: int = 0,
    model: str = "",
    cost_estimate: float = 0.0,
    extra: dict | None = None,
) -> dict:
    record = {
        "trace_id": uuid.uuid4().hex,
        "agent_name": agent_name,
        "started_at": started_at,
        "ended_at": _now_iso(),
        "duration_ms": duration_ms,
        "input_tokens": int(input_tokens),
        "output_tokens": int(output_tokens),
        "model": model,
        "cost_estimate": float(cost_estimate),
        "status": status,
        "kind": kind,
    }
    if extra:
        record.update(extra)
    return record


# ---------------------------------------------------------------------------
# @traced(...)
# ---------------------------------------------------------------------------


def traced(agent_name: str) -> Callable:
    """Decorator that records a trace on function entry/exit.

    Works with both sync and ``async def`` functions. Token counts are
    not known here — the decorated function should additionally call
    :func:`record_llm_call` with the real numbers when it has them.
    """

    def decorator(func: Callable) -> Callable:
        if asyncio.iscoroutinefunction(func):

            @functools.wraps(func)
            async def awrapper(*args: Any, **kwargs: Any) -> Any:
                started = _now_iso()
                t0 = time.perf_counter()
                status = "ok"
                try:
                    return await func(*args, **kwargs)
                except Exception:
                    status = "error"
                    raise
                finally:
                    _emit(
                        _build_record(
                            agent_name=agent_name,
                            started_at=started,
                            duration_ms=int((time.perf_counter() - t0) * 1000),
                            status=status,
                            kind="call",
                        )
                    )

            return awrapper
        else:

            @functools.wraps(func)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                started = _now_iso()
                t0 = time.perf_counter()
                status = "ok"
                try:
                    return func(*args, **kwargs)
                except Exception:
                    status = "error"
                    raise
                finally:
                    _emit(
                        _build_record(
                            agent_name=agent_name,
                            started_at=started,
                            duration_ms=int((time.perf_counter() - t0) * 1000),
                            status=status,
                            kind="call",
                        )
                    )

            return wrapper

    return decorator


# ---------------------------------------------------------------------------
# trace_span context manager
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def trace_span(name: str) -> Iterator[None]:
    """Lightweight span around a block of work."""

    started = _now_iso()
    t0 = time.perf_counter()
    status = "ok"
    try:
        yield
    except Exception:
        status = "error"
        raise
    finally:
        _emit(
            _build_record(
                agent_name=name,
                started_at=started,
                duration_ms=int((time.perf_counter() - t0) * 1000),
                status=status,
                kind="span",
            )
        )


# ---------------------------------------------------------------------------
# record_llm_call — explicit token sink
# ---------------------------------------------------------------------------


def record_llm_call(
    agent_name: str,
    model: str,
    input_tokens: int,
    output_tokens: int,
    *,
    duration_ms: int = 0,
    cost_estimate: float = 0.0,
    status: str = "ok",
    extra: dict | None = None,
) -> None:
    """Append a token-cost line for one LLM call.

    ``agent_system.llm_retry`` calls this automatically for every real
    LLM call that goes through ``call_with_retry``/
    ``async_call_with_retry`` (i.e. every one of Intent/Planner/
    Evaluator/Analyst/Synthesizer/Critic/target-recovery) — this is
    still exposed directly for any caller that needs to record a token
    count outside that path.
    """

    _emit(
        _build_record(
            agent_name=agent_name,
            started_at=_now_iso(),
            duration_ms=duration_ms,
            status=status,
            kind="llm_call",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model=model,
            cost_estimate=cost_estimate,
            extra=extra,
        )
    )


# ---------------------------------------------------------------------------
# current_budget
# ---------------------------------------------------------------------------


def _iter_log_records() -> Iterator[dict]:
    try:
        path = _token_log_path()
    except Exception:
        return
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def current_budget() -> dict:
    """Aggregate the JSONL token log for the demo notebook."""

    total_tokens = 0
    total_cost = 0.0
    by_agent: dict[str, int] = {}
    by_model: dict[str, int] = {}
    llm_call_count = 0

    for rec in _iter_log_records():
        # A "span" (trace_span/@traced) is timing instrumentation around
        # a step, not an LLM call — only "llm_call" records (written by
        # record_llm_call, which agent_system.llm_retry calls for every
        # real Gemini/ADK call) count toward LLM Call Count. Token/cost
        # totals don't need the same filter: a span record always
        # carries input_tokens=output_tokens=cost_estimate=0.
        if rec.get("kind") == "llm_call":
            llm_call_count += 1
        tokens = int(rec.get("input_tokens", 0)) + int(rec.get("output_tokens", 0))
        cost = float(rec.get("cost_estimate", 0.0))
        total_tokens += tokens
        total_cost += cost
        agent = rec.get("agent_name", "?")
        model = rec.get("model") or "?"
        by_agent[agent] = by_agent.get(agent, 0) + tokens
        by_model[model] = by_model.get(model, 0) + tokens

    return {
        # Kept for backward compatibility with existing callers — now
        # correctly scoped to llm_call records only (used to also count
        # "span"/"call" records, over-counting LLM Call Count).
        "n_calls": llm_call_count,
        "llm_call_count": llm_call_count,
        "total_tokens": total_tokens,
        "total_cost_estimate": round(total_cost, 6),
        "by_agent": by_agent,
        "by_model": by_model,
    }


def summarize_llm_calls(run_id: str) -> dict:
    """Aggregate this run's ``llm_call`` telemetry records (see
    ``agent_system.llm_retry``'s automatic recording) by ``extra.
    run_id``. Used by Orchestrator to build its own run-level summary
    (which also needs pipeline facts — intent, research_rounds,
    retrieved_posts, ... — that this module has no way to know)."""

    llm_call_count = 0
    input_tokens = 0
    output_tokens = 0
    fallback_count = 0
    by_component: dict[str, int] = {}

    for rec in _iter_log_records():
        if rec.get("kind") != "llm_call":
            continue
        if rec.get("run_id") != run_id:
            continue
        llm_call_count += 1
        input_tokens += int(rec.get("input_tokens", 0))
        output_tokens += int(rec.get("output_tokens", 0))
        if rec.get("status") == "failed":
            fallback_count += 1
        component = rec.get("agent_name", "?")
        by_component[component] = by_component.get(component, 0) + 1

    return {
        "llm_call_count": llm_call_count,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "fallback_count": fallback_count,
        "by_component": by_component,
    }
