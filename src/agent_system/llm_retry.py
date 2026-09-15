"""Shared Gemini/LLM call retry policy — used by every module that
calls Gemini directly (Analyst, Synthesizer, Critic, and the
Orchestrator's own comparison-target-recovery call) or through the ADK
Runner (Orchestrator._invoke_agent, backing Intent/Planner/Evaluator).

One retry policy lives here so it isn't copied 5-6 times across
modules. What it deliberately does NOT own: what happens once retries
are exhausted. That is product-level, differs per module (fall back to
a default plan, skip one article, stop the research loop, fail the
whole task, mark "verification_unavailable", ...), and stays with each
caller — see each module's own docstring for its specific fallback.

Two entry points, same policy, different loop-control primitive:

* :func:`call_with_retry` — sync, for the direct
  ``client.models.generate_content(...)`` callers (Analyst, Synthesizer,
  Critic, Orchestrator's target recovery).
* :func:`async_call_with_retry` — async, for
  ``Orchestrator._invoke_agent`` (the ADK ``Runner.run_async`` path),
  which is async end to end and must not block the event loop with a
  blocking ``time.sleep`` during backoff.

Retryable errors are transient infra failures only: 429 (rate limit)
and 5xx (server-side). Anything else — a malformed request, an auth
failure, a response that doesn't parse — is not retried; retrying a
guaranteed-repeat failure just wastes round-trips.

Also the single place real LLM-call telemetry gets recorded (see
:func:`note_usage`, :func:`set_current_run_id`, and
``agent_system.observability.tracing.record_llm_call``) — every caller
routes through here, so this is where a call's real token usage,
latency, attempt count, and final status get logged once, rather than
duplicated at each of the 7 call sites that use this module.
"""

from __future__ import annotations

import asyncio
import contextvars
import logging
import time
from collections.abc import Awaitable, Callable
from typing import Any, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")

# A 429 is a 4xx in google.genai's own ClientError/ServerError split, so
# this checks the numeric status directly rather than the exception
# class — the two "retryable" families (rate limit, server error) don't
# line up with that class boundary.
RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}
MAX_LLM_ATTEMPTS = 3
RETRY_BACKOFF_SECONDS = (1.0, 2.0)  # wait before attempt 2, before attempt 3

# Per-run correlation id for telemetry — set once near the top of
# Orchestrator.run_async via set_current_run_id (reusing the ADK
# session_id, already unique per run), read implicitly here so the 7
# call sites don't need to thread a run_id through every call_with_retry
# / async_call_with_retry invocation.
_run_id_ctx: contextvars.ContextVar[str] = contextvars.ContextVar(
    "_run_id_ctx", default=""
)
# Real token usage for the attempt currently in flight, set by the `fn`
# closure itself (see note_usage) right after it has a raw SDK response
# — never guessed. Reset before each attempt so a prior attempt's
# numbers can't leak onto a different one.
_usage_ctx: contextvars.ContextVar[tuple[int, int] | None] = contextvars.ContextVar(
    "_usage_ctx", default=None
)


def set_current_run_id(run_id: str) -> None:
    """Tag every LLM-call telemetry record produced during this run
    (however deep in the call stack) with *run_id*."""

    _run_id_ctx.set(run_id)


def note_usage(input_tokens: int | None, output_tokens: int | None) -> None:
    """Report real token usage for the attempt in progress — call this
    from inside an ``fn`` passed to :func:`call_with_retry` /
    :func:`async_call_with_retry`, right after getting a raw SDK
    response (e.g. ``response.usage_metadata.prompt_token_count`` /
    ``.candidates_token_count``, or an ADK event's ``usage_metadata``).
    Optional: a closure that never calls this just means that call's
    telemetry has no token counts (0, never a guessed/estimated number)
    — never call this with made-up numbers.
    """

    if input_tokens is None and output_tokens is None:
        return
    _usage_ctx.set((int(input_tokens or 0), int(output_tokens or 0)))


def note_usage_from_gemini_response(response: Any) -> None:
    """Convenience for the common case (Analyst/Synthesizer/Critic/
    target-recovery, all direct ``client.models.generate_content(...)``
    callers): pull ``prompt_token_count``/``candidates_token_count`` off
    a raw ``google.genai`` response's ``usage_metadata`` and report them
    via :func:`note_usage` — a one-line no-op when the response carries
    no usage_metadata (never fabricates a count itself)."""

    usage = getattr(response, "usage_metadata", None)
    if usage is None:
        return
    note_usage(
        getattr(usage, "prompt_token_count", None),
        getattr(usage, "candidates_token_count", None),
    )


def _elapsed_ms(start: float) -> int:
    return int((time.perf_counter() - start) * 1000)


def _record_telemetry(
    component: str, model: str, attempt: int, latency_ms: int, status: str
) -> None:
    """Best-effort — telemetry must never be able to break a real LLM
    call. Errors are logged, not raised."""

    try:
        from agent_system.observability.tracing import record_llm_call

        usage = _usage_ctx.get()
        input_tokens, output_tokens = usage if usage else (0, 0)
        record_llm_call(
            component,
            model,
            input_tokens,
            output_tokens,
            duration_ms=latency_ms,
            status=status,
            extra={"run_id": _run_id_ctx.get(), "attempt_count": attempt},
        )
    except Exception:
        logger.exception("component=%s failed to record LLM telemetry", component)


def is_retryable_gemini_error(exc: Exception) -> bool:
    code = getattr(exc, "code", None)
    return isinstance(code, int) and code in RETRYABLE_STATUS_CODES


def _log_attempt_failure(component: str, model: str, attempt: int, exc: Exception) -> bool:
    """Logs component/model/attempt/error_type/retryable — never the
    prompt content or any credential — and returns whether *exc* is
    retryable."""

    retryable = is_retryable_gemini_error(exc)
    logger.warning(
        "component=%s model=%s attempt=%d/%d error_type=%s retryable=%s",
        component,
        model,
        attempt,
        MAX_LLM_ATTEMPTS,
        type(exc).__name__,
        retryable,
    )
    return retryable


def _log_recovered(component: str, model: str, attempt: int) -> None:
    if attempt > 1:
        logger.info(
            "component=%s model=%s attempt=%d/%d final_status=recovered",
            component,
            model,
            attempt,
            MAX_LLM_ATTEMPTS,
        )


def _log_exhausted(component: str, model: str, exc: Exception) -> None:
    logger.error(
        "component=%s model=%s attempts=%d final_status=failed error_type=%s",
        component,
        model,
        MAX_LLM_ATTEMPTS,
        type(exc).__name__,
    )


def call_with_retry(component: str, model: str, fn: Callable[[], T]) -> T:
    """Call ``fn()``, retrying transient (429/5xx) errors with a short
    backoff (blocking ``time.sleep`` — for sync callers only).

    Raises the final exception when every attempt fails, or immediately
    when the first failure isn't retryable. Callers decide what
    "exhausted" means for them — this function only ever retries or
    raises, it never returns a degraded/fallback value itself.
    """

    last_exc: Exception | None = None
    start = time.perf_counter()
    attempt = 0
    for attempt in range(1, MAX_LLM_ATTEMPTS + 1):
        _usage_ctx.set(None)
        try:
            result = fn()
            _log_recovered(component, model, attempt)
            _record_telemetry(component, model, attempt, _elapsed_ms(start), "ok")
            return result
        except Exception as exc:
            last_exc = exc
            retryable = _log_attempt_failure(component, model, attempt, exc)
            if not retryable or attempt == MAX_LLM_ATTEMPTS:
                break
            time.sleep(RETRY_BACKOFF_SECONDS[attempt - 1])

    assert last_exc is not None  # loop always runs >=1 iteration
    _log_exhausted(component, model, last_exc)
    _record_telemetry(component, model, attempt, _elapsed_ms(start), "failed")
    raise last_exc


async def async_call_with_retry(
    component: str, model: str, fn: Callable[[], Awaitable[T]]
) -> T:
    """Async twin of :func:`call_with_retry` — same policy,
    ``asyncio.sleep`` instead of ``time.sleep`` so backoff doesn't block
    the event loop. For the ADK-driven callers (Orchestrator's
    Intent/Planner/Evaluator, which run through ``Runner.run_async``)."""

    last_exc: Exception | None = None
    start = time.perf_counter()
    attempt = 0
    for attempt in range(1, MAX_LLM_ATTEMPTS + 1):
        _usage_ctx.set(None)
        try:
            result = await fn()
            _log_recovered(component, model, attempt)
            _record_telemetry(component, model, attempt, _elapsed_ms(start), "ok")
            return result
        except Exception as exc:
            last_exc = exc
            retryable = _log_attempt_failure(component, model, attempt, exc)
            if not retryable or attempt == MAX_LLM_ATTEMPTS:
                break
            await asyncio.sleep(RETRY_BACKOFF_SECONDS[attempt - 1])

    assert last_exc is not None
    _log_exhausted(component, model, last_exc)
    _record_telemetry(component, model, attempt, _elapsed_ms(start), "failed")
    raise last_exc
