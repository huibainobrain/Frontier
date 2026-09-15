"""Tests for the shared LLM call retry policy (agent_system.llm_retry).

This is the mechanism every module (Analyst, Synthesizer, Critic, and
the Orchestrator's ADK-driven Intent/Planner/Evaluator calls) delegates
to — tested once here in isolation; each module's own tests then only
need to prove they're correctly wired to it (see test_analyst.py,
test_critic.py, test_orchestrator.py) rather than re-proving the
retry/backoff algorithm itself.
"""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from google.genai import errors as genai_errors

from agent_system.llm_retry import (
    MAX_LLM_ATTEMPTS,
    async_call_with_retry,
    call_with_retry,
    is_retryable_gemini_error,
    note_usage,
)


def _server_error(code: int = 503) -> genai_errors.ServerError:
    return genai_errors.ServerError(code, {"message": "high demand"}, None)


def _client_error(code: int = 400) -> genai_errors.ClientError:
    return genai_errors.ClientError(code, {"message": "bad request"}, None)


# ---------------------------------------------------------------------------
# is_retryable_gemini_error
# ---------------------------------------------------------------------------


def test_retryable_status_codes():
    assert is_retryable_gemini_error(_server_error(503)) is True
    assert is_retryable_gemini_error(_server_error(500)) is True
    assert is_retryable_gemini_error(_server_error(502)) is True
    assert is_retryable_gemini_error(_server_error(504)) is True
    assert is_retryable_gemini_error(_client_error(429)) is True  # rate limit


def test_non_retryable_status_codes():
    assert is_retryable_gemini_error(_client_error(400)) is False
    assert is_retryable_gemini_error(_client_error(401)) is False
    assert is_retryable_gemini_error(_client_error(404)) is False


def test_non_api_error_is_not_retryable():
    assert is_retryable_gemini_error(ValueError("not an API error")) is False
    assert is_retryable_gemini_error(TypeError("bug")) is False


# ---------------------------------------------------------------------------
# call_with_retry (sync)
# ---------------------------------------------------------------------------


def test_call_with_retry_succeeds_first_attempt_no_sleep():
    fn = MagicMock(return_value="ok")
    with patch("agent_system.llm_retry.time.sleep") as mock_sleep:
        result = call_with_retry("test", "model-x", fn)
    assert result == "ok"
    fn.assert_called_once()
    mock_sleep.assert_not_called()


def test_call_with_retry_recovers_after_one_transient_failure():
    fn = MagicMock(side_effect=[_server_error(503), "ok"])
    with patch("agent_system.llm_retry.time.sleep") as mock_sleep:
        result = call_with_retry("test", "model-x", fn)
    assert result == "ok"
    assert fn.call_count == 2
    mock_sleep.assert_called_once_with(1.0)


def test_call_with_retry_raises_after_exhausting_all_attempts():
    fn = MagicMock(side_effect=[_server_error(503)] * MAX_LLM_ATTEMPTS)
    with patch("agent_system.llm_retry.time.sleep") as mock_sleep:
        with pytest.raises(genai_errors.ServerError):
            call_with_retry("test", "model-x", fn)
    assert fn.call_count == MAX_LLM_ATTEMPTS
    assert mock_sleep.call_count == MAX_LLM_ATTEMPTS - 1


def test_call_with_retry_does_not_retry_non_retryable_error():
    fn = MagicMock(side_effect=_client_error(400))
    with patch("agent_system.llm_retry.time.sleep") as mock_sleep:
        with pytest.raises(genai_errors.ClientError):
            call_with_retry("test", "model-x", fn)
    fn.assert_called_once()
    mock_sleep.assert_not_called()


# ---------------------------------------------------------------------------
# async_call_with_retry
# ---------------------------------------------------------------------------


def test_async_call_with_retry_recovers_after_one_transient_failure():
    calls = {"n": 0}

    async def fn():
        calls["n"] += 1
        if calls["n"] == 1:
            raise _server_error(503)
        return "ok"

    with patch("agent_system.llm_retry.asyncio.sleep", new=AsyncMock()) as mock_sleep:
        result = asyncio.run(async_call_with_retry("test", "model-x", fn))

    assert result == "ok"
    assert calls["n"] == 2
    mock_sleep.assert_called_once_with(1.0)


def test_async_call_with_retry_raises_after_exhausting_all_attempts():
    async def fn():
        raise _server_error(503)

    with patch("agent_system.llm_retry.asyncio.sleep", new=AsyncMock()):
        with pytest.raises(genai_errors.ServerError):
            asyncio.run(async_call_with_retry("test", "model-x", fn))


def test_async_call_with_retry_does_not_retry_non_retryable_error():
    calls = {"n": 0}

    async def fn():
        calls["n"] += 1
        raise _client_error(400)

    with patch("agent_system.llm_retry.asyncio.sleep", new=AsyncMock()) as mock_sleep:
        with pytest.raises(genai_errors.ClientError):
            asyncio.run(async_call_with_retry("test", "model-x", fn))

    assert calls["n"] == 1
    mock_sleep.assert_not_called()


# ---------------------------------------------------------------------------
# Eval Readiness: real LLM-call telemetry (component/model/tokens/
# latency/attempt_count/status), recorded once per call_with_retry /
# async_call_with_retry invocation
# ---------------------------------------------------------------------------


def _read_llm_call_records(tmp_path) -> list[dict]:
    log_path = tmp_path / "token_log.jsonl"
    if not log_path.exists():
        return []
    records = [json.loads(line) for line in log_path.read_text().splitlines() if line.strip()]
    return [r for r in records if r.get("kind") == "llm_call"]


def test_call_with_retry_records_real_telemetry_on_success(tmp_path):
    """Test C4: a successful call records component/model/real token
    usage (via note_usage — never a guessed number)/latency/
    attempt_count/status."""

    def fn():
        note_usage(120, 40)
        return "ok"

    result = call_with_retry("test_component", "test-model", fn)
    assert result == "ok"

    records = _read_llm_call_records(tmp_path)
    assert len(records) == 1
    rec = records[0]
    assert rec["agent_name"] == "test_component"
    assert rec["model"] == "test-model"
    assert rec["input_tokens"] == 120
    assert rec["output_tokens"] == 40
    assert rec["attempt_count"] == 1
    assert rec["status"] == "ok"
    assert isinstance(rec["duration_ms"], int)
    assert rec["run_id"] == ""  # no run in progress -- set_current_run_id not called


def test_call_with_retry_records_zero_tokens_when_usage_never_reported(tmp_path):
    """A closure that never calls note_usage must record 0 tokens, not
    a guessed/estimated number."""

    result = call_with_retry("test_component", "test-model", lambda: "ok")
    assert result == "ok"

    records = _read_llm_call_records(tmp_path)
    assert records[0]["input_tokens"] == 0
    assert records[0]["output_tokens"] == 0


def test_call_with_retry_records_failed_status_and_attempt_count_after_exhaustion(tmp_path):
    fn = MagicMock(side_effect=[_server_error(503)] * MAX_LLM_ATTEMPTS)
    with patch("agent_system.llm_retry.time.sleep"):
        with pytest.raises(genai_errors.ServerError):
            call_with_retry("test_component", "test-model", fn)

    records = _read_llm_call_records(tmp_path)
    assert len(records) == 1
    assert records[0]["status"] == "failed"
    assert records[0]["attempt_count"] == MAX_LLM_ATTEMPTS


def test_async_call_with_retry_records_telemetry_and_run_id(tmp_path):
    from agent_system.llm_retry import set_current_run_id

    async def fn():
        note_usage(50, 25)
        return "ok"

    set_current_run_id("run-xyz")
    try:
        result = asyncio.run(async_call_with_retry("evaluator", "model-x", fn))
    finally:
        set_current_run_id("")  # don't leak into other tests

    assert result == "ok"
    records = _read_llm_call_records(tmp_path)
    assert records[0]["run_id"] == "run-xyz"
    assert records[0]["input_tokens"] == 50
    assert records[0]["output_tokens"] == 25
