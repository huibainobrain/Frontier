"""Shared test fixtures.

Autouse fixture: redirect the LLM/observability telemetry log to a
per-test tmp path, so exercising anything that goes through
``trace_span``/``@traced`` (every ``Orchestrator.run_async`` step) or
``agent_system.llm_retry.call_with_retry``/``async_call_with_retry``
(which now records real per-call telemetry — see ``note_usage``/
``_record_telemetry`` in ``agent_system.llm_retry``) never appends to
the project's real ``data/token_log.jsonl`` as a side effect of running
the test suite.
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _isolate_token_log(tmp_path, monkeypatch):
    from agent_system.observability import tracing

    log_path = tmp_path / "token_log.jsonl"
    monkeypatch.setattr(tracing, "_token_log_path", lambda: log_path)
