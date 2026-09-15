"""Observability tests."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from agent_system.config import Settings


def _make_settings(tmp_path: Path) -> Settings:
    return Settings(
        google_api_key="test-key",
        data_dir=tmp_path,
        sqlite_path=tmp_path / "x.sqlite",
        chroma_dir=tmp_path / "chroma",
        token_log_path=tmp_path / "token_log.jsonl",
    )


@pytest.fixture
def settings(tmp_path):
    s = _make_settings(tmp_path)
    with patch("agent_system.observability.tracing._token_log_path", return_value=s.token_log_path):
        yield s


def _read_log(settings: Settings) -> list[dict]:
    if not settings.token_log_path.exists():
        return []
    return [
        json.loads(line)
        for line in settings.token_log_path.read_text().splitlines()
        if line.strip()
    ]


def test_traced_sync_writes_record(settings):
    from agent_system.observability.tracing import traced

    @traced("test_agent.sync")
    def f(x):
        return x + 1

    assert f(2) == 3
    records = _read_log(settings)
    assert len(records) == 1
    assert records[0]["agent_name"] == "test_agent.sync"
    assert records[0]["status"] == "ok"


def test_traced_async_writes_record(settings):
    from agent_system.observability.tracing import traced

    @traced("test_agent.async")
    async def f(x):
        return x + 1

    assert asyncio.run(f(2)) == 3
    records = _read_log(settings)
    assert len(records) == 1
    assert records[0]["agent_name"] == "test_agent.async"


def test_trace_span_writes_record(settings):
    from agent_system.observability.tracing import trace_span

    with trace_span("my.span"):
        x = 1 + 1

    records = _read_log(settings)
    assert len(records) == 1
    assert records[0]["agent_name"] == "my.span"
    assert records[0]["kind"] == "span"


def test_traced_records_error_status(settings):
    from agent_system.observability.tracing import traced

    @traced("test_agent.boom")
    def f():
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        f()
    records = _read_log(settings)
    assert records[0]["status"] == "error"


def test_record_llm_call_writes_token_counts(settings):
    from agent_system.observability.tracing import record_llm_call

    record_llm_call(
        agent_name="orchestrator.intent",
        model="gemini-2.0-flash",
        input_tokens=120,
        output_tokens=40,
        cost_estimate=0.000123,
    )
    records = _read_log(settings)
    assert records[0]["input_tokens"] == 120
    assert records[0]["output_tokens"] == 40
    assert records[0]["model"] == "gemini-2.0-flash"


def test_current_budget_aggregates(settings):
    from agent_system.observability.tracing import current_budget, record_llm_call

    record_llm_call("orch", "gemini-2.0-flash", 100, 50, cost_estimate=0.001)
    record_llm_call("orch", "gemini-2.5-pro", 200, 100, cost_estimate=0.01)
    record_llm_call("scout", "gemini-2.0-flash", 50, 25, cost_estimate=0.0005)

    budget = current_budget()
    assert budget["n_calls"] == 3
    assert budget["llm_call_count"] == 3
    assert budget["total_tokens"] == 100 + 50 + 200 + 100 + 50 + 25
    assert pytest.approx(budget["total_cost_estimate"], rel=1e-6) == 0.0115
    assert budget["by_agent"]["orch"] == 100 + 50 + 200 + 100
    assert budget["by_model"]["gemini-2.0-flash"] == 100 + 50 + 50 + 25


def test_current_budget_llm_call_count_excludes_spans(settings):
    """Test C5: a trace mixing span/llm_call/span/llm_call records must
    count only the llm_call ones toward LLM Call Count — not 4. Before
    this fix, n_calls counted every JSONL line regardless of kind."""

    from agent_system.observability.tracing import (
        current_budget,
        record_llm_call,
        trace_span,
    )

    with trace_span("some.step"):
        pass
    record_llm_call("orch", "model-a", 10, 5)
    with trace_span("another.step"):
        pass
    record_llm_call("orch", "model-a", 20, 10)

    records = _read_log(settings)
    assert len(records) == 4  # 2 spans + 2 llm_calls actually logged

    budget = current_budget()
    assert budget["llm_call_count"] == 2
    assert budget["n_calls"] == 2


def test_summarize_llm_calls_filters_by_run_id(settings):
    """summarize_llm_calls must scope to one run_id and ignore other
    runs' (or run-id-less) records — this is what
    Orchestrator._build_run_summary uses for the per-run telemetry
    rollup."""

    from agent_system.observability.tracing import record_llm_call, summarize_llm_calls

    record_llm_call(
        "planner", "model-a", 100, 50, status="ok",
        extra={"run_id": "run-1", "attempt_count": 1},
    )
    record_llm_call(
        "analyst", "model-a", 30, 10, status="failed",
        extra={"run_id": "run-1", "attempt_count": 3},
    )
    record_llm_call(
        "planner", "model-a", 999, 999, status="ok",
        extra={"run_id": "run-2", "attempt_count": 1},
    )

    summary = summarize_llm_calls("run-1")
    assert summary["llm_call_count"] == 2
    assert summary["input_tokens"] == 130
    assert summary["output_tokens"] == 60
    assert summary["total_tokens"] == 190
    assert summary["fallback_count"] == 1
    assert summary["by_component"] == {"planner": 1, "analyst": 1}
