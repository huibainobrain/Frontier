"""Guardrail layer — chains injection / PII / topic checks per pipeline stage.

The Orchestrator calls :func:`run_guardrails` at three points:

* ``stage="input"``  — on the raw user goal before any LLM call.
  PII redaction, prompt-injection detection, and topic/scope
  screening (is this goal within FrontierLit's research scope?) all run
  here. Topic screening belongs here rather than on the output: it's a
  gate on what the *user is asking for*, and default-permissive on
  ambiguity — see ``guardrails.topic_filter`` for why.
* ``stage="content"`` — on every fetched post body before the Analyst
  reads it. (Wired by the Analyst, not the Orchestrator.)
* ``stage="output"`` — on the rendered synthesis text before it is
  returned to the user. PII redaction only; citation completeness is a
  separate, structured check
  (``guardrails.output_checker.find_missing_citations`` /
  ``check_claim_citations``) that operates on the ``DraftSynthesis``
  object rather than rendered text, so it doesn't fit this text-only
  dispatcher — it's called directly by the Orchestrator's
  ``_guard_output``.
"""

from __future__ import annotations

from typing import Literal

from agent_system.guardrails.pii import redact_pii
from agent_system.guardrails.prompt_injection import check_injection
from agent_system.guardrails.result import GuardrailResult
from agent_system.guardrails.topic_filter import is_on_topic

GuardrailStage = Literal["input", "content", "output"]


def run_guardrails(text: str, stage: GuardrailStage) -> GuardrailResult:
    """Run the chain of guardrails appropriate for ``stage`` and return
    a single composite :class:`GuardrailResult`."""

    if stage == "input":
        sanitized = redact_pii(text)
        injection = check_injection(sanitized)
        if not injection.safe:
            return injection
        topic = is_on_topic(sanitized, None)
        if not topic.safe:
            return topic
        return GuardrailResult(safe=True, sanitized_text=sanitized)

    if stage == "content":
        return check_injection(text)

    if stage == "output":
        # PII redaction only — see the module docstring for why the
        # citation check isn't dispatched from here.
        return GuardrailResult(safe=True, sanitized_text=redact_pii(text))

    return GuardrailResult(safe=False, reason=f"unknown guardrail stage: {stage!r}")


__all__ = [
    "GuardrailResult",
    "GuardrailStage",
    "run_guardrails",
    "check_injection",
    "redact_pii",
    "is_on_topic",
]
