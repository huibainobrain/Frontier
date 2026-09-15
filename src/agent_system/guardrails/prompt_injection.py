"""Prompt-injection detector."""

from __future__ import annotations

from agent_system.guardrails.result import GuardrailResult

_INJECTION_PATTERNS = [
    "ignore previous instructions",
    "disregard previous instructions",
    "reveal your system prompt",
    "show your system prompt",
    "developer message",
    "system prompt",
    "send private data",
    "exfiltrate",
    "bypass safety",
    "disable guardrails",
    "forget your instructions",
    "you are now",
]


def check_injection(text: str) -> GuardrailResult:
    lowered = text.lower()

    for pattern in _INJECTION_PATTERNS:
        if pattern in lowered:
            return GuardrailResult(
                safe=False,
                reason=f"Prompt injection risk detected: {pattern}",
                sanitized_text=None,
            )

    return GuardrailResult(safe=True, sanitized_text=text)