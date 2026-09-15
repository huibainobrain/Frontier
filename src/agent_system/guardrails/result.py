"""Shared dataclass for guardrail return values.

Lives in its own module to keep the per-stage guardrails (prompt
injection, PII, topic filter) free of circular imports with
``guardrails/__init__.py``.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class GuardrailResult:
    """Return value for every guardrail check.

    * ``safe`` — True if the input passes; False blocks the pipeline.
    * ``reason`` — short human-readable string when ``safe=False``.
    * ``sanitized_text`` — optional sanitized version (e.g. PII-redacted)
      that downstream agents should use instead of the original.
    """

    safe: bool
    reason: str = ""
    sanitized_text: str | None = None
