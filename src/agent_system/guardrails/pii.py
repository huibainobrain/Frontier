"""PII redaction."""

from __future__ import annotations

import re

_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_PHONE_RE = re.compile(r"\b(?:\+?\d{1,3}[-.\s]?)?(?:\(?\d{2,4}\)?[-.\s]?)?\d{3,4}[-.\s]?\d{4}\b")
_API_KEY_RE = re.compile(r"\b(?:sk|pk|api|key|token)-[A-Za-z0-9_\-]{16,}\b", re.IGNORECASE)
_CARD_RE = re.compile(r"\b(?:\d[ -]*?){13,19}\b")


def redact_pii(text: str) -> str:
    redacted = text
    redacted = _EMAIL_RE.sub("[REDACTED_EMAIL]", redacted)
    redacted = _API_KEY_RE.sub("[REDACTED_API_KEY]", redacted)
    redacted = _CARD_RE.sub("[REDACTED_CARD]", redacted)
    redacted = _PHONE_RE.sub("[REDACTED_PHONE]", redacted)
    return redacted