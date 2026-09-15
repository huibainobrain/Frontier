"""Source trust scoring for retrieved frontier AI literature."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SourceTrustScore:
    score: int
    reason: str


_OFFICIAL_LAB_SOURCES = [
    "anthropic",
    "openai",
    "deepmind",
    "google research",
    "meta ai",
]

_MID_HIGH_SOURCES = [
    "arxiv",
    "hugging face",
    "github",
]


def score_source(
    source: str,
    url: str = "",
    published_at: str = "",
    content_type: str = "blog",
) -> SourceTrustScore:
    """Return a simple 1-5 trust score for a retrieved source.

    5 = official / highly reliable
    4 = research or implementation source
    3 = acceptable but less authoritative
    2 = weak source
    1 = missing key metadata
    """
    source_l = source.lower()
    url_l = url.lower()

    score = 3
    reasons: list[str] = []

    if any(s in source_l or s in url_l for s in _OFFICIAL_LAB_SOURCES):
        score = 5
        reasons.append("official frontier AI lab source")
    elif any(s in source_l or s in url_l for s in _MID_HIGH_SOURCES):
        score = 4
        reasons.append("research or implementation source")
    else:
        reasons.append("non-official or unclear source")

    if content_type == "paper" and "arxiv" in source_l + url_l:
        score = max(score, 4)
        reasons.append("arXiv paper")

    if not url:
        score -= 1
        reasons.append("missing URL")

    if not published_at:
        score -= 1
        reasons.append("missing publication date")

    score = max(1, min(5, score))

    return SourceTrustScore(
        score=score,
        reason="; ".join(reasons),
    )