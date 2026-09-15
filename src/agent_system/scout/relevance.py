"""Lightweight keyword-relevance scoring for RSS/sitemap sources.

Academic search APIs (arXiv, Semantic Scholar, OpenAlex) already take the
current query as a real search parameter, so a Planner/Replanner query
rewrite genuinely changes what they return. RSS feeds and the Anthropic
sitemap adapter have no search endpoint at all — before this module
existed, they just returned the N most recent entries regardless of what
the query said, so a query rewrite never actually changed what came back
from those sources (see ``scout/agent.py::_fetch_rss`` and
``scout/anthropic_source.py::fetch_anthropic``).

Deliberately NOT a reranker or embedding model — plain, case-insensitive
token overlap between the query and each candidate's available text
(title/summary for RSS, the URL slug for the sitemap adapter, which has
no title/summary before a page is actually fetched). Just enough that a
query change can change which candidates survive down to ``max_results``,
without pretending to be a real search algorithm.
"""

from __future__ import annotations

import re

_TOKEN_RE = re.compile(r"[a-zA-Z0-9]{3,}")


def query_tokens(query: str) -> set[str]:
    """Lowercased, deduped tokens (3+ chars) extracted from a query
    string. An empty/whitespace-only query yields an empty set."""

    return set(_TOKEN_RE.findall(query.lower()))


def relevance_score(tokens: set[str], *texts: str) -> int:
    """Count of *tokens* that appear as a substring anywhere across
    *texts* (joined, lowercased). 0 means no overlap at all — callers
    should treat that as "no signal to rank on", not "irrelevant, drop
    it": this is a coarse filter, not a precision search engine, and a
    real query with genuinely no matching content in the current
    candidate pool should still degrade gracefully to recency order
    rather than returning nothing.
    """

    if not tokens:
        return 0
    haystack = " ".join(t for t in texts if t).lower()
    return sum(1 for tok in tokens if tok in haystack)
