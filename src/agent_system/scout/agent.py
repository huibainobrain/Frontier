"""Scout agent — multi-source literature retrieval.

Implements the interface expected by the Orchestrator:
    Scout(settings) -> plan_sources(goal, profile) -> SourcePlan
                     -> fetch(source_plan) -> list[RawPost]

Uses RSS feeds (feedparser), web fetch (requests + bs4), and Semantic
Scholar / OpenAlex APIs.  No google.generativeai dependency — the Scout
runs as a plain-Python sub-agent driven by the Orchestrator.
"""

from __future__ import annotations

import hashlib
import logging
import re
from datetime import date, datetime, timedelta
from typing import Any

from agent_system.config import Settings, get_settings
from agent_system.schemas import RawPost, SourcePlan, UserProfile
from agent_system.scout.anthropic_source import fetch_anthropic as _fetch_anthropic
from agent_system.scout.relevance import query_tokens, relevance_score
from agent_system.scout.url_utils import canonicalize_url

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Source registry
# ---------------------------------------------------------------------------

RSS_FEEDS: dict[str, str] = {
    # Verified live (curl, 2026-09-14). openai/google_research were pointed
    # at stale URLs (307/302 redirects to nowhere) — swapped for the
    # addresses those redirects were presumably meant to reach.
    "openai": "https://openai.com/news/rss.xml",
    "deepmind": "https://deepmind.google/blog/rss.xml",
    "google_research": "https://research.google/blog/rss/",
    # meta_ai: no working RSS found at any guessed path (ai.meta.com/blog/rss.xml,
    # /blog/rss/, /blog/feed/, /rss/ all 404 or redirect nowhere as of
    # 2026-09-14). about.fb.com has a feed but it's Meta corporate news, not
    # the AI research blog — wrong topic. Left as-is (source will just
    # silently return 0 posts via _fetch_rss's 404 handling) until a real
    # feed is found or this source is dropped.
    "meta_ai": "https://ai.meta.com/blog/rss/",
    "hugging_face": "https://huggingface.co/blog/feed.xml",
}

# Sources that must be queried via REST API instead of RSS
API_SOURCES: set[str] = {"arxiv_cs_ai", "semantic_scholar", "openalex"}

# Anthropic has no RSS feed (verified live) — it's fetched via its own
# adapter (sitemap.xml discovery), dispatched separately in fetch()
# below, not through RSS_FEEDS or FETCHERS.
_ANTHROPIC_SOURCE_ID = "anthropic"

# How much larger a candidate pool _fetch_rss scores/ranks than what it
# actually returns — see agent_system.scout.relevance. Scoring only
# needs title+summary (no extra fetch), so a bigger pool costs nothing
# beyond parsing a few more feed entries already present in the same
# response.
_RSS_CANDIDATE_POOL_MULTIPLIER = 3

ALL_SOURCES: list[str] = [_ANTHROPIC_SOURCE_ID] + list(RSS_FEEDS) + list(API_SOURCES)

# source_id -> publisher/lab name. Only covers sources that are
# themselves a single official lab/company channel — deliberately does
# NOT cover arxiv_cs_ai / semantic_scholar / openalex, since those are
# multi-institution aggregators and a single paper's affiliation can't
# be reliably inferred from the source alone. Downstream code (Analyst,
# Synthesizer) must use this lookup instead of asking an LLM to guess an
# organization from article content.
SOURCE_ORGANIZATIONS: dict[str, str] = {
    _ANTHROPIC_SOURCE_ID: "Anthropic",
    "openai": "OpenAI",
    "deepmind": "DeepMind",
    "google_research": "Google",
    "meta_ai": "Meta",
    "hugging_face": "Hugging Face",
}


def organization_for_source(source: str) -> str:
    """Deterministic source_id -> organization lookup. Returns "" for
    aggregator sources (arXiv, Semantic Scholar, OpenAlex) or anything
    unrecognized — an unknown organization, never a guess."""

    return SOURCE_ORGANIZATIONS.get(source, "")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _post_id(source: str, url: str) -> str:
    """Hash the *canonical* URL, not the raw one — so the same article
    re-discovered with a different tracking param / trailing slash /
    fragment gets the same post_id, and dedup (within one fetch, across
    research rounds, and against the Analyst's SQLite cache) all work
    off one stable identity instead of silently missing the duplicate."""

    return hashlib.sha256(f"{source}:{canonicalize_url(url)}".encode()).hexdigest()[:16]


def _hash_content(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def _fetch_page(url: str) -> str:
    """Return page text, or empty string on failure."""

    import requests

    try:
        resp = requests.get(url, timeout=15, headers={"User-Agent": "FrontierLitAgent/1.0"})
        resp.raise_for_status()
        return resp.text
    except Exception as exc:
        logger.warning("fetch %s failed: %s", url, exc)
        return ""


def _html_to_text(html: str) -> str:
    """Crude HTML → plain text (no bs4 dependency required)."""

    try:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        return soup.get_text(separator="\n", strip=True)
    except ImportError:
        # fallback: strip tags with regex
        return re.sub(r"<[^>]+>", " ", html)


# ---------------------------------------------------------------------------
# RSS fetching
# ---------------------------------------------------------------------------


def _fetch_rss(
    source_id: str,
    since: date | None = None,
    query: str = "",
    max_results: int = 5,
) -> list[RawPost]:
    """Fetch RSS entries ranked by relevance to *query*, not just the
    most recent ``max_results`` entries.

    Scores a candidate pool larger than ``max_results`` (title+summary
    keyword overlap against *query* — see ``agent_system.scout.
    relevance``; no extra fetch needed, both fields already come from
    the feed response) and keeps the top-scoring ones. When nothing in
    the pool overlaps the query at all (or *query* is empty), every
    candidate ties at score 0 and the stable sort below falls back to
    the feed's own newest-first order — i.e. this degrades to the old
    "latest N" behavior rather than returning an arbitrary or empty
    result. This is what makes a Planner/Replanner query rewrite
    actually change what an RSS-backed source returns.
    """

    import feedparser

    url = RSS_FEEDS.get(source_id)
    if url is None:
        return []

    logger.info("RSS fetch: %s (%s)", source_id, url)
    feed = feedparser.parse(url)
    tokens = query_tokens(query)
    pool_size = max(max_results * _RSS_CANDIDATE_POOL_MULTIPLIER, max_results)

    candidates: list[dict[str, Any]] = []
    for order, entry in enumerate(feed.entries[:pool_size]):
        published_parsed = getattr(entry, "published_parsed", None)
        published_dt = (
            datetime(*published_parsed[:6]) if published_parsed else datetime.now()
        )
        if since and published_dt.date() < since:
            continue
        title = getattr(entry, "title", "Untitled")
        summary = getattr(entry, "summary", "")
        candidates.append(
            {
                "entry": entry,
                "title": title,
                "summary": summary,
                "published_dt": published_dt,
                "score": relevance_score(tokens, title, summary),
                "order": order,
            }
        )

    # Highest score first; ties broken by feed order (lowest "order" =
    # most recent), which is also the only signal available when every
    # candidate scores 0.
    candidates.sort(key=lambda c: (-c["score"], c["order"]))

    posts: list[RawPost] = []
    for c in candidates[:max_results]:
        entry = c["entry"]
        link = getattr(entry, "link", "")

        # use RSS summary first, only fetch full page if empty
        content = c["summary"]
        html = ""
        if not content:
            html = _fetch_page(link)
            content = _html_to_text(html) if html else ""

        if not content:
            continue

        authors = []
        if hasattr(entry, "authors"):
            authors = [a.get("name", "") for a in entry.authors if a.get("name")]

        posts.append(
            RawPost(
                post_id=_post_id(source_id, link),
                source=source_id,
                url=link,
                title=c["title"],
                authors=authors,
                published_at=c["published_dt"].isoformat(),
                content=content[:8000],
                content_type="blog",
                raw_html_hash=_hash_content(html) if html else "",
            )
        )

    logger.info("RSS %s: %d posts (query_tokens=%d)", source_id, len(posts), len(tokens))
    return posts


# ---------------------------------------------------------------------------
# API-based sources
# ---------------------------------------------------------------------------


def _fetch_arxiv(query: str, max_results: int = 10) -> list[RawPost]:
    """Search arXiv via its public API (no library needed)."""

    import urllib.parse
    import xml.etree.ElementTree as ET

    params = urllib.parse.urlencode(
        {"search_query": f"all:{query}", "start": 0, "max_results": max_results}
    )
    url = f"http://export.arxiv.org/api/query?{params}"
    xml_text = _fetch_page(url)
    if not xml_text:
        return []

    ns = {"atom": "http://www.w3.org/2005/Atom"}
    root = ET.fromstring(xml_text)
    posts: list[RawPost] = []

    for entry in root.findall("atom:entry", ns):
        eid = entry.findtext("atom:id", "", ns)
        title = entry.findtext("atom:title", "", ns).strip().replace("\n", " ")
        summary = entry.findtext("atom:summary", "", ns).strip()
        published = entry.findtext("atom:published", "", ns)
        authors = [
            a.findtext("atom:name", "", ns)
            for a in entry.findall("atom:author", ns)
        ]

        posts.append(
            RawPost(
                post_id=_post_id("arxiv", eid),
                source="arxiv",
                url=eid,
                title=title,
                authors=authors,
                published_at=published,
                content=summary,
                content_type="paper",
            )
        )

    logger.info("arXiv '%s': %d results", query, len(posts))
    return posts


def _fetch_semantic_scholar(query: str, max_results: int = 10) -> list[RawPost]:
    """Search Semantic Scholar's public API."""

    import requests

    try:
        resp = requests.get(
            "https://api.semanticscholar.org/graph/v1/paper/search",
            params={"query": query, "limit": max_results, "fields": "title,abstract,authors,year,url"},
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        logger.warning("Semantic Scholar failed: %s", exc)
        return []

    posts: list[RawPost] = []
    for item in data.get("data", []):
        abstract = item.get("abstract") or ""
        if not abstract:
            continue
        url = item.get("url", "")
        authors = [a["name"] for a in item.get("authors", [])]
        year = item.get("year") or ""

        posts.append(
            RawPost(
                post_id=_post_id("semantic_scholar", url or item.get("paperId", "")),
                source="semantic_scholar",
                url=url,
                title=item.get("title", ""),
                authors=authors,
                published_at=f"{year}-01-01" if year else "",
                content=abstract,
                content_type="paper",
            )
        )

    logger.info("Semantic Scholar '%s': %d results", query, len(posts))
    return posts


def _fetch_openalex(query: str, max_results: int = 10) -> list[RawPost]:
    """Search OpenAlex REST API."""

    import requests

    try:
        resp = requests.get(
            "https://api.openalex.org/works",
            params={"search": query, "per_page": max_results, "mailto": "demo@example.com"},
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        logger.warning("OpenAlex failed: %s", exc)
        return []

    posts: list[RawPost] = []
    for item in data.get("results", []):
        # reconstruct abstract from inverted index
        inv_idx = item.get("abstract_inverted_index")
        if inv_idx:
            max_idx = max((max(ids) for ids in inv_idx.values() if ids), default=0)
            words = [""] * (max_idx + 1)
            for word, indices in inv_idx.items():
                for i in indices:
                    words[i] = word
            abstract = " ".join(w for w in words if w)
        else:
            abstract = ""
        if not abstract:
            continue

        url = item.get("doi") or item.get("id", "")
        authors = [a["author"]["display_name"] for a in item.get("authorships", [])]

        posts.append(
            RawPost(
                post_id=_post_id("openalex", url),
                source="openalex",
                url=url,
                title=item.get("display_name", ""),
                authors=authors,
                published_at=item.get("publication_date", ""),
                content=abstract[:4000],
                content_type="paper",
            )
        )

    logger.info("OpenAlex '%s': %d results", query, len(posts))
    return posts


# ---------------------------------------------------------------------------
# Scout class (Orchestrator-compatible interface)
# ---------------------------------------------------------------------------

FETCHERS: dict[str, Any] = {
    "arxiv_cs_ai": _fetch_arxiv,
    "semantic_scholar": _fetch_semantic_scholar,
    "openalex": _fetch_openalex,
}


class Scout:
    """Retrieval agent — the Orchestrator calls ``plan_sources`` then ``fetch``."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings if settings is not None else get_settings()

    # ---- interface expected by Orchestrator --------------------------------

    def plan_sources(self, user_goal: str, profile: UserProfile) -> SourcePlan:
        """Simple heuristic planner — no LLM call required."""

        keywords = self._extract_keywords(user_goal)
        return SourcePlan(
            sources_to_query=[
                _ANTHROPIC_SOURCE_ID,
                "openai",
                "deepmind",
                "hugging_face",
                "arxiv_cs_ai",
            ],
            time_window_days=14,
            filter_keywords=keywords,
            max_posts=20,
        )

    def fetch(self, plan: SourcePlan) -> list[RawPost]:
        """Execute the source plan and return deduplicated posts."""

        since = date.today() - timedelta(days=plan.time_window_days)
        all_posts: list[RawPost] = []
        query = " ".join(plan.filter_keywords) or "AI agent"

        for source in plan.sources_to_query:
            try:
                if source in RSS_FEEDS:
                    all_posts.extend(
                        _fetch_rss(source, since=since, query=query, max_results=5)
                    )
                elif source == _ANTHROPIC_SOURCE_ID:
                    all_posts.extend(
                        _fetch_anthropic(since=since, max_results=5, query=query)
                    )
                elif source in FETCHERS:
                    all_posts.extend(FETCHERS[source](query, max_results=5))
                else:
                    logger.warning("unknown source: %s", source)
            except Exception as exc:
                logger.error("source %s failed: %s", source, exc)

        # deduplicate
        seen: dict[str, RawPost] = {}
        for p in all_posts:
            if p.post_id not in seen:
                seen[p.post_id] = p

        result = list(seen.values())[:plan.max_posts]
        logger.info("Scout.fetch: %d posts (deduped from %d)", len(result), len(all_posts))
        self._persist(result)
        return result

    # ---- helpers -----------------------------------------------------------

    def _persist(self, posts: list[RawPost]) -> None:
        """Cache fetched posts to SQLite.

        This is what makes a post's post_id findable by the Analyst's
        own cache (:meth:`agent_system.analyst.agent.Analyst._get_cached`),
        so a post that was already analyzed in a prior run is never
        re-sent to Gemini. Storage failures are logged, not raised — a
        persistence hiccup should not fail a fetch that already
        succeeded.
        """

        from agent_system.storage import db as storage_db

        for post in posts:
            try:
                storage_db.save_raw_post(post, settings=self.settings)
            except Exception:
                logger.exception("Failed to cache RawPost %s", post.post_id)

    @staticmethod
    def _extract_keywords(text: str) -> list[str]:
        """Pull meaningful words from user goal for source filtering."""

        stop = {"the", "a", "an", "is", "are", "what", "how", "latest", "new", "recent", "and", "or", "of", "in", "for", "to", "from", "with", "on", "about"}
        words = re.findall(r"[a-zA-Z]{3,}", text.lower())
        return [w for w in words if w not in stop][:8]
