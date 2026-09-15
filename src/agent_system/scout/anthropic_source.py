"""Anthropic official-source adapter — newsroom / research / engineering.

Anthropic doesn't publish a classic RSS feed (verified live, 2026-09:
/news/rss, /rss.xml, /feed, /engineering/rss.xml all 404). Their public
``sitemap.xml`` does list every ``/news/``, ``/research/``, and
``/engineering/`` page with a ``lastmod`` timestamp — a stable,
documented, publicly-served index file, not something scraped off a
rendered page. Fetching each discovered page is a single plain HTTP GET
(confirmed live: Anthropic's article pages are server-rendered, the
text is present in the raw HTML — no headless browser needed).

Kept as its own module — independent of ``scout/agent.py`` in both
directions — so Anthropic's URL layout changing only touches this file,
and so it can be unit-tested and reasoned about on its own. It exposes
one function, :func:`fetch_anthropic`, with the same
``(since, max_results) -> list[RawPost]`` shape Scout's other fetchers
use, so ``Scout.fetch`` only needs one dispatch branch to use it.

Failure isolation: every external call in here is wrapped so a broken
sitemap, a changed URL layout, or a network failure logs a warning and
returns an empty list rather than raising — this source failing must
never take down the rest of a research task (the same contract Scout's
own per-source try/except already assumes of every fetcher).
"""

from __future__ import annotations

import hashlib
import html
import logging
import re
import xml.etree.ElementTree as ET
from datetime import date, datetime

from agent_system.schemas import RawPost
from agent_system.scout.relevance import query_tokens, relevance_score
from agent_system.scout.url_utils import canonicalize_url

logger = logging.getLogger(__name__)

SITEMAP_URL = "https://www.anthropic.com/sitemap.xml"

# Official public content sections. Deliberately narrow — the sitemap
# also lists /careers, /company, /legal, etc., which aren't research
# literature.
_CONTENT_PREFIXES = ("/news/", "/research/", "/engineering/")

_SITEMAP_NS = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}


# ---------------------------------------------------------------------------
# Small self-contained helpers (deliberately not shared with scout/agent.py —
# see module docstring: this adapter has no dependency on it in either
# direction).
# ---------------------------------------------------------------------------


def _post_id(url: str) -> str:
    """Hash the canonical URL — see agent_system.scout.url_utils — so a
    URL re-discovered with a different tracking param / trailing slash
    gets the same post_id as before, instead of silently duplicating."""

    return hashlib.sha256(f"anthropic:{canonicalize_url(url)}".encode()).hexdigest()[:16]


def _hash_content(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def _http_get(url: str, timeout: int = 15) -> str:
    """Return response text, or "" on any failure. Never raises."""

    import requests

    try:
        resp = requests.get(
            url, timeout=timeout, headers={"User-Agent": "FrontierLitAgent/1.0"}
        )
        resp.raise_for_status()
        return resp.text
    except Exception as exc:
        logger.warning("Anthropic: fetch %s failed: %s", url, exc)
        return ""


def _html_to_text(html_text: str) -> str:
    """Crude HTML -> plain text: drop script/style, strip tags, collapse
    whitespace. No bs4 dependency — confirmed live that Anthropic's pages
    are server-rendered, so this is enough to get real article text."""

    text = re.sub(r"<script[^>]*>.*?</script>", " ", html_text, flags=re.S | re.I)
    text = re.sub(r"<style[^>]*>.*?</style>", " ", text, flags=re.S | re.I)
    text = re.sub(r"<[^>]+>", " ", text)
    text = html.unescape(text)
    return re.sub(r"\s+", " ", text).strip()


def _title_from_html(html_text: str, url: str) -> str:
    match = re.search(r"<title[^>]*>(.*?)</title>", html_text, re.IGNORECASE | re.DOTALL)
    if match:
        title = html.unescape(re.sub(r"\s+", " ", match.group(1)).strip())
        # Anthropic's <title> is "Page Name \ Anthropic" — drop the suffix.
        title = re.sub(r"\s*\\\s*Anthropic\s*$", "", title).strip()
        if title:
            return title
    slug = url.rstrip("/").rsplit("/", 1)[-1]
    return slug.replace("-", " ").strip().capitalize()


def _parse_lastmod(lastmod: str) -> datetime | None:
    if not lastmod:
        return None
    try:
        return datetime.fromisoformat(lastmod.replace("Z", "+00:00"))
    except ValueError:
        return None


def _slug_relevance(url: str, tokens: set[str]) -> int:
    """Weak, zero-cost relevance signal from the URL path alone — the
    sitemap gives no title/summary, so the slug (e.g. "/research/
    mechanistic-interpretability-sparse-autoencoders") is the only text
    available before actually fetching a candidate's page. See
    agent_system.scout.relevance for why this doesn't need to be
    precise, just enough that a query change can change which
    candidates get fetched at all."""

    slug_text = url.replace("-", " ").replace("_", " ").replace("/", " ")
    return relevance_score(tokens, slug_text)


def _is_content_url(url: str) -> bool:
    for prefix in _CONTENT_PREFIXES:
        idx = url.find(prefix)
        # must contain the prefix and have something after it (not just
        # the section landing page itself, e.g. bare ".../news/")
        if idx != -1 and len(url) > idx + len(prefix):
            return True
    return False


def discover_urls(sitemap_xml: str) -> list[tuple[str, datetime | None]]:
    """Parse a sitemap.xml body into [(url, lastmod_or_None), ...],
    filtered to newsroom/research/engineering content and sorted newest
    first. Raises only on genuinely malformed XML — callers should still
    wrap this (see :func:`fetch_anthropic`)."""

    root = ET.fromstring(sitemap_xml)
    entries: list[tuple[str, datetime | None]] = []
    for url_el in root.findall("sm:url", _SITEMAP_NS):
        loc = (url_el.findtext("sm:loc", "", _SITEMAP_NS) or "").strip()
        lastmod = (url_el.findtext("sm:lastmod", "", _SITEMAP_NS) or "").strip()
        if loc and _is_content_url(loc):
            entries.append((loc, _parse_lastmod(lastmod)))

    entries.sort(key=lambda pair: pair[1] or datetime.min, reverse=True)
    return entries


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def fetch_anthropic(
    since: date | None = None, max_results: int = 5, query: str = ""
) -> list[RawPost]:
    """Discover and fetch recent Anthropic official posts, ranked by
    relevance to *query* against each candidate's URL slug (see
    ``_slug_relevance``) — a query rewrite changes which candidate
    pages actually get fetched, not just which N are newest.

    Never raises — any failure (sitemap unreachable, malformed XML, a
    page fetch failing) is logged and contributes an empty/partial
    result, so a broken Anthropic source degrades this one source, not
    the caller's whole research task.
    """

    try:
        sitemap_xml = _http_get(SITEMAP_URL)
        if not sitemap_xml:
            logger.warning("Anthropic: sitemap fetch returned no content")
            return []
        candidates = discover_urls(sitemap_xml)
    except Exception:
        logger.exception("Anthropic: sitemap discovery failed")
        return []

    if since is not None:
        # Entries with no parseable lastmod are kept (can't confirm
        # they're outside the window, so don't silently drop them) but
        # sort to the back since discover_urls already put dated entries
        # first.
        candidates = [
            (url, lastmod)
            for url, lastmod in candidates
            if lastmod is None or lastmod.date() >= since
        ]

    # Rank by relevance to *query*. A stable sort on relevance alone
    # (no explicit recency key) is enough: candidates arrive already
    # lastmod-descending from discover_urls/the since-filter above, so
    # ties (including the common all-zero case — no query, or nothing
    # in this batch matches it) keep that existing recency order rather
    # than being shuffled.
    tokens = query_tokens(query)
    candidates.sort(key=lambda pair: -_slug_relevance(pair[0], tokens))

    posts: list[RawPost] = []
    seen_urls: set[str] = set()
    for url, lastmod in candidates:
        if len(posts) >= max_results:
            break
        canonical = canonicalize_url(url)
        if canonical in seen_urls:
            continue
        seen_urls.add(canonical)

        page_html = _http_get(url)
        if not page_html:
            continue
        text = _html_to_text(page_html)
        if not text:
            continue

        posts.append(
            RawPost(
                post_id=_post_id(url),
                source="anthropic",
                url=url,
                title=_title_from_html(page_html, url),
                authors=[],
                published_at=lastmod.isoformat() if lastmod else "",
                content=text[:8000],
                content_type="blog",
                raw_html_hash=_hash_content(page_html),
            )
        )

    logger.info("Anthropic: %d post(s) fetched via sitemap discovery", len(posts))
    return posts
