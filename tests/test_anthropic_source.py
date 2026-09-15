"""Tests for the Anthropic source adapter.

All network calls (``_http_get``) are mocked — no real HTTP requests.
A live smoke test against the real anthropic.com sitemap was run
manually during development; these tests cover the adapter's logic in
isolation instead of depending on Anthropic's site being reachable
during CI.
"""

from __future__ import annotations

from datetime import date
from unittest.mock import patch

from agent_system.schemas import RawPost
from agent_system.scout.anthropic_source import discover_urls, fetch_anthropic

FAKE_SITEMAP = """<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
<url><loc>https://www.anthropic.com/careers</loc><lastmod>2026-09-01T00:00:00.000Z</lastmod></url>
<url><loc>https://www.anthropic.com/news/example-post</loc><lastmod>2026-09-10T00:00:00.000Z</lastmod></url>
<url><loc>https://www.anthropic.com/research/older-paper</loc><lastmod>2026-01-01T00:00:00.000Z</lastmod></url>
<url><loc>https://www.anthropic.com/research/newer-paper</loc><lastmod>2026-09-12T00:00:00.000Z</lastmod></url>
<url><loc>https://www.anthropic.com/engineering/infra-post</loc><lastmod>2026-09-05T00:00:00.000Z</lastmod></url>
<url><loc>https://www.anthropic.com/news/</loc><lastmod>2026-09-14T00:00:00.000Z</lastmod></url>
</urlset>"""

FAKE_PAGE_HTML = (
    "<html><head><title>Example Post \\ Anthropic</title></head>"
    "<body><p>Anthropic today announced a new capability for Claude, "
    "described in detail across several paragraphs of real content.</p></body></html>"
)


# ---------------------------------------------------------------------------
# discover_urls — sitemap parsing + filtering
# ---------------------------------------------------------------------------


def test_discover_urls_finds_at_least_one_official_post():
    entries = discover_urls(FAKE_SITEMAP)

    assert len(entries) > 0
    urls = [url for url, _ in entries]
    assert "https://www.anthropic.com/news/example-post" in urls


def test_discover_urls_excludes_non_content_sections():
    entries = discover_urls(FAKE_SITEMAP)
    urls = [url for url, _ in entries]

    assert "https://www.anthropic.com/careers" not in urls


def test_discover_urls_excludes_section_landing_page():
    """.../news/ itself (no slug after it) is a listing page, not a post."""

    entries = discover_urls(FAKE_SITEMAP)
    urls = [url for url, _ in entries]

    assert "https://www.anthropic.com/news/" not in urls


def test_discover_urls_sorts_newest_first():
    entries = discover_urls(FAKE_SITEMAP)
    urls = [url for url, _ in entries]

    assert urls.index("https://www.anthropic.com/research/newer-paper") < urls.index(
        "https://www.anthropic.com/research/older-paper"
    )


# ---------------------------------------------------------------------------
# fetch_anthropic — discovery + fetch + RawPost conversion
# ---------------------------------------------------------------------------


def test_fetch_anthropic_returns_valid_rawposts():
    with patch(
        "agent_system.scout.anthropic_source._http_get",
        side_effect=lambda url, timeout=15: FAKE_SITEMAP
        if url.endswith("sitemap.xml")
        else FAKE_PAGE_HTML,
    ):
        posts = fetch_anthropic(max_results=10)

    assert len(posts) > 0
    for post in posts:
        assert isinstance(post, RawPost)
        assert post.source == "anthropic"
        assert post.content_type == "blog"
        assert post.content  # real text, not empty
        assert post.title  # decoded, no stray HTML entities/suffix
        assert "\\ Anthropic" not in post.title


def test_fetch_anthropic_respects_since_window():
    with patch(
        "agent_system.scout.anthropic_source._http_get",
        side_effect=lambda url, timeout=15: FAKE_SITEMAP
        if url.endswith("sitemap.xml")
        else FAKE_PAGE_HTML,
    ):
        posts = fetch_anthropic(since=date(2026, 9, 1), max_results=10)

    urls = [p.url for p in posts]
    assert "https://www.anthropic.com/research/older-paper" not in urls  # 2026-01-01, too old


def test_fetch_anthropic_does_not_refetch_duplicate_urls():
    """discover_urls can never yield a URL twice from one sitemap, but
    fetch_anthropic's own seen_urls guard is what actually enforces it
    — this pins that behavior directly."""

    call_log: list[str] = []

    def fake_get(url, timeout=15):
        call_log.append(url)
        if url.endswith("sitemap.xml"):
            return FAKE_SITEMAP
        return FAKE_PAGE_HTML

    with patch("agent_system.scout.anthropic_source._http_get", side_effect=fake_get):
        fetch_anthropic(max_results=10)

    page_fetches = [u for u in call_log if not u.endswith("sitemap.xml")]
    assert len(page_fetches) == len(set(page_fetches))


def test_fetch_anthropic_caps_at_max_results():
    with patch(
        "agent_system.scout.anthropic_source._http_get",
        side_effect=lambda url, timeout=15: FAKE_SITEMAP
        if url.endswith("sitemap.xml")
        else FAKE_PAGE_HTML,
    ):
        posts = fetch_anthropic(max_results=2)

    assert len(posts) == 2


# ---------------------------------------------------------------------------
# Failure isolation — never raises, always degrades to []
# ---------------------------------------------------------------------------


def test_fetch_anthropic_survives_sitemap_unreachable():
    with patch("agent_system.scout.anthropic_source._http_get", return_value=""):
        posts = fetch_anthropic()

    assert posts == []


def test_fetch_anthropic_survives_malformed_sitemap():
    with patch(
        "agent_system.scout.anthropic_source._http_get",
        return_value="<not><valid xml",
    ):
        posts = fetch_anthropic()

    assert posts == []


def test_fetch_anthropic_survives_page_fetch_failures():
    """Sitemap discovery succeeds but every individual page fetch fails
    — must degrade to an empty list, not raise."""

    with patch(
        "agent_system.scout.anthropic_source._http_get",
        side_effect=lambda url, timeout=15: FAKE_SITEMAP
        if url.endswith("sitemap.xml")
        else "",
    ):
        posts = fetch_anthropic()

    assert posts == []


def test_fetch_anthropic_failure_does_not_affect_other_scout_sources(tmp_path):
    """Scout.fetch must keep going when the anthropic source raises —
    matches how every other per-source failure is already isolated."""

    from agent_system.config import Settings
    from agent_system.schemas import RawPost as RP
    from agent_system.schemas import SourcePlan
    from agent_system.scout.agent import Scout

    good_post = RP(
        post_id="ok-1",
        source="deepmind",
        url="https://example.com/ok-1",
        title="OK",
        authors=[],
        published_at="2026-09-01T00:00:00",
        content="Body.",
        content_type="blog",
    )
    settings = Settings(
        google_api_key="test-key",
        data_dir=tmp_path,
        sqlite_path=tmp_path / "agent.sqlite",
        chroma_dir=tmp_path / "chroma",
        token_log_path=tmp_path / "token_log.jsonl",
    )

    with patch(
        "agent_system.scout.agent._fetch_anthropic", side_effect=RuntimeError("boom")
    ), patch("agent_system.scout.agent._fetch_rss", return_value=[good_post]):
        scout = Scout(settings)
        plan = SourcePlan(
            sources_to_query=["anthropic", "deepmind"],
            time_window_days=14,
            filter_keywords=[],
            max_posts=20,
        )
        posts = scout.fetch(plan)

    assert [p.post_id for p in posts] == ["ok-1"]
