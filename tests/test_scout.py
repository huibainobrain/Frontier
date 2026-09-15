"""Tests for the Scout agent.

Network-touching fetchers (_fetch_rss, _fetch_arxiv, _fetch_semantic_scholar,
_fetch_openalex) are mocked at the module level — no network access needed.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from agent_system.config import Settings
from agent_system.schemas import RawPost, SourcePlan, UserProfile


def _make_settings(tmp_path: Path) -> Settings:
    return Settings(
        google_api_key="test-key",
        data_dir=tmp_path,
        sqlite_path=tmp_path / "agent.sqlite",
        chroma_dir=tmp_path / "chroma",
        token_log_path=tmp_path / "token_log.jsonl",
    )


def _raw_post(post_id: str, source: str = "deepmind") -> RawPost:
    return RawPost(
        post_id=post_id,
        source=source,
        url=f"https://example.com/{post_id}",
        title=f"Post {post_id}",
        authors=["A. N. Other"],
        published_at="2026-04-28T00:00:00",
        content="Body text about agentic AI systems.",
        content_type="blog",
    )


# ---------------------------------------------------------------------------
# plan_sources — pure heuristic, no network
# ---------------------------------------------------------------------------


class TestPlanSources:
    def test_returns_source_plan(self, tmp_path):
        from agent_system.scout.agent import Scout

        scout = Scout(_make_settings(tmp_path))
        profile = UserProfile(
            user_id="u1",
            interests=["agentic AI"],
            role_target="Engineer",
            seniority="mid",
        )
        plan = scout.plan_sources(
            "What's new from OpenAI and DeepMind on agentic RAG this week?", profile
        )

        assert isinstance(plan, SourcePlan)
        assert plan.sources_to_query
        assert plan.time_window_days > 0
        assert plan.max_posts > 0

    def test_extract_keywords_drops_stopwords(self):
        from agent_system.scout.agent import Scout

        keywords = Scout._extract_keywords(
            "What is the latest and new update on agentic RAG systems?"
        )
        assert "agentic" in keywords
        assert "systems" in keywords
        assert "the" not in keywords
        assert "what" not in keywords


# ---------------------------------------------------------------------------
# _post_id — P1-B: must hash the canonical URL, not the raw one
# ---------------------------------------------------------------------------


class TestPostIdCanonicalization:
    def test_post_id_stable_across_tracking_params(self):
        from agent_system.scout.agent import _post_id

        a = _post_id("anthropic", "https://example.com/article?utm_source=newsletter")
        b = _post_id("anthropic", "https://example.com/article")
        assert a == b

    def test_post_id_stable_across_trailing_slash(self):
        from agent_system.scout.agent import _post_id

        a = _post_id("anthropic", "https://example.com/article/")
        b = _post_id("anthropic", "https://example.com/article")
        assert a == b

    def test_post_id_differs_for_different_articles(self):
        from agent_system.scout.agent import _post_id

        a = _post_id("anthropic", "https://example.com/article-a")
        b = _post_id("anthropic", "https://example.com/article-b")
        assert a != b


# ---------------------------------------------------------------------------
# fetch — network fetchers mocked
# ---------------------------------------------------------------------------


class TestFetch:
    def test_fetch_dedupes_across_sources(self, tmp_path):
        from agent_system.scout.agent import Scout

        dup = _raw_post("dup-1", source="deepmind")
        only_hf = _raw_post("hf-1", source="hugging_face")

        def rss_side_effect(source_id, since=None, query="", max_results=5):
            return {"deepmind": [dup], "hugging_face": [only_hf]}.get(source_id, [])

        with patch("agent_system.scout.agent._fetch_rss", side_effect=rss_side_effect):
            scout = Scout(_make_settings(tmp_path))
            plan = SourcePlan(
                sources_to_query=["deepmind", "hugging_face"],
                time_window_days=14,
                filter_keywords=[],
                max_posts=20,
            )
            posts = scout.fetch(plan)

        assert {p.post_id for p in posts} == {"dup-1", "hf-1"}

    def test_fetch_respects_max_posts(self, tmp_path):
        from agent_system.scout.agent import Scout

        many = [_raw_post(f"p{i}") for i in range(10)]

        with patch("agent_system.scout.agent._fetch_rss", return_value=many):
            scout = Scout(_make_settings(tmp_path))
            plan = SourcePlan(
                sources_to_query=["deepmind"],
                time_window_days=14,
                filter_keywords=[],
                max_posts=3,
            )
            posts = scout.fetch(plan)

        assert len(posts) == 3

    def test_fetch_survives_one_source_failing(self, tmp_path):
        """A dead/erroring feed must not take down the other sources."""

        from agent_system.scout.agent import Scout

        good = [_raw_post("ok-1")]

        def rss_side_effect(source_id, since=None, query="", max_results=5):
            if source_id == "deepmind":
                raise RuntimeError("feed unreachable")
            return good

        with patch("agent_system.scout.agent._fetch_rss", side_effect=rss_side_effect):
            scout = Scout(_make_settings(tmp_path))
            plan = SourcePlan(
                sources_to_query=["deepmind", "hugging_face"],
                time_window_days=14,
                filter_keywords=[],
                max_posts=20,
            )
            posts = scout.fetch(plan)

        assert [p.post_id for p in posts] == ["ok-1"]

    def test_fetch_unknown_source_is_skipped_not_fatal(self, tmp_path):
        from agent_system.scout.agent import Scout

        scout = Scout(_make_settings(tmp_path))
        plan = SourcePlan(
            sources_to_query=["not_a_real_source"],
            time_window_days=14,
            filter_keywords=[],
            max_posts=20,
        )
        posts = scout.fetch(plan)

        assert posts == []

    def test_fetch_persists_to_sqlite(self, tmp_path):
        """Fix for item 7: fetched posts must be cached to SQLite so the
        Analyst's own cache can find them on a later run."""

        from agent_system.scout.agent import Scout
        from agent_system.storage import db

        settings = _make_settings(tmp_path)
        posts = [_raw_post("cache-1"), _raw_post("cache-2")]

        with patch("agent_system.scout.agent._fetch_rss", return_value=posts):
            scout = Scout(settings)
            plan = SourcePlan(
                sources_to_query=["deepmind"],
                time_window_days=14,
                filter_keywords=[],
                max_posts=20,
            )
            scout.fetch(plan)

        stored = db.list_raw_posts(settings=settings)
        assert {p.post_id for p in stored} == {"cache-1", "cache-2"}


# ---------------------------------------------------------------------------
# Agent Action Fidelity: a query rewrite must actually change which RSS
# candidates come back, not just which academic-API call is made.
# ---------------------------------------------------------------------------


def _rss_entry(title: str, summary: str, url: str) -> SimpleNamespace:
    return SimpleNamespace(
        title=title, summary=summary, link=url, published_parsed=None
    )


class TestRssRelevanceRanking:
    def test_relevant_candidate_preferred_over_more_recent_ones(self):
        """Test A3: a candidate matching the current query must be
        preferred over purely-more-recent, unrelated ones — not just
        "the latest N regardless of query"."""

        from agent_system.scout.agent import _fetch_rss

        post_a = _rss_entry(
            "Unrelated latest roundup",
            "A general roundup of unrelated announcements.",
            "https://example.com/a",
        )
        post_b = _rss_entry(
            "Deep dive: agent memory and context engineering",
            "How our agents manage long-term memory via context engineering.",
            "https://example.com/b",
        )
        post_c = _rss_entry(
            "Another unrelated post", "More unrelated content.", "https://example.com/c"
        )
        fake_feed = SimpleNamespace(entries=[post_a, post_b, post_c])

        with patch("feedparser.parse", return_value=fake_feed):
            posts = _fetch_rss(
                "openai", query="agent memory context engineering", max_results=1
            )

        assert len(posts) == 1
        assert "context engineering" in posts[0].title.lower()

    def test_different_queries_change_selected_candidates(self):
        """Test A4: a Replanner query rewrite must actually change
        which RSS candidates are returned."""

        from agent_system.scout.agent import _fetch_rss

        post_safety = _rss_entry(
            "AI agent safety guidelines",
            "Guidelines for agent safety in deployment.",
            "https://example.com/safety",
        )
        post_interp = _rss_entry(
            "Mechanistic interpretability with sparse autoencoders",
            "New techniques for interpreting model internals via sparse autoencoders.",
            "https://example.com/interp",
        )
        fake_feed = SimpleNamespace(entries=[post_safety, post_interp])

        with patch("feedparser.parse", return_value=fake_feed):
            round1 = _fetch_rss("deepmind", query="agent safety", max_results=1)
            round2 = _fetch_rss(
                "deepmind",
                query="mechanistic interpretability sparse autoencoders",
                max_results=1,
            )

        assert round1[0].url != round2[0].url
        assert "safety" in round1[0].title.lower()
        assert "interpretability" in round2[0].title.lower()

    def test_no_query_signal_falls_back_to_recency(self):
        """An empty query (or one matching nothing in the pool) must
        degrade to the old latest-N behavior, not return nothing."""

        from agent_system.scout.agent import _fetch_rss

        newest = _rss_entry("Newest post", "Some content here.", "https://example.com/newest")
        older = _rss_entry("Older post", "Other content here.", "https://example.com/older")
        fake_feed = SimpleNamespace(entries=[newest, older])

        with patch("feedparser.parse", return_value=fake_feed):
            posts = _fetch_rss("openai", query="", max_results=1)

        assert len(posts) == 1
        assert posts[0].title == "Newest post"
