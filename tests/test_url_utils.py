"""Tests for URL canonicalization (P1-B)."""

from __future__ import annotations

from agent_system.scout.url_utils import canonicalize_url


def test_trailing_slash_is_normalized():
    assert canonicalize_url("https://example.com/article/") == canonicalize_url(
        "https://example.com/article"
    )


def test_utm_tracking_params_are_stripped():
    assert canonicalize_url(
        "https://example.com/article?utm_source=x&utm_medium=y"
    ) == canonicalize_url("https://example.com/article")


def test_other_tracking_params_are_stripped():
    assert canonicalize_url("https://example.com/article?gclid=abc123") == canonicalize_url(
        "https://example.com/article"
    )


def test_fragment_is_removed():
    assert canonicalize_url("https://example.com/article#section-2") == canonicalize_url(
        "https://example.com/article"
    )


def test_all_variants_from_the_task_brief_canonicalize_identically():
    variants = [
        "https://example.com/article",
        "https://example.com/article/",
        "https://example.com/article?utm_source=x",
        "https://example.com/article#section",
    ]
    canonical = {canonicalize_url(v) for v in variants}
    assert len(canonical) == 1


def test_scheme_and_host_case_is_normalized():
    assert canonicalize_url("HTTPS://Example.COM/article") == canonicalize_url(
        "https://example.com/article"
    )


def test_non_tracking_query_params_are_preserved():
    """Must not over-canonicalize — a genuinely different article (a
    real query param that selects different content) must not collapse
    to the same key as the bare URL."""

    assert canonicalize_url("https://example.com/article?id=42") != canonicalize_url(
        "https://example.com/article"
    )


def test_non_tracking_query_param_order_is_stabilized():
    assert canonicalize_url("https://example.com/a?b=2&a=1") == canonicalize_url(
        "https://example.com/a?a=1&b=2"
    )


def test_root_path_is_preserved_not_stripped_to_empty():
    assert canonicalize_url("https://example.com/").endswith("/")


def test_empty_and_none_like_input_returns_empty_string():
    assert canonicalize_url("") == ""
    assert canonicalize_url("   ") == ""


def test_distinct_articles_stay_distinct():
    assert canonicalize_url("https://example.com/article-a") != canonicalize_url(
        "https://example.com/article-b"
    )
