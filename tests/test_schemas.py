"""Schema round-trip tests."""

from __future__ import annotations

import json

import pytest

from agent_system.schemas import (
    AgentTrace,
    AnalyzedPost,
    Claim,
    ClaimVerdict,
    CriticReport,
    DraftSynthesis,
    ExecutionPlan,
    RawPost,
    SourcePlan,
    UserProfile,
    VerifiedSynthesis,
)


def _roundtrip(obj):
    """Serialize → JSON string → parse → from_dict → equal."""

    data = obj.to_dict()
    rehydrated = type(obj).from_dict(json.loads(json.dumps(data)))
    return rehydrated


# ---------------------------------------------------------------------------
# Constructors + round-trip
# ---------------------------------------------------------------------------


def test_user_profile_roundtrip():
    obj = UserProfile(
        user_id="u1",
        interests=["agents", "rag"],
        role_target="Solutions Engineer",
        seniority="early-career",
        reading_history=["p1"],
        feedback_log=[{"post_id": "p1", "rating": 1}],
    )
    assert _roundtrip(obj) == obj


def test_raw_post_roundtrip():
    obj = RawPost(
        post_id="p1",
        source="anthropic",
        url="https://x/y",
        title="t",
        authors=["a"],
        published_at="2026-04-25",
        content="body",
        content_type="blog",
    )
    assert _roundtrip(obj) == obj


def test_raw_post_rejects_bad_content_type():
    with pytest.raises(ValueError):
        RawPost(
            post_id="p1", source="anthropic", url="u", title="t",
            authors=[], published_at="x", content="c", content_type="podcast",
        )


def test_analyzed_post_roundtrip():
    obj = AnalyzedPost(
        post_id="p1",
        category="capability",
        key_claim="claim",
        practitioner_takeaway="do x",
        ships_in_product=False,
        concepts_introduced=["c"],
        relation_to_prior=[{"post_id": "p2", "relation": "extends"}],
        confidence=0.8,
        evidence_quotes=["q"],
    )
    assert _roundtrip(obj) == obj


def test_analyzed_post_provenance_fields_roundtrip():
    """P0-1: title/source/organization/url/published_at/content_type/authors
    must survive a to_dict -> JSON -> from_dict cycle, since this is
    exactly what SQLite's payload_json does on every save/load."""

    obj = AnalyzedPost(
        post_id="p1",
        category="capability",
        key_claim="claim",
        practitioner_takeaway="do x",
        ships_in_product=False,
        concepts_introduced=["c"],
        relation_to_prior=[],
        confidence=0.8,
        evidence_quotes=["q"],
        title="Example Agent Memory Paper",
        source="anthropic",
        url="https://anthropic.com/news/example",
        authors=["Jane Doe"],
        published_at="2026-05-01T00:00:00+00:00",
        content_type="blog",
        organization="Anthropic",
    )
    rt = _roundtrip(obj)
    assert rt == obj
    assert rt.title == "Example Agent Memory Paper"
    assert rt.source == "anthropic"
    assert rt.organization == "Anthropic"
    assert rt.url == "https://anthropic.com/news/example"
    assert rt.published_at == "2026-05-01T00:00:00+00:00"


def test_analyzed_post_from_dict_backward_compatible_without_provenance():
    """A payload_json blob written before this field set existed (no
    title/source/organization/... keys) must still deserialize instead
    of raising — this is what makes the P0-1 fix backward compatible
    with data already sitting in SQLite/Chroma."""

    old_style = {
        "post_id": "p1",
        "category": "capability",
        "key_claim": "claim",
        "practitioner_takeaway": "do x",
        "ships_in_product": None,
        "concepts_introduced": [],
        "relation_to_prior": [],
        "confidence": 0.5,
        "evidence_quotes": [],
    }
    obj = AnalyzedPost.from_dict(old_style)
    assert obj.title == ""
    assert obj.source == ""
    assert obj.organization == ""
    assert obj.url == ""
    assert obj.authors == []
    # P0-B: a dict with no schema_version key predates the concept
    # entirely and must default to 1 (always stale), never to CURRENT —
    # defaulting to CURRENT here would silently treat genuinely
    # incomplete old data as trustworthy.
    assert obj.schema_version == 1


def test_analyzed_post_new_construction_stamps_current_schema_version():
    from agent_system.schemas import CURRENT_ANALYSIS_SCHEMA_VERSION

    obj = AnalyzedPost(
        post_id="p1", category="capability", key_claim="x",
        practitioner_takeaway="y", ships_in_product=None,
        concepts_introduced=[], relation_to_prior=[], confidence=0.5,
        evidence_quotes=[],
    )
    assert obj.schema_version == CURRENT_ANALYSIS_SCHEMA_VERSION


def test_analyzed_post_confidence_bounds():
    base = dict(
        post_id="p1", category="capability", key_claim="x",
        practitioner_takeaway="y", ships_in_product=None,
        concepts_introduced=[], relation_to_prior=[], evidence_quotes=[],
    )
    with pytest.raises(ValueError):
        AnalyzedPost(**base, confidence=1.2)
    with pytest.raises(ValueError):
        AnalyzedPost(**base, confidence=-0.1)


def test_analyzed_post_rejects_bad_category():
    with pytest.raises(ValueError):
        AnalyzedPost(
            post_id="p1", category="vibes", key_claim="x",
            practitioner_takeaway="y", ships_in_product=None,
            concepts_introduced=[], relation_to_prior=[],
            confidence=0.5, evidence_quotes=[],
        )


def test_analyzed_post_rejects_bad_relation():
    with pytest.raises(ValueError):
        AnalyzedPost(
            post_id="p1", category="capability", key_claim="x",
            practitioner_takeaway="y", ships_in_product=None,
            concepts_introduced=[],
            relation_to_prior=[{"post_id": "p2", "relation": "vibes"}],
            confidence=0.5, evidence_quotes=[],
        )


def test_claim_roundtrip():
    obj = Claim(text="t", supporting_post_ids=["p"], supporting_quotes=["q"])
    assert _roundtrip(obj) == obj


def test_draft_synthesis_roundtrip_with_nested_claims():
    obj = DraftSynthesis(
        synthesis_type="digest",
        title="Title",
        sections=[
            {
                "heading": "h",
                "claims": [Claim(text="t", supporting_post_ids=["p"])],
                "prose": "p",
            }
        ],
        posts_covered=["p"],
    )
    rt = _roundtrip(obj)
    assert rt.synthesis_type == "digest"
    assert rt.sections[0]["claims"][0].text == "t"


def test_draft_rejects_bad_synthesis_type():
    with pytest.raises(ValueError):
        DraftSynthesis(
            synthesis_type="vibes",
            title="t",
            sections=[],
            posts_covered=[],
        )


def test_claim_verdict_roundtrip():
    obj = ClaimVerdict(
        claim=Claim(text="t", supporting_post_ids=["p"]),
        verdict="supported",
        reasoning="because",
    )
    rt = _roundtrip(obj)
    assert rt == obj


def test_claim_verdict_rejects_bad_verdict():
    with pytest.raises(ValueError):
        ClaimVerdict(
            claim=Claim(text="t", supporting_post_ids=["p"]),
            verdict="lol",
            reasoning="",
        )


def test_claim_verdict_accepts_verification_unavailable():
    """P1-A: distinct from "unsupported" — a transient Gemini failure
    is not the same claim as "evidence doesn't support this"."""

    obj = ClaimVerdict(
        claim=Claim(text="t", supporting_post_ids=["p"]),
        verdict="verification_unavailable",
        reasoning="Gemini 503 after retries.",
    )
    rt = _roundtrip(obj)
    assert rt.verdict == "verification_unavailable"


def test_critic_report_roundtrip():
    obj = CriticReport(
        verdicts=[
            ClaimVerdict(
                claim=Claim(text="t", supporting_post_ids=["p"]),
                verdict="supported",
                reasoning="ok",
            )
        ],
        num_unsupported=0,
        revision_needed=False,
        revision_notes="",
    )
    rt = _roundtrip(obj)
    assert rt.num_unsupported == 0
    assert rt.verdicts[0].verdict == "supported"


def test_verified_synthesis_roundtrip():
    draft = DraftSynthesis(
        synthesis_type="digest", title="t", sections=[], posts_covered=[]
    )
    report = CriticReport(
        verdicts=[], num_unsupported=0, revision_needed=False, revision_notes=""
    )
    obj = VerifiedSynthesis(
        draft=draft, critic_report=report, revision_count=0, final=True
    )
    rt = _roundtrip(obj)
    assert rt.final is True
    assert rt.draft.synthesis_type == "digest"
    assert rt.citations == {}


def test_verified_synthesis_citations_roundtrip():
    """P0-1 item 5: the final answer object must carry a post_id ->
    {title, source, organization, published_at, url} map so a bare
    citation is resolvable, and that map must survive persistence."""

    draft = DraftSynthesis(
        synthesis_type="digest", title="t", sections=[], posts_covered=["p1"]
    )
    report = CriticReport(
        verdicts=[], num_unsupported=0, revision_needed=False, revision_notes=""
    )
    obj = VerifiedSynthesis(
        draft=draft,
        critic_report=report,
        revision_count=0,
        final=True,
        citations={
            "p1": {
                "title": "Example Post",
                "source": "anthropic",
                "organization": "Anthropic",
                "published_at": "2026-05-01",
                "url": "https://anthropic.com/news/example",
            }
        },
    )
    rt = _roundtrip(obj)
    assert rt.citations["p1"]["title"] == "Example Post"
    assert rt.citations["p1"]["organization"] == "Anthropic"


def test_verified_synthesis_from_dict_backward_compatible_without_citations():
    draft = DraftSynthesis(
        synthesis_type="digest", title="t", sections=[], posts_covered=[]
    )
    report = CriticReport(
        verdicts=[], num_unsupported=0, revision_needed=False, revision_notes=""
    )
    old_style = {"draft": draft.to_dict(), "critic_report": report.to_dict(), "final": True}
    obj = VerifiedSynthesis.from_dict(old_style)
    assert obj.citations == {}


def test_source_plan_roundtrip():
    obj = SourcePlan(
        sources_to_query=["anthropic"],
        time_window_days=7,
        filter_keywords=["x"],
        max_posts=10,
    )
    assert _roundtrip(obj) == obj


def test_execution_plan_roundtrip():
    obj = ExecutionPlan(
        intent="digest",
        source_plan=SourcePlan(
            sources_to_query=["anthropic"],
            time_window_days=7,
            filter_keywords=[],
            max_posts=10,
        ),
        synthesis_type="digest",
        user_profile_summary="u",
        reasoning="r",
    )
    rt = _roundtrip(obj)
    assert rt.source_plan.sources_to_query == ["anthropic"]
    assert rt.research_targets == []


def test_execution_plan_research_targets_roundtrip():
    """P0-A: research_targets is a distinct concept from
    source_plan.sources_to_query — different vocabulary (entity display
    names, not source ids) and must survive persistence independently."""

    obj = ExecutionPlan(
        intent="comparison",
        source_plan=SourcePlan(
            sources_to_query=["openai", "anthropic", "arxiv_cs_ai"],
            time_window_days=365,
            filter_keywords=[],
            max_posts=24,
        ),
        synthesis_type="comparison",
        user_profile_summary="u",
        reasoning="r",
        research_targets=["OpenAI", "Anthropic"],
    )
    rt = _roundtrip(obj)
    assert rt.research_targets == ["OpenAI", "Anthropic"]
    assert "arxiv_cs_ai" not in rt.research_targets


def test_execution_plan_from_dict_backward_compatible_without_research_targets():
    old_style = {
        "intent": "digest",
        "source_plan": {
            "sources_to_query": ["anthropic"], "time_window_days": 7,
            "filter_keywords": [], "max_posts": 10,
        },
        "synthesis_type": "digest",
        "user_profile_summary": "u",
        "reasoning": "r",
    }
    obj = ExecutionPlan.from_dict(old_style)
    assert obj.research_targets == []


def test_execution_plan_rejects_bad_intent():
    with pytest.raises(ValueError):
        ExecutionPlan(
            intent="vibes",
            source_plan=SourcePlan(
                sources_to_query=[], time_window_days=7,
                filter_keywords=[], max_posts=10,
            ),
            synthesis_type="digest",
            user_profile_summary="",
            reasoning="",
        )


def test_agent_trace_roundtrip():
    obj = AgentTrace(
        trace_id="t1",
        agent_name="orchestrator",
        started_at="2026-04-28T00:00:00",
        ended_at="2026-04-28T00:00:01",
        input_tokens=10,
        output_tokens=20,
        model="gemini-2.0-flash",
        cost_estimate=0.0001,
        status="ok",
    )
    assert _roundtrip(obj) == obj


def test_agent_trace_rejects_bad_status():
    with pytest.raises(ValueError):
        AgentTrace(
            trace_id="t",
            agent_name="a",
            started_at="x",
            ended_at="y",
            status="dunno",
        )
