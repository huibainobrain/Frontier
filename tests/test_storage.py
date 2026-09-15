"""Storage layer tests.

SQLite uses a tmp dir; ChromaDB tests pass embeddings explicitly so
no network calls happen.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agent_system.config import Settings
from agent_system.schemas import (
    AgentTrace,
    AnalyzedPost,
    Claim,
    ClaimVerdict,
    CriticReport,
    DraftSynthesis,
    RawPost,
    VerifiedSynthesis,
)
from agent_system.storage import db


def _make_settings(tmp_path: Path) -> Settings:
    return Settings(
        google_api_key="test-key",
        data_dir=tmp_path,
        sqlite_path=tmp_path / "test.sqlite",
        chroma_dir=tmp_path / "chroma",
        token_log_path=tmp_path / "token_log.jsonl",
    )


# ---------------------------------------------------------------------------
# SQLite
# ---------------------------------------------------------------------------


def test_init_db_creates_tables(tmp_path):
    settings = _make_settings(tmp_path)
    db.init_db(settings)
    assert settings.sqlite_path.exists()


def test_raw_post_roundtrip(tmp_path):
    settings = _make_settings(tmp_path)
    post = RawPost(
        post_id="p1",
        source="anthropic",
        url="https://x/y",
        title="title",
        authors=["a"],
        published_at="2026-04-25",
        content="body",
        content_type="blog",
    )
    db.save_raw_post(post, settings=settings)
    loaded = db.get_raw_post("p1", settings=settings)
    assert loaded == post


def test_list_raw_posts_filters_by_source(tmp_path):
    settings = _make_settings(tmp_path)
    for i, source in enumerate(["anthropic", "openai", "anthropic"]):
        db.save_raw_post(
            RawPost(
                post_id=f"p{i}",
                source=source,
                url="u",
                title="t",
                authors=[],
                published_at="2026-04-25",
                content="c",
                content_type="blog",
            ),
            settings=settings,
        )
    anth = db.list_raw_posts(source="anthropic", settings=settings)
    assert len(anth) == 2
    all_posts = db.list_raw_posts(settings=settings)
    assert len(all_posts) == 3


def test_analyzed_post_roundtrip(tmp_path):
    settings = _make_settings(tmp_path)
    post = AnalyzedPost(
        post_id="p1",
        category="capability",
        key_claim="claim",
        practitioner_takeaway="do x",
        ships_in_product=False,
        concepts_introduced=["c"],
        relation_to_prior=[],
        confidence=0.7,
        evidence_quotes=["q"],
    )
    db.save_analyzed_post(post, settings=settings)
    assert db.get_analyzed_post("p1", settings=settings) == post


# ---------------------------------------------------------------------------
# P0-B: Analysis schema versioning — stale rows must not be reused
# ---------------------------------------------------------------------------


def _versioned_post(post_id: str = "p1", *, schema_version: int) -> AnalyzedPost:
    return AnalyzedPost(
        post_id=post_id,
        category="capability",
        key_claim="claim",
        practitioner_takeaway="do x",
        ships_in_product=False,
        concepts_introduced=["c"],
        relation_to_prior=[],
        confidence=0.7,
        evidence_quotes=["q"],
        title="A Title",
        source="anthropic",
        organization="Anthropic",
        schema_version=schema_version,
    )


def test_stale_schema_version_is_not_reused(tmp_path):
    """Test B1: a record saved under an old schema_version must come
    back as a miss (None), not be handed back as a valid cache hit."""

    from agent_system.schemas import CURRENT_ANALYSIS_SCHEMA_VERSION

    settings = _make_settings(tmp_path)
    stale = _versioned_post("p1", schema_version=1)
    db.save_analyzed_post(stale, settings=settings)

    assert CURRENT_ANALYSIS_SCHEMA_VERSION > 1  # sanity-check the fixture is actually stale
    assert db.get_analyzed_post("p1", settings=settings) is None


def test_current_schema_version_is_reused(tmp_path):
    """Test B2: a record saved under the current schema_version is a
    normal cache hit."""

    from agent_system.schemas import CURRENT_ANALYSIS_SCHEMA_VERSION

    settings = _make_settings(tmp_path)
    current = _versioned_post("p1", schema_version=CURRENT_ANALYSIS_SCHEMA_VERSION)
    db.save_analyzed_post(current, settings=settings)

    fetched = db.get_analyzed_post("p1", settings=settings)
    assert fetched is not None
    assert fetched == current


def test_analyzed_post_without_schema_version_column_defaults_to_stale(tmp_path):
    """A row saved before this field existed at all (payload_json has
    no "schema_version" key, exactly like the real pre-P0-1 rows found
    in local dev databases) must default to version 1 on load — and
    therefore also be treated as stale, not crash."""

    import json as _json

    settings = _make_settings(tmp_path)
    db.init_db(settings)
    pre_versioning_payload = {
        "post_id": "p1",
        "category": "capability",
        "key_claim": "claim",
        "practitioner_takeaway": "do x",
        "ships_in_product": None,
        "concepts_introduced": [],
        "relation_to_prior": [],
        "confidence": 0.5,
        "evidence_quotes": [],
        # No "schema_version" key, no title/source/... — the real
        # pre-P0-1 row shape.
    }
    with db._conn(settings) as c:
        c.execute(
            "INSERT INTO analyzed_posts (post_id, category, confidence, "
            "analyzed_at, payload_json) VALUES (?, ?, ?, ?, ?)",
            ("p1", "capability", 0.5, "2026-01-01T00:00:00", _json.dumps(pre_versioning_payload)),
        )
    assert db.get_analyzed_post("p1", settings=settings) is None


def test_list_analyzed_posts_filters_by_category_and_confidence(tmp_path):
    settings = _make_settings(tmp_path)
    for i, (cat, conf) in enumerate(
        [("capability", 0.9), ("safety", 0.4), ("capability", 0.3)]
    ):
        db.save_analyzed_post(
            AnalyzedPost(
                post_id=f"p{i}",
                category=cat,
                key_claim="x",
                practitioner_takeaway="y",
                ships_in_product=None,
                concepts_introduced=[],
                relation_to_prior=[],
                confidence=conf,
                evidence_quotes=[],
            ),
            settings=settings,
        )
    high_conf = db.list_analyzed_posts(min_confidence=0.5, settings=settings)
    assert {p.post_id for p in high_conf} == {"p0"}
    cap = db.list_analyzed_posts(category="capability", settings=settings)
    assert {p.post_id for p in cap} == {"p0", "p2"}


def test_save_synthesis_roundtrip(tmp_path):
    settings = _make_settings(tmp_path)
    draft = DraftSynthesis(
        synthesis_type="digest",
        title="t",
        sections=[
            {
                "heading": "h",
                "claims": [Claim(text="c", supporting_post_ids=["p"])],
                "prose": "p",
            }
        ],
        posts_covered=["p"],
    )
    report = CriticReport(
        verdicts=[
            ClaimVerdict(
                claim=Claim(text="c", supporting_post_ids=["p"]),
                verdict="supported",
                reasoning="ok",
            )
        ],
        num_unsupported=0,
        revision_needed=False,
    )
    verified = VerifiedSynthesis(
        draft=draft, critic_report=report, revision_count=0, final=True
    )
    sid = db.save_synthesis(verified, settings=settings)
    rt = db.get_synthesis(sid, settings=settings)
    assert rt is not None
    assert rt.draft.title == "t"
    assert rt.critic_report.verdicts[0].verdict == "supported"


def test_user_feedback_save(tmp_path):
    settings = _make_settings(tmp_path)
    fid = db.save_user_feedback(
        {
            "user_id": "u1",
            "intent": "digest",
            "synthesis_type": "digest",
            "final": True,
            "revision_count": 0,
            "logged_at": "2026-04-28T00:00:00Z",
            "extra": "anything",
        },
        settings=settings,
    )
    assert fid > 0


def test_agent_trace_roundtrip(tmp_path):
    settings = _make_settings(tmp_path)
    trace = AgentTrace(
        trace_id="t1",
        agent_name="orchestrator.intent",
        started_at="2026-04-28T00:00:00",
        ended_at="2026-04-28T00:00:01",
        input_tokens=10,
        output_tokens=20,
        model="gemini-2.0-flash",
        cost_estimate=0.0001,
        status="ok",
    )
    db.save_agent_trace(trace, settings=settings)
    rows = db.list_agent_traces(settings=settings)
    assert len(rows) == 1
    assert rows[0] == trace


# ---------------------------------------------------------------------------
# Vectors (ChromaDB) — embeddings injected to skip network
# ---------------------------------------------------------------------------


def test_vector_add_and_search_with_explicit_embeddings(tmp_path):
    pytest.importorskip("chromadb")
    from agent_system.storage import vectors

    settings = _make_settings(tmp_path)

    post_a = AnalyzedPost(
        post_id="a",
        category="capability",
        key_claim="agentic systems orchestrate sub-agents",
        practitioner_takeaway="design for routing",
        ships_in_product=None,
        concepts_introduced=["agent"],
        relation_to_prior=[],
        confidence=0.9,
        evidence_quotes=[],
    )
    post_b = AnalyzedPost(
        post_id="b",
        category="safety",
        key_claim="constitutional AI uses self-critique",
        practitioner_takeaway="audit critic prompts",
        ships_in_product=None,
        concepts_introduced=["constitution"],
        relation_to_prior=[],
        confidence=0.85,
        evidence_quotes=[],
    )
    db.save_analyzed_post(post_a, settings=settings)
    db.save_analyzed_post(post_b, settings=settings)

    # Inject simple deterministic vectors so no embedding API call occurs.
    vec_a = [1.0, 0.0, 0.0]
    vec_b = [0.0, 1.0, 0.0]
    vectors.add_analyzed_post(post_a, embedding=vec_a, settings=settings)
    vectors.add_analyzed_post(post_b, embedding=vec_b, settings=settings)

    hits = vectors.search_similar(
        query="agentic", k=2, embedding=vec_a, settings=settings
    )
    assert hits, "expected at least one hit"
    # The closest-by-vector hit should be post_a.
    assert hits[0].post_id == "a"


def test_stale_analyzed_post_not_returned_by_rag_search(tmp_path):
    """Test B3: a record that predates the current analysis schema
    must not surface as valid RAG memory, even though its vector is
    still sitting in Chroma and would otherwise be the closest match —
    search_similar hydrates every hit through get_analyzed_post, which
    already filters stale rows (P0-B), so the stale hit is silently
    skipped rather than handed back as "high-confidence prior
    knowledge"."""

    pytest.importorskip("chromadb")
    from agent_system.storage import vectors

    settings = _make_settings(tmp_path)

    stale = AnalyzedPost(
        post_id="stale-1",
        category="capability",
        key_claim="a stale pre-P0-1 analysis",
        practitioner_takeaway="do x",
        ships_in_product=None,
        concepts_introduced=["agent"],
        relation_to_prior=[],
        confidence=0.95,
        evidence_quotes=[],
        schema_version=1,
    )
    fresh = AnalyzedPost(
        post_id="fresh-1",
        category="capability",
        key_claim="an unrelated fresh analysis",
        practitioner_takeaway="do y",
        ships_in_product=None,
        concepts_introduced=["unrelated"],
        relation_to_prior=[],
        confidence=0.5,
        evidence_quotes=[],
    )
    db.save_analyzed_post(stale, settings=settings)
    db.save_analyzed_post(fresh, settings=settings)

    query_vec = [1.0, 0.0, 0.0]
    # The stale post gets the *closest* vector on purpose — if
    # staleness weren't enforced, it would rank first.
    vectors.add_analyzed_post(stale, embedding=[1.0, 0.0, 0.0], settings=settings)
    vectors.add_analyzed_post(fresh, embedding=[0.0, 1.0, 0.0], settings=settings)

    hits = vectors.search_similar(query="agent", k=2, embedding=query_vec, settings=settings)
    assert "stale-1" not in {h.post_id for h in hits}
