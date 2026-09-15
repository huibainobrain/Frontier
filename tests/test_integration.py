"""End-to-end integration test.

Boots a real Orchestrator, replaces the **ADK seams** and **Scout**
(the only sub-agent whose real implementation touches the network) so
the whole run is fully offline and deterministic, and drives the real
Analyst / Synthesizer / Critic classes through mocked Gemini calls.

What this test proves:

* The schemas, storage, prompts, observability, guardrails, and
  Orchestrator all import together cleanly.
* The pipeline produces a ``VerifiedSynthesis`` with ``final=True``.
* Storage and token-log writes happen on a tmp dir.

Scout is mocked rather than driven for real: its real implementation
hits live RSS/arXiv/Anthropic-sitemap endpoints, which would make this
test's outcome depend on network availability and on whatever those
sources happen to return that day — exactly the kind of flakiness a
test named "integration" shouldn't have. Scout's own behavior (source
dispatch, dedup, per-source failure isolation, Anthropic discovery) has
its own dedicated tests in test_scout.py / test_anthropic_source.py.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from agent_system.config import Settings
from agent_system.schemas import (
    RawPost,
    SourcePlan,
    UserProfile,
    VerifiedSynthesis,
)


def _make_settings(tmp_path: Path) -> Settings:
    return Settings(
        google_api_key="test-key",
        data_dir=tmp_path,
        sqlite_path=tmp_path / "agent.sqlite",
        chroma_dir=tmp_path / "chroma",
        token_log_path=tmp_path / "token_log.jsonl",
        max_revisions=2,
    )


def test_end_to_end_with_stubs(tmp_path):
    settings = _make_settings(tmp_path)

    # Patch the ADK factories so __init__ doesn't try to build real Agents
    # (which would need google-adk installed and a real API key).
    with patch(
        "agent_system.orchestrator.agent.build_intent_agent", return_value=MagicMock()
    ), patch(
        "agent_system.orchestrator.agent.build_planner_agent", return_value=MagicMock()
    ), patch(
        "agent_system.orchestrator.agent.build_evaluator_agent", return_value=MagicMock()
    ), patch(
        "agent_system.orchestrator.agent.build_pipeline", return_value=MagicMock()
    ), patch(
        "agent_system.orchestrator.agent.InMemorySessionService"
    ), patch(
        "agent_system.observability.tracing._token_log_path",
        return_value=settings.token_log_path,
    ):
        from agent_system.orchestrator.agent import (
            EVALUATOR_OUTPUT_KEY,
            INTENT_OUTPUT_KEY,
            PLAN_OUTPUT_KEY,
            Orchestrator,
        )

        # Real Settings, real sub-agents (default constructed inside
        # Orchestrator.__init__ from agent_system.{scout,analyst,...}.agent).
        orch = Orchestrator(settings)

        # Replace the two ADK seams with AsyncMocks so no real LLM call.
        orch._create_session = AsyncMock(return_value="sess-it")

        async def fake_invoke(
            agent, user_text, profile, session_id, output_key, *, component=None
        ):
            if output_key == INTENT_OUTPUT_KEY:
                return {
                    "intent": "digest",
                    "confidence": 0.95,
                    "reasoning": "Recency window.",
                }
            if output_key == PLAN_OUTPUT_KEY:
                return {
                    "source_plan": {
                        "sources_to_query": ["anthropic", "openai"],
                        "time_window_days": 7,
                        "filter_keywords": ["agentic"],
                        "max_posts": 10,
                    },
                    "synthesis_type": "digest",
                    "reasoning": "Default rules applied.",
                }
            if output_key == EVALUATOR_OUTPUT_KEY:
                # Round 1 evidence is sufficient — no replan, matching
                # this test's "single fetch, single analyze" assertions.
                return {
                    "is_sufficient": True,
                    "covered_dimensions": [],
                    "evidence_gaps": [],
                    "continue_research": False,
                    "next_actions": [],
                    "stop_reason": "Sufficient for this stubbed run.",
                }
            raise AssertionError(f"unexpected output_key={output_key!r}")

        orch._invoke_agent = AsyncMock(side_effect=fake_invoke)

        # Replace Scout with a mock so the test never touches the network
        # (its real implementation hits live RSS/arXiv/Anthropic-sitemap
        # endpoints). post_id="post-stub" matches what the Synthesizer's
        # mocked response below cites, so posts_covered comes out non-empty.
        orch.scout = MagicMock()
        orch.scout.plan_sources.return_value = SourcePlan(
            sources_to_query=["anthropic"],
            time_window_days=7,
            filter_keywords=["agentic"],
            max_posts=10,
        )
        orch.scout.fetch.return_value = [
            RawPost(
                post_id="post-stub",
                source="anthropic",
                url="https://example.com/post-stub",
                title="Stub Post",
                authors=["A. N. Other"],
                published_at="2026-04-28T00:00:00",
                content="Stub content about agentic AI systems.",
                content_type="blog",
            )
        ]

        # Mock Gemini calls in Analyst / Synthesizer so the integration test
        # never hits the real API.
        orch.analyst._call_gemini = MagicMock(
            return_value={
                "category": "capability",
                "key_claim": "Stub claim from integration test.",
                "practitioner_takeaway": "Stub takeaway.",
                "concepts_introduced": ["stub"],
                "confidence": 0.8,
                "evidence_quotes": ["stub evidence"],
            }
        )
        orch.synthesizer._call_gemini = MagicMock(
            return_value={
                "title": "Weekly AI Digest (integration test)",
                "sections": [
                    {
                        "heading": "Headline",
                        "claims": [
                            {
                                "text": "Stub claim",
                                "supporting_post_ids": ["post-stub"],
                                "supporting_quotes": ["evidence"],
                            }
                        ],
                        "prose": "Integration test prose.",
                    }
                ],
            }
        )

        profile = UserProfile(
            user_id="u-int",
            interests=["agentic AI", "RAG"],
            role_target="Solutions Engineer",
            seniority="early-career",
            reading_history=[],
            feedback_log=[],
        )

        verified = orch.run("Give me this week's frontier AI digest.", profile)

    assert isinstance(verified, VerifiedSynthesis)
    assert verified.final is True
    # The Synthesizer stub produces a digest with one section.
    assert verified.draft.synthesis_type == "digest"
    assert len(verified.draft.posts_covered) >= 1

    # Storage went to the tmp DB.
    assert settings.sqlite_path.exists()
    # Trace log got entries from the @traced/trace_span calls.
    assert settings.token_log_path.exists()
    assert settings.token_log_path.stat().st_size > 0
