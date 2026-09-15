"""Tests for the hybrid Critic (deterministic pre-checks + LLM semantic
verification).

Convention matches Analyst/Synthesizer tests: construct via
``Critic.__new__(Critic)``, set ``.settings``/``._client`` directly, and
patch ``_get_client`` to avoid any real network call.
"""

import json
from unittest.mock import MagicMock, patch

from google.genai import errors as genai_errors

from agent_system.config import Settings
from agent_system.critic.agent import Critic
from agent_system.schemas import AnalyzedPost, Claim, DraftSynthesis


def _server_error(code: int = 503) -> genai_errors.ServerError:
    """A real google.genai ServerError, matching what the SDK actually
    raises on a 5xx (confirmed live in this project against a genuine
    Gemini 503 'high demand' response)."""

    return genai_errors.ServerError(code, {"message": "high demand"}, None)


def _client_error(code: int = 400) -> genai_errors.ClientError:
    return genai_errors.ClientError(code, {"message": "bad request"}, None)


def make_critic() -> Critic:
    critic = Critic.__new__(Critic)
    critic.settings = Settings(google_api_key="test-key")
    critic._client = MagicMock()
    return critic


def make_post(post_id: str, quote: str) -> AnalyzedPost:
    return AnalyzedPost(
        post_id=post_id,
        category="capability",
        key_claim=quote,
        practitioner_takeaway="Test takeaway.",
        ships_in_product=None,
        concepts_introduced=[],
        relation_to_prior=[],
        confidence=0.9,
        evidence_quotes=[quote] if quote else [],
    )


def make_draft(claim: Claim) -> DraftSynthesis:
    return DraftSynthesis(
        synthesis_type="digest",
        title="Test Digest",
        sections=[
            {
                "heading": "Test Section",
                "claims": [claim],
                "prose": claim.text,
            }
        ],
        posts_covered=claim.supporting_post_ids,
    )


def _mock_llm_response(critic: Critic, *payloads: dict) -> MagicMock:
    """Patch critic._get_client so each generate_content call returns the
    next payload in *payloads*, in order."""

    responses = []
    for payload in payloads:
        resp = MagicMock()
        resp.text = json.dumps(payload)
        responses.append(resp)

    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = responses
    return patch.object(critic, "_get_client", return_value=mock_client)


# ---------------------------------------------------------------------------
# Layer 2 — semantic verification (LLM), mocked
# ---------------------------------------------------------------------------


def test_supported_claim():
    critic = make_critic()
    claim = Claim(
        text="Anthropic released Claude 3.5 Sonnet.",
        supporting_post_ids=["p1"],
        supporting_quotes=["Anthropic announced Claude 3.5 Sonnet."],
    )
    with _mock_llm_response(
        critic, {"verdict": "supported", "reasoning": "Direct match.", "corrected_text": None}
    ):
        report = critic.review(make_draft(claim), [make_post("p1", claim.supporting_quotes[0])])

    assert report.verdicts[0].verdict == "supported"
    assert report.revision_needed is False


def test_high_lexical_overlap_but_not_semantically_supported():
    """The old rule-based Critic would have called this "supported" on
    word-overlap alone (GPT-5/outperforms/model/benchmark all shared).
    The hybrid Critic must defer to semantic verification instead of
    declaring "supported" just because the rules didn't object."""

    critic = make_critic()
    claim = Claim(
        text="GPT-5 outperforms every other model on all benchmarks.",
        supporting_post_ids=["p1"],
        supporting_quotes=["GPT-5 outperforms every other model on the MMLU benchmark."],
    )
    with _mock_llm_response(
        critic,
        {
            "verdict": "partial",
            "reasoning": "Evidence only covers one benchmark, claim generalizes to all.",
            "corrected_text": "GPT-5 outperforms every other model on MMLU.",
        },
    ):
        report = critic.review(make_draft(claim), [make_post("p1", claim.supporting_quotes[0])])

    assert report.verdicts[0].verdict == "partial"
    assert report.verdicts[0].verdict != "supported"


def test_evidence_supports_only_part_of_claim():
    critic = make_critic()
    claim = Claim(
        text="DeepMind is leading embodied intelligence research.",
        supporting_post_ids=["p1"],
        supporting_quotes=["DeepMind released a new robotics benchmark."],
    )
    with _mock_llm_response(
        critic,
        {
            "verdict": "partial",
            "reasoning": "Evidence shows one benchmark release, not overall research leadership.",
            "corrected_text": "DeepMind released a new robotics benchmark.",
        },
    ):
        report = critic.review(make_draft(claim), [make_post("p1", claim.supporting_quotes[0])])

    assert report.verdicts[0].verdict == "partial"


def test_unsupported_claim():
    critic = make_critic()
    claim = Claim(
        text="OpenAI open-sourced the model.",
        supporting_post_ids=["p1"],
        supporting_quotes=["OpenAI announced API access to the model."],
    )
    with _mock_llm_response(
        critic,
        {
            "verdict": "unsupported",
            "reasoning": "API access is not the same as open-sourcing.",
            "corrected_text": "Remove this claim.",
        },
    ):
        report = critic.review(make_draft(claim), [make_post("p1", claim.supporting_quotes[0])])

    assert report.verdicts[0].verdict == "unsupported"


def test_semantic_contradiction_not_caught_by_keyword_rules():
    """A contradiction the narrow rule-based pre-check doesn't recognize
    (no "open source" / "not released" phrasing) must still be caught —
    by semantic verification, not by the rules."""

    critic = make_critic()
    claim = Claim(
        text="The method scales linearly with input size.",
        supporting_post_ids=["p1"],
        supporting_quotes=["Experiments show runtime growing quadratically with input size."],
    )
    with _mock_llm_response(
        critic,
        {
            "verdict": "contradicted",
            "reasoning": "Evidence states quadratic scaling, the opposite of linear.",
            "corrected_text": "The method scales quadratically with input size.",
        },
    ):
        report = critic.review(make_draft(claim), [make_post("p1", claim.supporting_quotes[0])])

    assert report.verdicts[0].verdict == "contradicted"
    assert report.revision_needed is True


# ---------------------------------------------------------------------------
# Layer 1 — deterministic pre-checks (short-circuit, LLM never called)
# ---------------------------------------------------------------------------


def test_contradicted_claim_short_circuits_without_llm_call():
    """Matches the original rule-based contradiction pattern — must not
    spend an LLM call to detect it."""

    critic = make_critic()
    claim = Claim(
        text="The model is open source.",
        supporting_post_ids=["p1"],
        supporting_quotes=["The model weights are not publicly released."],
    )
    with patch.object(critic, "_get_client") as mock_get_client:
        report = critic.review(make_draft(claim), [make_post("p1", claim.supporting_quotes[0])])
        mock_get_client.assert_not_called()

    assert report.verdicts[0].verdict == "contradicted"
    assert report.revision_needed is True


def test_missing_citation_short_circuits_without_llm_call():
    critic = make_critic()
    claim = Claim(text="Some unverified claim.", supporting_post_ids=[], supporting_quotes=[])
    with patch.object(critic, "_get_client") as mock_get_client:
        report = critic.review(make_draft(claim), [])
        mock_get_client.assert_not_called()

    assert report.verdicts[0].verdict == "unsupported"
    assert "no citation" in report.verdicts[0].reasoning.lower()


def test_nonexistent_post_id_short_circuits_without_llm_call():
    """Citing a post_id that was never actually analyzed this run must
    be caught deterministically, not passed to the LLM as if it had
    real evidence."""

    critic = make_critic()
    claim = Claim(
        text="OpenAI published a new safety framework.",
        supporting_post_ids=["post-does-not-exist"],
        supporting_quotes=[],
    )
    with patch.object(critic, "_get_client") as mock_get_client:
        # post_lookup is built from [] — "post-does-not-exist" resolves to nothing.
        report = critic.review(make_draft(claim), [])
        mock_get_client.assert_not_called()

    assert report.verdicts[0].verdict == "unsupported"
    assert "not found" in report.verdicts[0].reasoning.lower()


def test_empty_evidence_short_circuits_without_llm_call():
    """A citation that resolves to a real post, but with no actual quote
    text, must be caught deterministically."""

    critic = make_critic()
    claim = Claim(text="Some claim.", supporting_post_ids=["p1"], supporting_quotes=[])
    empty_post = make_post("p1", quote="")  # evidence_quotes=[]

    with patch.object(critic, "_get_client") as mock_get_client:
        report = critic.review(make_draft(claim), [empty_post])
        mock_get_client.assert_not_called()

    assert report.verdicts[0].verdict == "unsupported"
    assert "no evidence text" in report.verdicts[0].reasoning.lower()


def test_empty_claim_text_short_circuits_without_llm_call():
    critic = make_critic()
    claim = Claim(text="", supporting_post_ids=["p1"], supporting_quotes=["some evidence"])
    with patch.object(critic, "_get_client") as mock_get_client:
        report = critic.review(make_draft(claim), [make_post("p1", "some evidence")])
        mock_get_client.assert_not_called()

    assert report.verdicts[0].verdict == "unsupported"


# ---------------------------------------------------------------------------
# Report-level aggregation (revision_needed / num_unsupported)
# ---------------------------------------------------------------------------


def test_single_unsupported_claim_triggers_revision_at_default_threshold():
    """Test B1: settings.critic_unsupported_threshold defaults to 1 —
    a single unsupported claim must be enough to require revision, not
    just "more than one". A final answer must not carry even one
    unverified claim by default."""

    critic = make_critic()
    assert critic.settings.critic_unsupported_threshold == 1
    claim = Claim(
        text="OpenAI open-sourced the model.",
        supporting_post_ids=["p1"],
        supporting_quotes=["OpenAI announced API access to the model."],
    )
    with _mock_llm_response(
        critic,
        {"verdict": "unsupported", "reasoning": "API access != open source.", "corrected_text": None},
    ):
        report = critic.review(make_draft(claim), [make_post("p1", claim.supporting_quotes[0])])

    assert report.num_unsupported == 1
    assert report.revision_needed is True


def test_more_than_one_unsupported_triggers_revision():
    critic = make_critic()
    claim1 = Claim(
        text="OpenAI open-sourced the model.",
        supporting_post_ids=["p1"],
        supporting_quotes=["OpenAI announced API access to the model."],
    )
    claim2 = Claim(
        text="Anthropic released all model weights.",
        supporting_post_ids=["p2"],
        supporting_quotes=["Anthropic described model capabilities in a blog post."],
    )
    draft = DraftSynthesis(
        synthesis_type="digest",
        title="Test Digest",
        sections=[
            {
                "heading": "Test Section",
                "claims": [claim1, claim2],
                "prose": "Test prose.",
            }
        ],
        posts_covered=["p1", "p2"],
    )

    with _mock_llm_response(
        critic,
        {"verdict": "unsupported", "reasoning": "API access != open source.", "corrected_text": None},
        {"verdict": "unsupported", "reasoning": "Blog post doesn't confirm all weights released.", "corrected_text": None},
    ):
        report = critic.review(
            draft,
            [
                make_post("p1", claim1.supporting_quotes[0]),
                make_post("p2", claim2.supporting_quotes[0]),
            ],
        )

    assert report.num_unsupported == 2
    assert report.revision_needed is True


# ---------------------------------------------------------------------------
# P1-A: Gemini transient-error retry/backoff, and verification_unavailable
# ---------------------------------------------------------------------------


def test_semantic_verify_retries_transient_error_then_succeeds():
    """Test D1: first attempt 503, second attempt succeeds — retry
    happened, and the final verdict is the real semantic-verification
    result, not a degraded fallback."""

    critic = make_critic()
    claim = Claim(
        text="Anthropic released Claude 3.5 Sonnet.",
        supporting_post_ids=["p1"],
        supporting_quotes=["Anthropic announced Claude 3.5 Sonnet."],
    )
    success = MagicMock()
    success.text = json.dumps(
        {"verdict": "supported", "reasoning": "Direct match.", "corrected_text": None}
    )
    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = [_server_error(503), success]

    with patch.object(critic, "_get_client", return_value=mock_client), patch(
        "agent_system.llm_retry.time.sleep"
    ) as mock_sleep:
        report = critic.review(
            make_draft(claim), [make_post("p1", claim.supporting_quotes[0])]
        )

    assert report.verdicts[0].verdict == "supported"
    assert mock_client.models.generate_content.call_count == 2
    mock_sleep.assert_called_once()


def test_semantic_verify_repeated_transient_failure_is_verification_unavailable():
    """Test D2: every attempt hits a transient error (503 repeated) —
    the verdict must be "verification_unavailable", explicitly NOT
    "unsupported". A service outage is not evidence the claim is
    false, and this verdict must not silently get treated as one."""

    critic = make_critic()
    claim = Claim(
        text="Anthropic released Claude 3.5 Sonnet.",
        supporting_post_ids=["p1"],
        supporting_quotes=["Anthropic announced Claude 3.5 Sonnet."],
    )
    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = [
        _server_error(503),
        _server_error(503),
        _server_error(503),
    ]

    with patch.object(critic, "_get_client", return_value=mock_client), patch(
        "agent_system.llm_retry.time.sleep"
    ):
        report = critic.review(
            make_draft(claim), [make_post("p1", claim.supporting_quotes[0])]
        )

    verdict = report.verdicts[0].verdict
    assert verdict == "verification_unavailable"
    assert verdict != "unsupported"
    assert mock_client.models.generate_content.call_count == 3  # _MAX_LLM_ATTEMPTS

    # Must not be silently treated as disproven: doesn't count toward
    # num_unsupported, and alone doesn't force a revision.
    assert report.num_unsupported == 0
    assert report.revision_needed is False


def test_semantic_verify_non_retryable_error_is_not_retried():
    """Test D3: a non-retryable error (e.g. a malformed request, 400)
    must not spend 2 more wasted round-trips retrying a guaranteed
    repeat failure — one attempt, straight to verification_unavailable."""

    critic = make_critic()
    claim = Claim(
        text="Anthropic released Claude 3.5 Sonnet.",
        supporting_post_ids=["p1"],
        supporting_quotes=["Anthropic announced Claude 3.5 Sonnet."],
    )
    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = _client_error(400)

    with patch.object(critic, "_get_client", return_value=mock_client), patch(
        "agent_system.llm_retry.time.sleep"
    ) as mock_sleep:
        report = critic.review(
            make_draft(claim), [make_post("p1", claim.supporting_quotes[0])]
        )

    assert report.verdicts[0].verdict == "verification_unavailable"
    assert mock_client.models.generate_content.call_count == 1
    mock_sleep.assert_not_called()


def test_semantic_verify_unretryable_status_code_not_in_retryable_set():
    """A 4xx that isn't 429 (e.g. 404) must not be retried either —
    only the explicitly-listed transient codes are. This policy now
    lives in the shared agent_system.llm_retry module (P0-1 of the
    retry-unification round), not in critic/agent.py."""

    from agent_system.llm_retry import is_retryable_gemini_error

    assert is_retryable_gemini_error(_client_error(404)) is False
    assert is_retryable_gemini_error(_client_error(429)) is True
    assert is_retryable_gemini_error(_server_error(503)) is True
    assert is_retryable_gemini_error(_server_error(500)) is True
    assert is_retryable_gemini_error(ValueError("not an API error")) is False


# ---------------------------------------------------------------------------
# Evidence / Critic closed loop: authoritative evidence must come from
# real AnalyzedPost fields, never the Synthesizer's own supporting_quotes
# ---------------------------------------------------------------------------


def test_semantic_verify_evidence_excludes_synthesizer_supporting_quotes():
    """Test B3: the claim's own supporting_quotes is a fabricated quote
    the cited AnalyzedPost's real evidence_quotes does not contain — the
    evidence text sent to the LLM for semantic verification must be
    built entirely from the cited AnalyzedPost's own real fields, never
    from claim.supporting_quotes, so a hallucinated quote can never
    reach the LLM disguised as real evidence."""

    critic = make_critic()
    claim = Claim(
        text="Anthropic released a new 100-trillion-parameter model.",
        supporting_post_ids=["p1"],
        # Hallucinated: this text does not appear anywhere in the real
        # post's own evidence_quotes below.
        supporting_quotes=["We announce our new 100-trillion-parameter model."],
    )
    real_post = make_post("p1", "Anthropic published research on interpretability.")

    resp = MagicMock()
    resp.text = json.dumps({"verdict": "unsupported", "reasoning": "no match", "corrected_text": None})
    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = resp

    with patch.object(critic, "_get_client", return_value=mock_client), patch(
        "agent_system.critic.agent.load_prompt", return_value="dummy prompt"
    ) as mock_load_prompt:
        critic.review(make_draft(claim), [real_post])

    evidence_sent = mock_load_prompt.call_args.kwargs["evidence"]
    assert "100-trillion-parameter" not in evidence_sent
    assert "interpretability" in evidence_sent


def test_collect_evidence_never_includes_claim_supporting_quotes():
    """Same principle as above, at the unit level: _collect_evidence's
    output must never contain text that only exists in
    claim.supporting_quotes."""

    critic = make_critic()
    claim = Claim(
        text="x",
        supporting_post_ids=["p1"],
        supporting_quotes=["ONLY_IN_SYNTHESIZER_QUOTE"],
    )
    post = make_post("p1", "ONLY_IN_ANALYZED_POST")
    evidence = critic._collect_evidence(claim, {"p1": post})

    assert "ONLY_IN_SYNTHESIZER_QUOTE" not in evidence
    assert "ONLY_IN_ANALYZED_POST" in evidence


def test_uncited_claim_with_only_a_synthesizer_quote_is_unsupported():
    """A claim with a supporting_quotes entry but NO supporting_post_ids
    at all must be treated exactly like a claim with no evidence
    whatsoever — a Synthesizer-generated quote is not, by itself,
    independently verifiable evidence."""

    critic = make_critic()
    claim = Claim(
        text="Some claim.", supporting_post_ids=[], supporting_quotes=["a fabricated quote"]
    )
    with patch.object(critic, "_get_client") as mock_get_client:
        report = critic.review(make_draft(claim), [])
        mock_get_client.assert_not_called()

    assert report.verdicts[0].verdict == "unsupported"
    assert "no citation" in report.verdicts[0].reasoning.lower()


# ---------------------------------------------------------------------------
# Evidence entailment, not world-knowledge judgment
# ---------------------------------------------------------------------------


def test_prompt_instructs_against_stale_world_knowledge():
    """Test B4 (prompt-content half — the only part a unit test can
    assert deterministically; whether a real Gemini call actually
    honors this is a live-run concern): the Critic prompt must
    explicitly instruct the model not to reject a claim just because it
    conflicts with the model's own prior/training knowledge."""

    from agent_system.prompts import load_prompt

    rendered = load_prompt("critic", claim_text="x", evidence="y").lower()
    assert "world-knowledge" in rendered or "world knowledge" in rendered
    assert "do not reject" in rendered


def test_supported_claim_with_current_evidence_not_overridden():
    """Test B4 (behavioral half): given evidence that clearly supports a
    claim about something recent, and a (mocked) LLM verdict of
    "supported", the Critic must not have any deterministic override
    that downgrades it just because the claim describes something
    recent/unfamiliar-sounding — nothing in the pipeline between the
    LLM's verdict and the final ClaimVerdict second-guesses "supported"."""

    critic = make_critic()
    claim = Claim(
        text="Anthropic released Claude Opus 5 in 2026.",
        supporting_post_ids=["p1"],
        supporting_quotes=[],
    )
    real_post = make_post("p1", "Anthropic officially released Claude Opus 5 today, 2026.")
    with _mock_llm_response(
        critic,
        {
            "verdict": "supported",
            "reasoning": "Evidence directly confirms the release.",
            "corrected_text": None,
        },
    ):
        report = critic.review(make_draft(claim), [real_post])

    assert report.verdicts[0].verdict == "supported"
    assert report.revision_needed is False


# ---------------------------------------------------------------------------
# source_trust — unchanged, unrelated to this task
# ---------------------------------------------------------------------------


from agent_system.critic.source_trust import score_source


def test_official_lab_source_high_trust():
    result = score_source(
        source="Anthropic",
        url="https://www.anthropic.com/news/example",
        published_at="2026-05-01",
        content_type="blog",
    )

    assert result.score == 5
    assert "official" in result.reason.lower()


def test_arxiv_source_mid_high_trust():
    result = score_source(
        source="arXiv",
        url="https://arxiv.org/abs/1234.5678",
        published_at="2026-05-01",
        content_type="paper",
    )

    assert result.score >= 4


def test_missing_metadata_lowers_trust():
    result = score_source(
        source="Unknown newsletter",
        url="",
        published_at="",
        content_type="blog",
    )

    assert result.score <= 2
