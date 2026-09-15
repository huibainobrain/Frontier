"""Critic agent — hybrid deterministic + LLM semantic verification.

Two layers, in order, per claim:

1. **Deterministic validation** (cheap, no LLM, short-circuits): catches
   structural problems that don't need semantic judgment — an empty
   claim, no citation at all, a citation whose post_id was never
   actually analyzed this run, or a citation that resolves to no
   evidence text. A small set of high-precision textual contradiction
   patterns (kept from the original rule-based Critic) also
   short-circuits straight to "contradicted" — that's a real
   deterministic signal, not a semantic guess, so it's cheap to keep.

2. **Semantic verification** (Gemini, ``prompts/critic.md``): for a
   claim that survives layer 1 — i.e. it has real evidence attached —
   the LLM judges whether that evidence actually, semantically supports
   the claim as worded. This is the layer that replaces the old
   "supported" heuristics (lexical word-overlap, "released"+"announced"
   keyword matching): passing the deterministic checks is never enough
   on its own to call a claim "supported" — only semantic verification
   produces that verdict now.

The ``ClaimVerdict`` / ``CriticReport`` schema and the
revision-needed rule (any "contradicted", or more than one
"unsupported") are unchanged from the original Critic — the
Orchestrator's critic-revision loop doesn't need to know verdicts now
come from two layers instead of one.

The Gemini call backing layer 2 retries a bounded number of times
(``agent_system.llm_retry``, shared with Analyst/Synthesizer/the
Orchestrator's ADK calls) with a short backoff on transient infra
errors (429/5xx) before giving up. If it still can't complete —
transient failure that outlasted the retries, or any non-retryable
error — the claim is marked ``"verification_unavailable"``, never
``"unsupported"``: a Gemini outage is not evidence the claim is false,
and ``CriticReport.num_unsupported`` / ``revision_needed`` deliberately
don't count this verdict (see :meth:`Critic._semantic_verify`).
"""

from __future__ import annotations

import json
import logging
from typing import Any

from agent_system.config import Settings
from agent_system.llm_retry import call_with_retry, note_usage_from_gemini_response
from agent_system.prompts import load_prompt
from agent_system.schemas import (
    AnalyzedPost,
    Claim,
    ClaimVerdict,
    CriticReport,
    DraftSynthesis,
)

logger = logging.getLogger(__name__)

# Verdicts the LLM itself is allowed to return. Deliberately does NOT
# include "verification_unavailable" — that verdict is assigned by
# _semantic_verify's own exception handler when the LLM couldn't be
# reached at all, never something the model claims about itself.
_VALID_VERDICTS = {"supported", "partial", "unsupported", "contradicted"}

# High-precision contradiction pattern, kept from the original
# rule-based Critic as a pre-check — cheap and precise enough to trust
# without an LLM call. Deliberately narrow and one-directional: it only
# ever short-circuits to "contradicted", never to "supported" (that
# verdict now only comes from semantic verification).
_CONTRADICTION_CLAIM_TERMS = ("open source", "open-sourced", "model weights")
_CONTRADICTION_EVIDENCE_TERMS = (
    "not publicly released",
    "not open source",
    "not released",
    "closed source",
)


class Critic:
    """Claim-level critic: deterministic pre-checks first, then LLM
    semantic verification for whatever survives them."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._client: Any = None

    # ------------------------------------------------------------------
    # Lazy Gemini client (same pattern as Analyst / Synthesizer)
    # ------------------------------------------------------------------

    def _get_client(self) -> Any:
        if self._client is None:
            from google.genai import Client

            self._client = Client()
        return self._client

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def review(
        self,
        draft: DraftSynthesis,
        posts: list[AnalyzedPost],
    ) -> CriticReport:
        post_lookup = {p.post_id: p for p in posts}
        verdicts = [
            self._verify_claim(claim, post_lookup) for claim in self._iter_claims(draft)
        ]

        num_unsupported = sum(v.verdict == "unsupported" for v in verdicts)
        has_contradiction = any(v.verdict == "contradicted" for v in verdicts)
        # settings.critic_unsupported_threshold (default 1): by default
        # a single unsupported claim is enough to require revision — a
        # final answer must not carry even one claim the evidence
        # doesn't back up.
        threshold = max(1, int(self.settings.critic_unsupported_threshold))
        revision_needed = has_contradiction or num_unsupported >= threshold

        revision_notes = ""
        if revision_needed:
            revision_notes = (
                "Revision needed: remove or rewrite unsupported/contradicted "
                "claims and keep only claims directly supported by the cited evidence."
            )

        return CriticReport(
            verdicts=verdicts,
            num_unsupported=num_unsupported,
            revision_needed=revision_needed,
            revision_notes=revision_notes,
        )

    # ------------------------------------------------------------------
    # Layer 1 — deterministic validation
    # ------------------------------------------------------------------

    def _verify_claim(
        self, claim: Claim, post_lookup: dict[str, AnalyzedPost]
    ) -> ClaimVerdict:
        deterministic = self._deterministic_check(claim, post_lookup)
        if deterministic is not None:
            return deterministic

        evidence = self._collect_evidence(claim, post_lookup)
        return self._semantic_verify(claim, evidence)

    def _deterministic_check(
        self, claim: Claim, post_lookup: dict[str, AnalyzedPost]
    ) -> ClaimVerdict | None:
        """Structural / precise-pattern checks that don't need semantic
        judgment. Returns a ClaimVerdict to short-circuit (skip the LLM
        call entirely), or ``None`` to fall through to semantic
        verification.

        Citation presence/resolution is judged ONLY on
        ``claim.supporting_post_ids`` resolving to a real, previously
        analyzed post with real evidence_quotes — ``claim.
        supporting_quotes`` (the Synthesizer's own wording, generated at
        synthesis time) is never treated as evidence on its own. A claim
        with a quote but no real post_id is exactly as unsupported as
        one with nothing at all: otherwise a hallucinated claim could
        carry a hallucinated quote that "supports" itself. See
        :meth:`_collect_evidence`.
        """

        text = (claim.text or "").strip()
        if not text:
            return ClaimVerdict(
                claim=claim,
                verdict="unsupported",
                reasoning="Claim has no text.",
                corrected_text="Remove this claim.",
            )

        cited_ids = [pid for pid in claim.supporting_post_ids if pid]
        if not cited_ids:
            return ClaimVerdict(
                claim=claim,
                verdict="unsupported",
                reasoning=(
                    "No citation: the claim cites no post_id from this run's "
                    "analyzed posts (a Synthesizer-provided quote alone is "
                    "not independently verifiable evidence)."
                ),
                corrected_text="Remove this claim unless supporting evidence is retrieved.",
            )

        known_posts = [post_lookup[pid] for pid in cited_ids if pid in post_lookup]
        if not known_posts:
            return ClaimVerdict(
                claim=claim,
                verdict="unsupported",
                reasoning=(
                    "Cited source(s) not found among this run's analyzed posts: "
                    f"{', '.join(cited_ids)}."
                ),
                corrected_text="Remove this claim or cite a real source.",
            )

        if not any(post.evidence_quotes for post in known_posts):
            return ClaimVerdict(
                claim=claim,
                verdict="unsupported",
                reasoning=(
                    "Citation present but no evidence text could be resolved "
                    "from the cited post(s) — they have no extracted "
                    "evidence_quotes to verify against."
                ),
                corrected_text=(
                    "Remove this claim unless additional supporting evidence is retrieved."
                ),
            )

        evidence = self._collect_evidence(claim, post_lookup)
        return self._rule_based_contradiction(claim, evidence)

    def _rule_based_contradiction(
        self, claim: Claim, evidence: str
    ) -> ClaimVerdict | None:
        """High-precision textual contradiction check, kept as a cheap
        pre-check. Deliberately one-directional: this only ever
        short-circuits to "contradicted", never to "supported"."""

        text_l = claim.text.lower()
        ev_l = evidence.lower()
        if any(term in text_l for term in _CONTRADICTION_CLAIM_TERMS) and any(
            term in ev_l for term in _CONTRADICTION_EVIDENCE_TERMS
        ):
            return ClaimVerdict(
                claim=claim,
                verdict="contradicted",
                reasoning=(
                    "The cited evidence explicitly states the opposite of an "
                    "open-source/release claim."
                ),
                corrected_text="Remove this claim or rewrite it to match the cited evidence.",
            )
        return None

    # ------------------------------------------------------------------
    # Layer 2 — semantic verification (Gemini)
    # ------------------------------------------------------------------

    def _semantic_verify(self, claim: Claim, evidence: str) -> ClaimVerdict:
        prompt = load_prompt("critic", claim_text=claim.text, evidence=evidence)
        try:
            result = self._call_gemini(prompt)
        except Exception as exc:
            # NOT "unsupported": that verdict means verification *ran*
            # and the evidence didn't hold up — a semantically different
            # claim from "we couldn't check". Conflating them means a
            # transient Gemini 503 gets reported to the user as "this
            # evidence doesn't support the claim", which is false. The
            # claim already passed layer 1 (it has real evidence
            # attached) — this only marks that the *semantic* check
            # specifically didn't complete, and deliberately doesn't
            # count toward CriticReport.num_unsupported / revision_needed
            # (see review()), so a transient outage doesn't get a
            # correct claim rewritten or dropped either.
            logger.error(
                "Critic semantic verification unavailable: model=%s error_type=%s "
                "final_status=verification_unavailable claim=%r",
                self.settings.model_pro,
                type(exc).__name__,
                claim.text[:80],
            )
            return ClaimVerdict(
                claim=claim,
                verdict="verification_unavailable",
                reasoning=(
                    f"Semantic verification could not be completed ({type(exc).__name__}); "
                    "this reflects a service/infrastructure issue, not a judgment "
                    "that the evidence fails to support the claim."
                ),
                corrected_text=None,
            )

        verdict = str(result.get("verdict", "unsupported")).strip().lower()
        if verdict not in _VALID_VERDICTS:
            logger.warning(
                "Critic LLM returned unknown verdict %r; defaulting to 'unsupported'.",
                verdict,
            )
            verdict = "unsupported"

        reasoning = str(result.get("reasoning", "")).strip() or "Semantic verification."
        corrected = result.get("corrected_text")
        corrected_text = str(corrected).strip() if corrected else None
        if verdict == "supported":
            # A supported claim needs no correction even if the model
            # produced one anyway.
            corrected_text = None

        return ClaimVerdict(
            claim=claim, verdict=verdict, reasoning=reasoning, corrected_text=corrected_text
        )

    def _call_gemini(self, prompt: str) -> dict[str, Any]:
        """Call Gemini with JSON response mode and return the parsed dict.

        Uses ``model_pro``, matching the original design intent
        (meeting/tech_roadmap.md: "Gemini Pro for Synthesizer + Critic —
        accuracy matters"). On a free-tier key with billing off,
        ``model_pro`` currently resolves to the same model as
        ``model_flash`` (see config.py) — a no-op today, but it keeps
        the right cost-tier signal wired for when billing is enabled.

        Retries transient infra errors via the shared
        ``agent_system.llm_retry`` policy. The final exception (if every
        attempt fails, or the first failure isn't retryable) propagates
        to the caller — ``_semantic_verify`` is what turns that into a
        ``"verification_unavailable"`` verdict rather than a crash.
        """

        from google.genai import types as genai_types

        def _once() -> dict[str, Any]:
            client = self._get_client()
            response = client.models.generate_content(
                model=self.settings.model_pro,
                contents=prompt,
                config=genai_types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.1,
                ),
            )
            note_usage_from_gemini_response(response)
            return json.loads(response.text)

        return call_with_retry("critic", self.settings.model_pro, _once)

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _iter_claims(self, draft: DraftSynthesis) -> list[Claim]:
        claims: list[Claim] = []

        for section in draft.sections:
            for raw_claim in section.get("claims", []):
                if isinstance(raw_claim, Claim):
                    claims.append(raw_claim)
                else:
                    claims.append(Claim.from_dict(raw_claim))

        return claims

    def _collect_evidence(self, claim: Claim, post_lookup: dict[str, AnalyzedPost]) -> str:
        """Authoritative evidence text for semantic verification — built
        ONLY from real, persisted ``AnalyzedPost`` fields resolved via
        ``claim.supporting_post_ids`` (source/organization/published
        date/evidence_quotes/key_claim). Deliberately excludes
        ``claim.supporting_quotes``: that field is written by the
        Synthesizer at synthesis time, not independently verifiable, and
        treating it as evidence would let a hallucinated claim "support
        itself" with a hallucinated quote. It may still be useful to a
        human reader for display/citation purposes elsewhere (e.g.
        ``main.py``'s rendering) — just never as fact input here.
        """

        blocks: list[str] = []
        for post_id in claim.supporting_post_ids:
            post = post_lookup.get(post_id)
            if post is None:
                continue
            header = (
                f'[{post.post_id}] "{post.title or "untitled"}" — '
                f"{post.organization or post.source or 'unknown source'} "
                f"({post.published_at or 'date unknown'})"
            )
            lines = [header]
            if post.key_claim:
                lines.append(f"  Analyzed summary: {post.key_claim}")
            quotes = [q for q in post.evidence_quotes if q]
            if quotes:
                lines.append(f"  Evidence quotes: {' | '.join(quotes)}")
            blocks.append("\n".join(lines))

        return "\n\n".join(blocks).strip()
