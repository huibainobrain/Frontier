"""Synthesizer agent — multi-article synthesis with Gemini.

Takes a list of :class:`AnalyzedPost` records plus a :class:`UserProfile`
and produces a :class:`DraftSynthesis` in one of four intents:
digest, tracker, comparison, or reading_plan.

Each intent has its own prompt template (``prompts/synthesizer_<type>.md``).
The ``revise`` method accepts Critic notes and re-generates the draft.

The Synthesizer is the node that produces the user-facing answer, so a
Gemini failure here must surface as an explicit task failure, never a
fabricated low-quality answer (e.g. concatenating key_claims together).
``synthesize()`` retries transient infra errors via the shared
``agent_system.llm_retry`` policy; if every attempt fails, it raises
:class:`SynthesisUnavailableError` rather than returning anything —
callers should not catch this to paper over it with a guessed answer.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from typing import Any

from agent_system.config import Settings, get_settings
from agent_system.llm_retry import call_with_retry, note_usage_from_gemini_response
from agent_system.prompts import load_prompt
from agent_system.schemas import (
    AnalyzedPost,
    Claim,
    DraftSynthesis,
    UserProfile,
    now_iso,
)


class SynthesisUnavailableError(Exception):
    """Raised when the Synthesizer's core answer-generation Gemini call
    fails even after retrying transient errors. Deliberately not caught
    anywhere and turned into a degraded answer — see module docstring."""

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(f"Synthesizer unavailable: {reason}")


def _format_posts(posts: list[AnalyzedPost]) -> str:
    """Serialize analyzed posts into a prompt-friendly text block.

    Includes each post's real provenance (title / organization or
    source / published date / URL) alongside the analysis, so the four
    synthesis templates can cite a real title and date instead of
    having the LLM invent one — Comparison, Tracker, and Reading Plan
    all depend on this being present here, since this is the only text
    block all four templates interpolate.
    """

    parts: list[str] = []
    for p in posts:
        org = p.organization or p.source or "unknown source"
        date = p.published_at or "date unknown"
        title = p.title or "(untitled)"
        parts.append(
            f'- [{p.post_id}] "{title}" — {org} ({date})\n'
            f"  URL: {p.url}\n"
            f"  Category: {p.category}\n"
            f"  Key claim: {p.key_claim}\n"
            f"  Takeaway: {p.practitioner_takeaway}\n"
            f"  Concepts: {', '.join(p.concepts_introduced)}\n"
            f"  Evidence: {'; '.join(p.evidence_quotes[:2])}"
        )
    return "\n".join(parts)


def _format_gap_notice(evidence_gaps: list[dict]) -> str:
    """Render unresolved research-loop evidence gaps as an explicit
    instruction block, so the synthesis honestly names a limitation
    instead of quietly writing around it. Only called when the
    research loop stopped (hard round cap) without the evaluator
    confirming sufficiency — see Orchestrator._run_research_loop."""

    if not evidence_gaps:
        return ""
    lines = [
        "\nKnown evidence gaps (from the research process — the search "
        "budget was exhausted before these were resolved). Be honest "
        "about them: explicitly note the limitation in prose for the "
        "affected section(s) rather than fabricating a claim to fill "
        "the gap, and never cite a post_id that doesn't actually "
        "support the missing point.",
    ]
    for g in evidence_gaps:
        target = g.get("target", "unknown") if isinstance(g, dict) else "unknown"
        gap = g.get("gap", "") if isinstance(g, dict) else ""
        lines.append(f"- {target}: {gap}")
    return "\n".join(lines)


def _format_profile(profile: UserProfile) -> str:
    return (
        f"Role: {profile.role_target}, Seniority: {profile.seniority}, "
        f"Interests: {', '.join(profile.interests)}"
    )


class Synthesizer:
    """Gemini-backed Synthesizer that produces multi-article summaries."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings if settings is not None else get_settings()
        self._client: Any = None

    # ------------------------------------------------------------------
    # Lazy Gemini client
    # ------------------------------------------------------------------

    def _get_client(self) -> Any:
        if self._client is None:
            from google.genai import Client

            self._client = Client()
        return self._client

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def synthesize(
        self,
        posts: list[AnalyzedPost],
        profile: UserProfile,
        synthesis_type: str,
        *,
        research_targets: list[str] | None = None,
        evidence_gaps: list[dict] | None = None,
    ) -> DraftSynthesis:
        """Produce a draft synthesis from analyzed articles.

        Args:
            research_targets: the real-world entities the plan says the
                goal is actually about (e.g. ``["OpenAI", "Anthropic"]``
                — set once by the Planner from the goal text, see
                ``schemas.ExecutionPlan.research_targets``). Drives the
                Comparison template's "labs". Deliberately NOT a list of
                source_ids or of sources a research round happened to
                query — a round may query extra sources purely for
                supporting evidence without those becoming comparison
                targets. Falls back to the organizations/sources present
                on *posts* only when no targets were resolved at all
                (e.g. a non-comparison synthesis type, or direct
                Synthesizer use in a test).
            evidence_gaps: unresolved gaps from the research loop's
                evaluator, if the loop stopped without confirming
                sufficiency (hard round cap). Injected into the prompt
                so the synthesis names the limitation instead of
                silently writing around it.
        """

        valid_types = {"digest", "tracker", "comparison", "reading_plan"}
        if synthesis_type not in valid_types:
            synthesis_type = "digest"

        posts_text = _format_posts(posts)
        gap_notice = _format_gap_notice(evidence_gaps or [])
        if gap_notice:
            posts_text = f"{posts_text}\n{gap_notice}"
        profile_text = _format_profile(profile)

        now = datetime.now(UTC)
        window_days = 7
        window_start = now - timedelta(days=window_days)
        date_range = f"{window_start.date().isoformat()} to {now.date().isoformat()}"

        template_vars: dict[str, str] = {
            "analyzed_posts": posts_text,
            "user_profile": profile_text,
            "date_range": date_range,
            "concept": profile.interests[0] if profile.interests else "AI",
            "labs": self._labs_for_comparison(posts, research_targets, synthesis_type),
            "timeline_window": date_range,
            "learning_goal": ", ".join(profile.interests),
            "already_read_ids": ", ".join(profile.reading_history[:10]),
        }

        prompt = load_prompt(f"synthesizer_{synthesis_type}", **template_vars)
        try:
            result = self._call_gemini(prompt)
        except Exception as exc:
            raise SynthesisUnavailableError(str(exc)) from exc

        sections: list[dict] = []
        for sec in result.get("sections", []):
            claims = [
                Claim(
                    text=c.get("text", ""),
                    supporting_post_ids=c.get("supporting_post_ids", []),
                    supporting_quotes=c.get("supporting_quotes", []),
                )
                for c in sec.get("claims", [])
            ]
            sections.append(
                {
                    "heading": sec.get("heading", ""),
                    "claims": claims,
                    "prose": sec.get("prose", ""),
                }
            )

        post_ids = [p.post_id for p in posts]
        return DraftSynthesis(
            synthesis_type=synthesis_type,
            title=result.get("title", f"Synthesis ({synthesis_type})"),
            sections=sections,
            posts_covered=post_ids,
            generated_at=now_iso(),
        )

    @staticmethod
    def _labs_for_comparison(
        posts: list[AnalyzedPost],
        research_targets: list[str] | None,
        synthesis_type: str = "comparison",
    ) -> str:
        """What to compare, for the Comparison template's ``{labs}`` slot.

        Priority, matching schemas.ExecutionPlan.research_targets:

        1. ``research_targets`` — the entities the goal is actually
           about, resolved once by the Planner (with the Orchestrator's
           target-recovery step as a safety net — see
           ``Orchestrator._recover_research_targets_if_needed``). Used
           verbatim (these are already display names, e.g. "OpenAI" —
           not source_ids needing translation). A round searching extra
           sources for supporting evidence (arXiv, a third lab) must
           never widen this list — that's the whole point of the field.
        2. For ``synthesis_type == "comparison"`` specifically, with no
           research_targets: retrieval evidence must NEVER stand in for
           the user's intent (P0-2) — if the Orchestrator's own recovery
           step already tried and still couldn't determine targets, this
           says so honestly rather than promoting whichever
           organizations happened to be retrieved into de facto
           comparison targets.
        3. Only for every OTHER synthesis type (digest/tracker/
           reading_plan aren't scoped to specific comparison targets in
           the first place): fall back to the organizations/sources
           actually present on the analyzed posts — unaffected by this
           fix, see task principle "非 Comparison 模式不要受到影响".

        Never derived from ``post.category`` (capability/safety/
        engineering/...) — that's a topic axis, not an organization,
        and conflating the two was the original bug here.
        """

        if research_targets:
            labs = {t.strip() for t in research_targets if t and t.strip()}
            return ", ".join(sorted(labs))
        if synthesis_type == "comparison":
            return (
                "the specific organizations requested (unable to confirm "
                "which from the request — do not guess from the sources "
                "retrieved)"
            )
        labs = {p.organization or p.source for p in posts if (p.organization or p.source)}
        return ", ".join(sorted(labs)) if labs else "the sources analyzed"

    def revise(
        self, draft: DraftSynthesis, critic_notes: str
    ) -> DraftSynthesis:
        """Apply Critic's revision notes and re-generate the draft."""

        from agent_system.synthesizer.templates import render_digest

        rendered = render_digest(draft)
        prompt = (
            "You are revising a research synthesis based on reviewer feedback.\n\n"
            f"Current synthesis:\n{rendered}\n\n"
            f"Reviewer feedback:\n{critic_notes}\n\n"
            "Produce a corrected version as JSON. Output ONLY valid JSON:\n"
            '{"title": "<revised title>", '
            '"sections": [{"heading": "<section heading>", '
            '"claims": [{"text": "<revised claim with citation [post-id]>", '
            '"supporting_post_ids": ["<post-id>"], '
            '"supporting_quotes": ["<quote>"]}], '
            '"prose": "<revised prose>"}]}'
        )
        result = self._call_gemini(prompt)

        sections: list[dict] = []
        for sec in result.get("sections", []):
            claims = [
                Claim(
                    text=c.get("text", ""),
                    supporting_post_ids=c.get("supporting_post_ids", []),
                    supporting_quotes=c.get("supporting_quotes", []),
                )
                for c in sec.get("claims", [])
            ]
            sections.append(
                {
                    "heading": sec.get("heading", ""),
                    "claims": claims,
                    "prose": sec.get("prose", ""),
                }
            )

        return DraftSynthesis(
            synthesis_type=draft.synthesis_type,
            title=result.get("title", f"{draft.title} (revised)"),
            sections=sections if sections else draft.sections,
            posts_covered=draft.posts_covered,
            generated_at=now_iso(),
        )

    # ------------------------------------------------------------------
    # Gemini invocation
    # ------------------------------------------------------------------

    def _call_gemini(self, prompt: str) -> dict[str, Any]:
        """Call Gemini Pro with JSON response mode and return parsed dict.

        Retries transient infra errors via the shared
        ``agent_system.llm_retry`` policy. Used by both ``synthesize()``
        (wraps the final exception in ``SynthesisUnavailableError``) and
        ``revise()`` (whose two callers — the Critic-revision loop and
        the output guard's citation repair — already have their own
        established handling for a failed revision, unchanged here).
        """

        from google.genai import types as genai_types

        def _once() -> dict[str, Any]:
            client = self._get_client()
            response = client.models.generate_content(
                model=self.settings.model_pro,
                contents=prompt,
                config=genai_types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.3,
                ),
            )
            note_usage_from_gemini_response(response)
            return json.loads(response.text)

        return call_with_retry("synthesizer", self.settings.model_pro, _once)
