"""Output guardrails for final synthesized answers.

``check_claim_citations`` (and the lower-level ``find_missing_citations``
it's built on) is the *structural* fail-safe: does every claim carry a
citation at all? Whether a citation actually, semantically *supports*
its claim is the Critic's job (``agent_system.critic.agent``), not
this one — the two checks run at different pipeline stages for
different purposes and are deliberately not merged.

Severity, not all-or-nothing: a lone claim missing a citation is a
repair/downgrade target, not a reason to fail the whole output. Only a
*systemic* gap — most of the draft's claims uncited, i.e. the core
answer itself lacking evidence rather than one supporting detail —
counts as unsafe. See ``Orchestrator._guard_output`` for how a
systemic gap gets one repair attempt before it actually blocks
anything.

Only ``claims`` are ever checked — a section's ``prose`` (the
executive-summary / transitional narrative connecting the claims) is
never required to carry a citation. That split already exists in the
schema (``DraftSynthesis.sections[i]`` has both ``prose`` and
``claims``); this module just respects it rather than re-deriving
"is this sentence a claim or a summary" itself.
"""

from __future__ import annotations

from agent_system.guardrails.pii import redact_pii
from agent_system.guardrails.result import GuardrailResult
from agent_system.schemas import Claim, DraftSynthesis

# A single unsupported claim in an otherwise well-cited draft is a
# repair/drop target, not a reason to fail the whole output. Only treat
# the gap as "systemic" (block-worthy) once it's the *majority* of the
# draft's claims — the core answer itself, not one supporting detail.
SYSTEMIC_GAP_RATIO = 0.5


def _to_claim(raw: Claim | dict) -> Claim:
    return raw if isinstance(raw, Claim) else Claim.from_dict(raw)


def _iter_claims(draft: DraftSynthesis):
    """Yield (section_index, claim_index, Claim) for every claim in the
    draft. Sections with no ``claims`` (prose-only) contribute nothing —
    prose is never citation-checked."""

    for section_index, section in enumerate(draft.sections):
        for claim_index, raw_claim in enumerate(section.get("claims", []) or []):
            yield section_index, claim_index, _to_claim(raw_claim)


def find_missing_citations(draft: DraftSynthesis) -> list[tuple[int, int]]:
    """Return (section_index, claim_index) for every claim with neither a
    supporting_post_id nor a supporting_quote."""

    return [
        (section_index, claim_index)
        for section_index, claim_index, claim in _iter_claims(draft)
        if not claim.supporting_post_ids and not claim.supporting_quotes
    ]


def strip_claims(draft: DraftSynthesis, targets: list[tuple[int, int]]) -> DraftSynthesis:
    """Return a new DraftSynthesis with the given (section_index,
    claim_index) claims removed.

    The deterministic fallback when a repair pass doesn't fully resolve
    a citation gap but the gap isn't systemic enough to block the
    output: downgrade the specific claim out of the draft rather than
    fail the whole thing. Sections are kept even if this empties their
    claim list (their prose may still stand on its own); posts_covered
    is left as-is since removing a claim doesn't mean the post it cited
    stopped being relevant to the rest of the draft.
    """

    if not targets:
        return draft

    drop_by_section: dict[int, set[int]] = {}
    for section_index, claim_index in targets:
        drop_by_section.setdefault(section_index, set()).add(claim_index)

    new_sections: list[dict] = []
    for section_index, section in enumerate(draft.sections):
        drop = drop_by_section.get(section_index)
        if not drop:
            new_sections.append(section)
            continue
        kept = [c for i, c in enumerate(section.get("claims", []) or []) if i not in drop]
        new_section = dict(section)
        new_section["claims"] = kept
        new_sections.append(new_section)

    return DraftSynthesis(
        synthesis_type=draft.synthesis_type,
        title=draft.title,
        sections=new_sections,
        posts_covered=draft.posts_covered,
        generated_at=draft.generated_at,
    )


def check_claim_citations(draft: DraftSynthesis) -> GuardrailResult:
    """Quick structural snapshot: is this draft's citation coverage OK
    right now?

    A minority of claims missing a citation is reported as *safe*
    (``reason`` still names them, for a caller that wants to act on it)
    — this only returns ``safe=False`` when the gap is systemic (see
    ``SYSTEMIC_GAP_RATIO``): most of the draft's claims are uncited,
    i.e. the core answer lacks evidence, not one supporting detail.
    """

    total = sum(1 for _ in _iter_claims(draft))
    missing = find_missing_citations(draft)

    if not missing:
        return GuardrailResult(
            safe=True, reason="All claims include supporting evidence.", sanitized_text=None
        )

    detail = "; ".join(
        f"section={s}, claim={c}: {_to_claim(draft.sections[s]['claims'][c]).text}"
        for s, c in missing
    )
    ratio = len(missing) / total if total else 1.0

    if ratio > SYSTEMIC_GAP_RATIO:
        return GuardrailResult(
            safe=False,
            reason=(
                f"Systemic citation gap: {len(missing)}/{total} claims missing "
                f"citation/evidence — {detail}"
            ),
            sanitized_text=None,
        )

    return GuardrailResult(
        safe=True,
        reason=(
            f"{len(missing)}/{total} claim(s) missing citation/evidence "
            f"(non-systemic, not blocking): {detail}"
        ),
        sanitized_text=None,
    )


def check_output_pii(text: str) -> GuardrailResult:
    """Redact PII from final output before display or logging."""
    redacted = redact_pii(text)

    if redacted != text:
        return GuardrailResult(
            safe=True,
            reason="PII redacted from output.",
            sanitized_text=redacted,
        )

    return GuardrailResult(
        safe=True,
        reason="No PII detected in output.",
        sanitized_text=text,
    )
