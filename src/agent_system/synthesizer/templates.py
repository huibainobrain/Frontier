"""Output templates for the four synthesis intents.

Each intent (digest, tracker, comparison, reading_plan) has a markdown
template that the Synthesizer fills in. Templates live here rather than in
``prompts/`` because they are post-LLM formatting, not prompt text.
"""

from __future__ import annotations

from agent_system.schemas import Claim, DraftSynthesis


def _normalize_claims(section: dict) -> list[dict]:
    """Return claims from a section as dicts regardless of storage type."""

    raw = section.get("claims", [])
    out: list[dict] = []
    for c in raw:
        if isinstance(c, Claim):
            out.append(c.to_dict())
        elif isinstance(c, dict):
            out.append(c)
        else:
            out.append({"text": str(c), "supporting_post_ids": [], "supporting_quotes": []})
    return out


# ---------------------------------------------------------------------------
# Digest
# ---------------------------------------------------------------------------


def render_digest(draft: DraftSynthesis) -> str:
    """Render a weekly digest as formatted markdown."""

    parts: list[str] = [f"# {draft.title}\n"]
    for sec in draft.sections:
        if not isinstance(sec, dict):
            continue
        heading = sec.get("heading", "")
        prose = sec.get("prose", "")
        if heading:
            parts.append(f"## {heading}\n")
        if prose:
            parts.append(f"{prose}\n")
        for c in _normalize_claims(sec):
            text = c.get("text", "")
            pids = c.get("supporting_post_ids", [])
            tag = f" [{pids[0]}]" if pids else ""
            parts.append(f"- {text}{tag}")
            for q in c.get("supporting_quotes", [])[:1]:
                parts.append(f"  > {q}")
        parts.append("")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Tracker
# ---------------------------------------------------------------------------


def render_tracker(draft: DraftSynthesis) -> str:
    """Render a concept tracker as a lab-by-lab markdown timeline."""

    parts: list[str] = [f"# {draft.title}\n"]
    for sec in draft.sections:
        if not isinstance(sec, dict):
            continue
        heading = sec.get("heading", "")
        prose = sec.get("prose", "")
        if heading:
            parts.append(f"## {heading}\n")
        if prose:
            parts.append(f"{prose}\n")
        for c in _normalize_claims(sec):
            text = c.get("text", "")
            pids = c.get("supporting_post_ids", [])
            tag = f" [{pids[0]}]" if pids else ""
            parts.append(f"- {text}{tag}")
        parts.append("")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------


def render_comparison(draft: DraftSynthesis) -> str:
    """Render a cross-lab comparison as dimension-segmented markdown."""

    parts: list[str] = [f"# {draft.title}\n"]
    for sec in draft.sections:
        if not isinstance(sec, dict):
            continue
        heading = sec.get("heading", "")
        prose = sec.get("prose", "")
        claims = _normalize_claims(sec)

        if heading:
            parts.append(f"## {heading}\n")

        if claims:
            parts.append("| Aspect | Finding | Source |")
            parts.append("|--------|---------|--------|")
            for c in claims:
                text = c.get("text", "")
                pids = c.get("supporting_post_ids", [])
                tag = pids[0] if pids else "—"
                parts.append(f"| Analysis | {text} | {tag} |")
            parts.append("")

        if prose:
            parts.append(f"{prose}\n")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Reading plan
# ---------------------------------------------------------------------------


def render_reading_plan(draft: DraftSynthesis) -> str:
    """Render a personalized reading plan as an ordered markdown list."""

    parts: list[str] = [f"# {draft.title}\n"]
    item_num = 1
    for sec in draft.sections:
        if not isinstance(sec, dict):
            continue
        heading = sec.get("heading", "")
        prose = sec.get("prose", "")
        if heading:
            parts.append(f"## {heading}\n")
        if prose:
            parts.append(f"{prose}\n")
        for c in _normalize_claims(sec):
            text = c.get("text", "")
            pids = c.get("supporting_post_ids", [])
            tag = f" [{pids[0]}]" if pids else ""
            parts.append(f"{item_num}. {text}{tag}")
            item_num += 1
        parts.append("")
    return "\n".join(parts)