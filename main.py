"""Entry point for the frontier literature agent pipeline.

Usage:
    python main.py                          # default goal
    python main.py "summarize latest RLHF"  # custom goal
"""

import json

import sys
from typing import Any

from agent_system.config import get_settings
from agent_system.orchestrator.agent import Orchestrator
from agent_system.schemas import UserProfile


def _cite_tag(post_ids: list[str], citations: dict[str, Any]) -> str:
    """Render a claim's citation tag, resolving each post_id to a
    readable "Title — Source" when the data is available (see
    Orchestrator._attach_citations) and falling back to the bare
    post_id otherwise — e.g. for a last_run.json written before
    citations existed."""

    if not post_ids:
        return ""
    parts = []
    for pid in post_ids:
        meta = citations.get(pid) or {}
        title = meta.get("title")
        org = meta.get("organization") or meta.get("source")
        if title and org:
            parts.append(f"{title} — {org} [{pid}]")
        elif title:
            parts.append(f"{title} [{pid}]")
        else:
            parts.append(f"[{pid}]")
    return " <sub>" + ", ".join(parts) + "</sub>"


def render_markdown(result: dict[str, Any]) -> str:
    """Render a VerifiedSynthesis dict as a human-readable Markdown report."""
    draft = result.get("draft", {})
    report = result.get("critic_report", {})
    citations = result.get("citations", {}) or {}

    lines: list[str] = []
    title = draft.get("title") or "Untitled Synthesis"
    lines.append(f"# {title}")
    lines.append("")

    meta = [
        f"**Type:** `{draft.get('synthesis_type', 'unknown')}`",
        f"**Generated:** {draft.get('generated_at', 'n/a')}",
        f"**Posts covered:** {len(draft.get('posts_covered', []))}",
        f"**Revisions:** {result.get('revision_count', 0)}",
        f"**Final:** {'yes' if result.get('final') else 'no'}",
    ]
    lines.append(" · ".join(meta))
    lines.append("")

    posts_covered = draft.get("posts_covered", [])
    if posts_covered:
        ids = ", ".join(f"`{p}`" for p in posts_covered)
        lines.append(f"<sub>Sources: {ids}</sub>")
        lines.append("")

    for sec in draft.get("sections", []):
        if not isinstance(sec, dict):
            continue
        heading = sec.get("heading", "").strip()
        prose = sec.get("prose", "").strip()
        claims = sec.get("claims", []) or []

        if heading:
            lines.append(f"## {heading}")
            lines.append("")
        if prose:
            lines.append(prose)
            lines.append("")

        for claim in claims:
            text = claim.get("text", "").strip() if isinstance(claim, dict) else ""
            post_ids = claim.get("supporting_post_ids", []) if isinstance(claim, dict) else []
            quotes = claim.get("supporting_quotes", []) if isinstance(claim, dict) else []
            if not text:
                continue
            cite = _cite_tag(post_ids, citations)
            lines.append(f"- {text}{cite}")
            if quotes:
                lines.append("")
                lines.append("  <details><summary>Evidence</summary>")
                lines.append("")
                for q in quotes:
                    q_clean = q.strip().replace("\n", " ")
                    lines.append(f"  > {q_clean}")
                    lines.append("")
                lines.append("  </details>")
            lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("## Critic Report")
    lines.append("")
    lines.append(
        f"- **Unsupported claims:** {report.get('num_unsupported', 0)}  "
        f"\n- **Revision needed:** {'yes' if report.get('revision_needed') else 'no'}"
    )
    notes = (report.get("revision_notes") or "").strip()
    if notes:
        lines.append("")
        lines.append("**Revision notes:**")
        lines.append("")
        lines.append(f"> {notes}")

    verdicts = report.get("verdicts", []) or []
    flagged = [v for v in verdicts if v.get("verdict") != "supported"]
    if flagged:
        lines.append("")
        lines.append("### Flagged claims")
        lines.append("")
        for v in flagged:
            claim = v.get("claim", {}) or {}
            ctext = (claim.get("text") if isinstance(claim, dict) else "") or ""
            verdict = v.get("verdict", "?")
            reasoning = (v.get("reasoning") or "").strip()
            corrected = (v.get("corrected_text") or "").strip()
            lines.append(f"- **[{verdict}]** {ctext}")
            if reasoning:
                lines.append(f"  - *Reasoning:* {reasoning}")
            if corrected:
                lines.append(f"  - *Corrected:* {corrected}")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def build_default_profile() -> UserProfile:
    return UserProfile(
        user_id="demo_user",
        interests=["large language models", "RLHF", "AI safety"],
        role_target="researcher",
        seniority="PhD student",
        reading_history=[],
    )


def main() -> None:
    goal = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "What are the latest developments in AI agent systems from major labs?"
    )

    print(f"Goal: {goal}")
    print("=" * 60)

    settings = get_settings()
    print(f"Model: {settings.model_flash} / {settings.model_pro}")
    print(f"API key: ...{settings.google_api_key[-4:]}")
    print()

    profile = build_default_profile()
    orchestrator = Orchestrator(settings)

    print("Running pipeline...")
    print()

    try:
        result = orchestrator.run(goal, profile)
    except Exception as exc:
        print(f"Pipeline failed: {exc}")
        sys.exit(1)

    # --- output ---
    draft = result.draft
    print(f"[{draft.synthesis_type.upper()}] {draft.title}")
    print(f"Revision count: {result.revision_count}  |  Final: {result.final}")
    print("-" * 60)

    for sec in draft.sections:
        if not isinstance(sec, dict):
            continue
        heading = sec.get("heading", "")
        prose = sec.get("prose", "")
        claims = sec.get("claims", [])
        if heading:
            print(f"\n## {heading}")
        if prose:
            print(prose)
        for c in claims:
            text = c.text if hasattr(c, "text") else c.get("text", "")
            quotes = (
                c.supporting_quotes
                if hasattr(c, "supporting_quotes")
                else c.get("supporting_quotes", [])
            )
            if text:
                print(f"\n  Claim: {text}")
            if quotes:
                print(f"  Evidence: {quotes[0][:120]}...")

    print("\n" + "-" * 60)
    report = result.critic_report
    print(f"Critic: {report.num_unsupported} unsupported claim(s)")
    print(f"Persisted to SQLite: {settings.sqlite_path}")

    # dump full JSON + human-readable Markdown
    settings.ensure_dirs()
    result_dict = result.to_dict()
    json_path = settings.data_dir / "last_run.json"
    md_path = settings.data_dir / "last_run.md"
    json_path.write_text(json.dumps(result_dict, indent=2, ensure_ascii=False))
    md_path.write_text(render_markdown(result_dict))
    print(f"Full output: {json_path}")
    print(f"Readable report: {md_path}")


if __name__ == "__main__":
    main()