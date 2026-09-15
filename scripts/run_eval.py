import json
from pathlib import Path

from agent_system.config import Settings
from agent_system.critic.agent import Critic
from agent_system.critic.source_trust import score_source
from agent_system.guardrails.output_checker import check_claim_citations
from agent_system.guardrails.pii import redact_pii
from agent_system.guardrails.prompt_injection import check_injection
from agent_system.schemas import AnalyzedPost, Claim, DraftSynthesis


ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = ROOT / "tests" / "eval_set"


def load_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def make_post(post_id: str, quote: str) -> AnalyzedPost:
    return AnalyzedPost(
        post_id=post_id,
        category="capability",
        key_claim=quote,
        practitioner_takeaway="Eval fixture.",
        ships_in_product=None,
        concepts_introduced=[],
        relation_to_prior=[],
        confidence=0.9,
        evidence_quotes=[quote],
    )


def make_draft(claim_text: str, quote: str) -> DraftSynthesis:
    claim = Claim(
        text=claim_text,
        supporting_post_ids=["p1"],
        supporting_quotes=[quote],
    )
    return DraftSynthesis(
        synthesis_type="digest",
        title="Eval Draft",
        sections=[{"heading": "Eval", "claims": [claim], "prose": claim_text}],
        posts_covered=["p1"],
    )


def eval_critic() -> tuple[int, int]:
    critic = Critic(Settings(google_api_key="test-key"))
    cases = load_jsonl(EVAL_DIR / "critic_cases.jsonl")

    passed = 0
    for case in cases:
        draft = make_draft(case["claim"], case["quote"])
        post = make_post("p1", case["quote"])
        report = critic.review(draft, [post])
        actual = report.verdicts[0].verdict
        if actual == case["expected"]:
            passed += 1
        else:
            print(f"[critic fail] {case['id']}: expected={case['expected']} actual={actual}")

    return passed, len(cases)


def eval_guardrails() -> tuple[int, int]:
    cases = load_jsonl(EVAL_DIR / "guardrail_cases.jsonl")

    passed = 0
    for case in cases:
        result = check_injection(case["input"])
        actual = "safe" if result.safe else "blocked"
        if actual == case["expected"]:
            passed += 1
        else:
            print(f"[guardrail fail] {case['id']}: expected={case['expected']} actual={actual}")

    return passed, len(cases)


def eval_pii() -> tuple[int, int]:
    cases = load_jsonl(EVAL_DIR / "pii_cases.jsonl")

    passed = 0
    for case in cases:
        redacted = redact_pii(case["input"])
        if case["expected_contains"] in redacted:
            passed += 1
        else:
            print(f"[pii fail] {case['id']}: expected token not found in {redacted}")

    return passed, len(cases)


def eval_source_trust() -> tuple[int, int]:
    cases = load_jsonl(EVAL_DIR / "source_trust_cases.jsonl")

    passed = 0
    for case in cases:
        result = score_source(
            source=case["source"],
            url=case["url"],
            published_at=case["published_at"],
            content_type=case["content_type"],
        )

        if "expected_min_score" in case:
            ok = result.score >= case["expected_min_score"]
        else:
            ok = result.score <= case["expected_max_score"]

        if ok:
            passed += 1
        else:
            print(f"[source trust fail] {case['id']}: score={result.score}, reason={result.reason}")

    return passed, len(cases)


def eval_output_citations() -> tuple[int, int]:
    cases = load_jsonl(EVAL_DIR / "output_cases.jsonl")

    passed = 0
    for case in cases:
        claim = Claim(
            text=case["claim"],
            supporting_post_ids=case["supporting_post_ids"],
            supporting_quotes=case["supporting_quotes"],
        )
        draft = DraftSynthesis(
            synthesis_type="digest",
            title="Output Eval",
            sections=[{"heading": "Claims", "claims": [claim], "prose": claim.text}],
            posts_covered=claim.supporting_post_ids,
        )

        result = check_claim_citations(draft)
        actual = "safe" if result.safe else "blocked"

        if actual == case["expected"]:
            passed += 1
        else:
            print(f"[output citation fail] {case['id']}: expected={case['expected']} actual={actual}")

    return passed, len(cases)


def print_metric(name: str, passed: int, total: int) -> None:
    rate = passed / total if total else 0
    print(f"{name}: {passed}/{total} passed ({rate:.0%})")


def main() -> None:
    print("TrustCritic Eval Results")
    print("=" * 32)

    print_metric("Critic verdict accuracy", *eval_critic())
    print_metric("Prompt injection block rate", *eval_guardrails())
    print_metric("PII redaction success", *eval_pii())
    print_metric("Source trust scoring accuracy", *eval_source_trust())
    print_metric("Output citation check rate", *eval_output_citations())


if __name__ == "__main__":
    main()