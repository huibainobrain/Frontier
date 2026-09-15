"""Interface contracts for the agent system.

Single source of truth for the data shapes that flow between agents.
Every agent imports from this module; changing a schema has blast
radius across the whole team, so edits should go through PR review.

All schemas are ``@dataclass``. They expose:

* ``to_dict()`` — JSON-safe nested dict (uses ``dataclasses.asdict``).
* ``from_dict()`` — classmethod that hydrates from such a dict,
  recursively reconstructing nested dataclasses where applicable.
* ``__post_init__`` — cheap validation for numeric ranges and Literal
  enum-like fields, so a bad dict raises ``ValueError`` *at the
  boundary* rather than corrupting downstream agents.

Type-level enums (``Literal[...]``) are also re-exported so consumers
can ``from agent_system.schemas import SynthesisType`` instead of
inlining the string union.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Literal, get_args

# ---------------------------------------------------------------------------
# Literal enums — exported for type hints elsewhere
# ---------------------------------------------------------------------------

ContentType = Literal["blog", "paper", "release_notes"]
Category = Literal[
    "capability", "safety", "engineering", "product", "interpretability", "eval"
]
RelationType = Literal["extends", "contradicts", "complements", "independent"]
SynthesisType = Literal["digest", "tracker", "comparison", "reading_plan"]
# "verification_unavailable": the Critic's semantic-verification LLM
# call could not be completed (transient service failure after
# retries, or another non-retryable error) — distinct from
# "unsupported", which means verification *ran* and the evidence
# doesn't hold up. Conflating the two would mean a Gemini 503 gets
# reported to the user as "evidence contradicts/doesn't support this
# claim", which is simply false. See critic/agent.py._semantic_verify.
Verdict = Literal[
    "supported", "partial", "unsupported", "contradicted", "verification_unavailable"
]
TraceStatus = Literal["ok", "error", "cached"]

# Bumped whenever AnalyzedPost's shape changes in a way that makes an
# older stored record meaningfully incomplete (e.g. P0-1 added
# title/source/organization/url/published_at — a record analyzed
# before that has none of them). A stored record without a
# schema_version, or with one below this, is treated as stale and
# never reused as a valid cache/RAG hit — see
# storage.db.get_analyzed_post. Bump this again the next time
# AnalyzedPost gains fields that existing callers actually depend on.
CURRENT_ANALYSIS_SCHEMA_VERSION = 2


_VALID_CONTENT_TYPES = set(get_args(ContentType))
_VALID_CATEGORIES = set(get_args(Category))
_VALID_RELATIONS = set(get_args(RelationType))
_VALID_SYNTHESIS = set(get_args(SynthesisType))
_VALID_VERDICTS = set(get_args(Verdict))
_VALID_TRACE_STATUS = set(get_args(TraceStatus))


def now_iso() -> str:
    """ISO 8601 timestamp in UTC, second resolution."""

    return datetime.now(UTC).isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# User
# ---------------------------------------------------------------------------


@dataclass
class UserProfile:
    user_id: str
    interests: list[str]
    role_target: str
    seniority: str
    reading_history: list[str] = field(default_factory=list)
    feedback_log: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> UserProfile:
        return cls(
            user_id=str(d["user_id"]),
            interests=list(d.get("interests", [])),
            role_target=str(d.get("role_target", "")),
            seniority=str(d.get("seniority", "")),
            reading_history=list(d.get("reading_history", [])),
            feedback_log=list(d.get("feedback_log", [])),
        )


# ---------------------------------------------------------------------------
# Scout output
# ---------------------------------------------------------------------------


@dataclass
class RawPost:
    post_id: str
    source: str
    url: str
    title: str
    authors: list[str]
    published_at: str
    content: str
    content_type: str
    linked_arxiv_ids: list[str] = field(default_factory=list)
    fetched_at: str = field(default_factory=now_iso)
    raw_html_hash: str = ""

    def __post_init__(self) -> None:
        if self.content_type not in _VALID_CONTENT_TYPES:
            raise ValueError(
                f"RawPost.content_type={self.content_type!r} not in "
                f"{sorted(_VALID_CONTENT_TYPES)}"
            )

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> RawPost:
        return cls(
            post_id=str(d["post_id"]),
            source=str(d["source"]),
            url=str(d["url"]),
            title=str(d.get("title", "")),
            authors=list(d.get("authors", [])),
            published_at=str(d.get("published_at", "")),
            content=str(d.get("content", "")),
            content_type=str(d.get("content_type", "blog")),
            linked_arxiv_ids=list(d.get("linked_arxiv_ids", [])),
            fetched_at=str(d.get("fetched_at", now_iso())),
            raw_html_hash=str(d.get("raw_html_hash", "")),
        )


# ---------------------------------------------------------------------------
# Analyst output
# ---------------------------------------------------------------------------


@dataclass
class AnalyzedPost:
    post_id: str
    category: str
    key_claim: str
    practitioner_takeaway: str
    ships_in_product: bool | None
    concepts_introduced: list[str]
    relation_to_prior: list[dict]  # [{"post_id": ..., "relation": "extends"|...}]
    confidence: float
    evidence_quotes: list[str]
    analyzed_at: str = field(default_factory=now_iso)

    # Provenance — copied verbatim from the source RawPost at analysis
    # time (never LLM-generated: an LLM re-stating a title/date/url from
    # memory is exactly the failure mode that broke Comparison/Tracker/
    # Reading Plan/Citation downstream). Default to "" / [] rather than
    # being required so old rows persisted before these fields existed
    # (SQLite payload_json / Chroma metadata) still deserialize instead
    # of raising.
    title: str = ""
    source: str = ""
    url: str = ""
    authors: list[str] = field(default_factory=list)
    published_at: str = ""
    content_type: str = ""
    # Publisher/lab, e.g. "Anthropic" — distinct from ``source`` (the
    # internal source_id, e.g. "anthropic"). Only set when the source
    # itself unambiguously identifies one organization (an official lab
    # blog); left "" for aggregators like arXiv/Semantic Scholar/OpenAlex
    # where a single paper's affiliation can't be inferred from the
    # source alone and must never be guessed by an LLM.
    organization: str = ""
    # Defaults to CURRENT so every freshly-constructed AnalyzedPost
    # auto-stamps the current shape with no call-site changes needed.
    # A record loaded from storage keeps whatever version it was saved
    # with (see from_dict) — that's what lets a stale row be detected.
    schema_version: int = CURRENT_ANALYSIS_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.category not in _VALID_CATEGORIES:
            raise ValueError(
                f"AnalyzedPost.category={self.category!r} not in "
                f"{sorted(_VALID_CATEGORIES)}"
            )
        if not 0.0 <= float(self.confidence) <= 1.0:
            raise ValueError(
                f"AnalyzedPost.confidence={self.confidence} must be in [0.0, 1.0]"
            )
        for r in self.relation_to_prior:
            rel = r.get("relation") if isinstance(r, dict) else None
            if rel is not None and rel not in _VALID_RELATIONS:
                raise ValueError(
                    f"relation_to_prior.relation={rel!r} not in "
                    f"{sorted(_VALID_RELATIONS)}"
                )

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> AnalyzedPost:
        return cls(
            post_id=str(d["post_id"]),
            category=str(d["category"]),
            key_claim=str(d.get("key_claim", "")),
            practitioner_takeaway=str(d.get("practitioner_takeaway", "")),
            ships_in_product=d.get("ships_in_product"),
            concepts_introduced=list(d.get("concepts_introduced", [])),
            relation_to_prior=list(d.get("relation_to_prior", [])),
            confidence=float(d.get("confidence", 0.0)),
            evidence_quotes=list(d.get("evidence_quotes", [])),
            analyzed_at=str(d.get("analyzed_at", now_iso())),
            title=str(d.get("title", "")),
            source=str(d.get("source", "")),
            url=str(d.get("url", "")),
            authors=list(d.get("authors", [])),
            published_at=str(d.get("published_at", "")),
            content_type=str(d.get("content_type", "")),
            organization=str(d.get("organization", "")),
            # A record with no schema_version key predates this field
            # entirely (schema_version=1, always stale) — never default
            # to CURRENT here, that would defeat the whole point.
            schema_version=int(d.get("schema_version", 1)),
        )


# ---------------------------------------------------------------------------
# Synthesizer output
# ---------------------------------------------------------------------------


@dataclass
class Claim:
    text: str
    supporting_post_ids: list[str]
    supporting_quotes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> Claim:
        return cls(
            text=str(d.get("text", "")),
            supporting_post_ids=list(d.get("supporting_post_ids", [])),
            supporting_quotes=list(d.get("supporting_quotes", [])),
        )


@dataclass
class DraftSynthesis:
    synthesis_type: str
    title: str
    sections: list[dict]  # [{"heading", "claims": list[Claim|dict], "prose"}]
    posts_covered: list[str]
    generated_at: str = field(default_factory=now_iso)

    def __post_init__(self) -> None:
        if self.synthesis_type not in _VALID_SYNTHESIS:
            raise ValueError(
                f"DraftSynthesis.synthesis_type={self.synthesis_type!r} not in "
                f"{sorted(_VALID_SYNTHESIS)}"
            )

    def to_dict(self) -> dict:
        # Sections may carry Claim dataclasses; collapse them to dicts.
        sections_serialized: list[dict] = []
        for sec in self.sections:
            sec_copy = dict(sec) if isinstance(sec, dict) else {}
            claims = sec_copy.get("claims", [])
            sec_copy["claims"] = [
                c.to_dict() if isinstance(c, Claim) else dict(c) for c in claims
            ]
            sections_serialized.append(sec_copy)
        return {
            "synthesis_type": self.synthesis_type,
            "title": self.title,
            "sections": sections_serialized,
            "posts_covered": list(self.posts_covered),
            "generated_at": self.generated_at,
        }

    @classmethod
    def from_dict(cls, d: dict) -> DraftSynthesis:
        sections = []
        for sec in d.get("sections", []):
            sec_copy = dict(sec) if isinstance(sec, dict) else {}
            claims_raw = sec_copy.get("claims", [])
            sec_copy["claims"] = [
                c if isinstance(c, Claim) else Claim.from_dict(c) for c in claims_raw
            ]
            sections.append(sec_copy)
        return cls(
            synthesis_type=str(d["synthesis_type"]),
            title=str(d.get("title", "")),
            sections=sections,
            posts_covered=list(d.get("posts_covered", [])),
            generated_at=str(d.get("generated_at", now_iso())),
        )


# ---------------------------------------------------------------------------
# Critic output
# ---------------------------------------------------------------------------


@dataclass
class ClaimVerdict:
    claim: Claim
    verdict: str
    reasoning: str
    corrected_text: str | None = None

    def __post_init__(self) -> None:
        if self.verdict not in _VALID_VERDICTS:
            raise ValueError(
                f"ClaimVerdict.verdict={self.verdict!r} not in "
                f"{sorted(_VALID_VERDICTS)}"
            )

    def to_dict(self) -> dict:
        return {
            "claim": self.claim.to_dict() if isinstance(self.claim, Claim) else self.claim,
            "verdict": self.verdict,
            "reasoning": self.reasoning,
            "corrected_text": self.corrected_text,
        }

    @classmethod
    def from_dict(cls, d: dict) -> ClaimVerdict:
        claim_raw = d["claim"]
        claim = claim_raw if isinstance(claim_raw, Claim) else Claim.from_dict(claim_raw)
        return cls(
            claim=claim,
            verdict=str(d["verdict"]),
            reasoning=str(d.get("reasoning", "")),
            corrected_text=d.get("corrected_text"),
        )


@dataclass
class CriticReport:
    verdicts: list[ClaimVerdict]
    num_unsupported: int
    revision_needed: bool
    revision_notes: str = ""

    def to_dict(self) -> dict:
        return {
            "verdicts": [
                v.to_dict() if isinstance(v, ClaimVerdict) else dict(v)
                for v in self.verdicts
            ],
            "num_unsupported": int(self.num_unsupported),
            "revision_needed": bool(self.revision_needed),
            "revision_notes": self.revision_notes,
        }

    @classmethod
    def from_dict(cls, d: dict) -> CriticReport:
        verdicts = [
            v if isinstance(v, ClaimVerdict) else ClaimVerdict.from_dict(v)
            for v in d.get("verdicts", [])
        ]
        return cls(
            verdicts=verdicts,
            num_unsupported=int(d.get("num_unsupported", 0)),
            revision_needed=bool(d.get("revision_needed", False)),
            revision_notes=str(d.get("revision_notes", "")),
        )


@dataclass
class VerifiedSynthesis:
    draft: DraftSynthesis
    critic_report: CriticReport
    revision_count: int
    final: bool
    # post_id -> {"title", "source", "organization", "published_at", "url"}.
    # Lets a caller resolve a claim's bare "[post_id]" citation back to
    # something a human can read, without a second lookup against
    # storage. Optional/"" fields default empty so old persisted rows
    # (saved before this field existed) still deserialize fine.
    citations: dict[str, dict[str, str]] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "draft": self.draft.to_dict()
            if isinstance(self.draft, DraftSynthesis)
            else self.draft,
            "critic_report": self.critic_report.to_dict()
            if isinstance(self.critic_report, CriticReport)
            else self.critic_report,
            "revision_count": int(self.revision_count),
            "final": bool(self.final),
            "citations": dict(self.citations),
        }

    @classmethod
    def from_dict(cls, d: dict) -> VerifiedSynthesis:
        draft_raw = d["draft"]
        report_raw = d["critic_report"]
        return cls(
            draft=draft_raw
            if isinstance(draft_raw, DraftSynthesis)
            else DraftSynthesis.from_dict(draft_raw),
            critic_report=report_raw
            if isinstance(report_raw, CriticReport)
            else CriticReport.from_dict(report_raw),
            revision_count=int(d.get("revision_count", 0)),
            final=bool(d.get("final", False)),
            citations=dict(d.get("citations", {})),
        )


# ---------------------------------------------------------------------------
# Orchestrator-only
# ---------------------------------------------------------------------------


@dataclass
class SourcePlan:
    sources_to_query: list[str]
    time_window_days: int
    filter_keywords: list[str]
    max_posts: int

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> SourcePlan:
        return cls(
            sources_to_query=list(d.get("sources_to_query", [])),
            time_window_days=int(d.get("time_window_days", 7)),
            filter_keywords=list(d.get("filter_keywords", [])),
            max_posts=int(d.get("max_posts", 20)),
        )


@dataclass
class ExecutionPlan:
    intent: str
    source_plan: SourcePlan
    synthesis_type: str
    user_profile_summary: str
    reasoning: str
    # The real-world entities the user's goal is *about* (e.g.
    # ["OpenAI", "Anthropic"] for "compare OpenAI and Anthropic..."),
    # resolved once by the Planner from the goal text itself. Distinct
    # from source_plan.sources_to_query (where to search, e.g.
    # "arxiv_cs_ai") and from whatever sources a research round
    # actually ends up using — this field must never be derived from
    # either of those, only from the goal. Empty when the goal doesn't
    # name specific target entities (e.g. a general digest).
    research_targets: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.intent not in _VALID_SYNTHESIS:
            raise ValueError(
                f"ExecutionPlan.intent={self.intent!r} not in "
                f"{sorted(_VALID_SYNTHESIS)}"
            )
        if self.synthesis_type not in _VALID_SYNTHESIS:
            raise ValueError(
                f"ExecutionPlan.synthesis_type={self.synthesis_type!r} not in "
                f"{sorted(_VALID_SYNTHESIS)}"
            )

    def to_dict(self) -> dict:
        return {
            "intent": self.intent,
            "source_plan": self.source_plan.to_dict()
            if isinstance(self.source_plan, SourcePlan)
            else self.source_plan,
            "synthesis_type": self.synthesis_type,
            "user_profile_summary": self.user_profile_summary,
            "reasoning": self.reasoning,
            "research_targets": list(self.research_targets),
        }

    @classmethod
    def from_dict(cls, d: dict) -> ExecutionPlan:
        sp_raw = d["source_plan"]
        return cls(
            intent=str(d["intent"]),
            source_plan=sp_raw
            if isinstance(sp_raw, SourcePlan)
            else SourcePlan.from_dict(sp_raw),
            synthesis_type=str(d["synthesis_type"]),
            user_profile_summary=str(d.get("user_profile_summary", "")),
            reasoning=str(d.get("reasoning", "")),
            research_targets=list(d.get("research_targets", [])),
        )


# ---------------------------------------------------------------------------
# Observability
# ---------------------------------------------------------------------------


@dataclass
class AgentTrace:
    trace_id: str
    agent_name: str
    started_at: str
    ended_at: str
    input_tokens: int = 0
    output_tokens: int = 0
    model: str = ""
    cost_estimate: float = 0.0
    status: str = "ok"

    def __post_init__(self) -> None:
        if self.status not in _VALID_TRACE_STATUS:
            raise ValueError(
                f"AgentTrace.status={self.status!r} not in "
                f"{sorted(_VALID_TRACE_STATUS)}"
            )

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> AgentTrace:
        return cls(
            trace_id=str(d["trace_id"]),
            agent_name=str(d["agent_name"]),
            started_at=str(d.get("started_at", "")),
            ended_at=str(d.get("ended_at", "")),
            input_tokens=int(d.get("input_tokens", 0)),
            output_tokens=int(d.get("output_tokens", 0)),
            model=str(d.get("model", "")),
            cost_estimate=float(d.get("cost_estimate", 0.0)),
            status=str(d.get("status", "ok")),
        )


__all__ = [
    "AgentTrace",
    "AnalyzedPost",
    "Category",
    "Claim",
    "ClaimVerdict",
    "ContentType",
    "CURRENT_ANALYSIS_SCHEMA_VERSION",
    "CriticReport",
    "DraftSynthesis",
    "ExecutionPlan",
    "RawPost",
    "RelationType",
    "SourcePlan",
    "SynthesisType",
    "TraceStatus",
    "UserProfile",
    "Verdict",
    "VerifiedSynthesis",
    "now_iso",
]
