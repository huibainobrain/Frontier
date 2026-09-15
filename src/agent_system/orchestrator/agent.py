"""Orchestrator agent built on Google ADK.

The Orchestrator owns intent classification, planning, the
Evaluator-driven research/replan loop, sub-agent routing, the Critic
revision loop, and final assembly of the :class:`VerifiedSynthesis`.
Two HITL checkpoints (plan approval and synthesis review) keep a human
in the loop.

Composition with ADK primitives::

    SequentialAgent("orchestrator_pipeline")
      ├── Agent("intent_classifier")          # Gemini Flash, JSON output
      ├── Agent("planner")                    # Gemini Flash, JSON output
      └── LoopAgent("critic_revision_loop")
            ├── Agent("synthesizer")
            └── Agent("critic")               # escalates when accepted
            max_iterations = settings.max_revisions + 1

The two LLM-driven steps that *belong* to the Orchestrator (intent
classification + planning, and the Evaluator used by the replan loop)
are full ADK Agents. Scout, Analyst, Synthesizer, and Critic are
deliberately plain Python classes driven directly by the Orchestrator,
not ADK Agents — the data flowing between them is strongly typed
(``RawPost`` → ``AnalyzedPost`` → ``DraftSynthesis`` → ``CriticReport``
→ ``VerifiedSynthesis``) and ADK's session-state dict layer would only
obscure that. ``build_pipeline`` optionally accepts ADK-wrapped
versions of them (see its docstring) for anyone who wants to compose
them into the same ``SequentialAgent`` end-to-end instead; the
production path never does.

Pipeline (each step wrapped in an observability span)::

     1. Input guardrail
     2. ADK ``intent_classifier`` (Gemini Flash, strict JSON)
     3. ADK ``planner``           (Gemini Flash, strict JSON)
     4. HITL #1 — plan approval (callback)
     5. Research loop: Scout fetch -> Analyst analyze -> ADK
        ``evaluator`` judges sufficiency -> targeted replan if needed
        (bounded by ``settings.max_research_rounds``, or skipped
        entirely when ``settings.agentic_replan_enabled`` is False —
        see ``_run_research_loop``)
     6. Synthesizer: synthesize
     7. Critic loop, capped at ``settings.max_revisions``
     8. HITL #2 — synthesis review (callback may edit)
     9. Output guardrail (deterministic only — no LLM rewrite after Critic)
    10. Persist VerifiedSynthesis + feedback row + run telemetry summary
    11. Return VerifiedSynthesis
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from google.adk.agents import Agent, LoopAgent, SequentialAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types as genai_types

from agent_system.analyst.agent import Analyst
from agent_system.config import Settings, get_settings
from agent_system.critic.agent import Critic
from agent_system.guardrails import run_guardrails
from agent_system.guardrails.output_checker import find_missing_citations, strip_claims
from agent_system.llm_retry import (
    async_call_with_retry,
    call_with_retry,
    note_usage,
    note_usage_from_gemini_response,
    set_current_run_id,
)
from agent_system.observability.tracing import trace_span
from agent_system.prompts import load_prompt
from agent_system.schemas import (
    AnalyzedPost,
    DraftSynthesis,
    ExecutionPlan,
    RawPost,
    SourcePlan,
    UserProfile,
    VerifiedSynthesis,
)
from agent_system.scout.agent import Scout
from agent_system.scout.url_utils import canonicalize_url
from agent_system.storage import db as storage_db
from agent_system.synthesizer.agent import Synthesizer

logger = logging.getLogger(__name__)

APP_NAME = "frontier_lit_agent"

VALID_INTENTS: tuple[str, ...] = (
    "digest",
    "tracker",
    "comparison",
    "reading_plan",
)

AVAILABLE_SOURCES: tuple[str, ...] = (
    "anthropic",
    "openai",
    "deepmind",
    "google_research",
    "meta_ai",
    "hugging_face",
    "arxiv_cs_ai",
    "semantic_scholar",
    "openalex",
)

# The subset of AVAILABLE_SOURCES that is NOT a single organization —
# multi-institution paper/citation indexes. A research_target
# containing one of these (e.g. the Planner confusing "arxiv_cs_ai" for
# an entity name) is a strong signal of exactly the retrieval-source/
# comparison-target mixup P0-2 fixes; the lab-specific ids
# (anthropic/openai/...) are deliberately excluded from this set since
# those DO correspond to real organizations and a target like "OpenAI"
# is legitimate even though it coincidentally shares a slug.
_NON_ORGANIZATION_SOURCE_IDS: frozenset[str] = frozenset(
    {"arxiv_cs_ai", "semantic_scholar", "openalex"}
)

INTENT_OUTPUT_KEY = "intent_payload"
PLAN_OUTPUT_KEY = "execution_plan"
EVALUATOR_OUTPUT_KEY = "research_evaluation"

# Output guard: an isolated claim missing a citation gets repaired or
# dropped, not blocked. Only a systemic gap — more than this fraction of
# the draft's claims still uncited after one repair attempt — actually
# fails the output. See Orchestrator._enforce_citation_completeness.
SYSTEMIC_CITATION_GAP_RATIO = 0.5

# Deterministic, high-confidence-only intent recovery — used ONLY when
# the Intent LLM call fails even after retries (see
# Orchestrator._recover_intent_deterministically). Checked in this
# order; comparison is checked first since it has the most to lose from
# a misfire (silently downgrading a comparison request to digest was
# exactly the bug this exists to prevent). Deliberately narrow regex
# patterns over very explicit phrasing rather than a real NLP
# classifier — a request that matches none of these is NOT guessed, see
# IntentClassificationUnavailableError.
_INTENT_RECOVERY_PATTERNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "comparison",
        (
            r"\bcompare\b",
            r"\bcomparison\b",
            r"\bvs\.?\b",
            r"\bversus\b",
            r"\bdifference between\b",
        ),
    ),
    (
        "reading_plan",
        (
            r"\breading plan\b",
            r"\breading list\b",
            r"\breading order\b",
            r"\bwhat should i read\b",
            r"\bhow should i learn\b",
            r"\blearning path\b",
        ),
    ),
    (
        "tracker",
        (
            r"\btrack\b",
            r"\btracking\b",
            r"\btimeline of\b",
            r"\bevolution of\b",
            r"\bhas changed\b",
            r"\bhave changed\b",
            r"\bdevelopment of\b",
        ),
    ),
    (
        "digest",
        (
            r"\bweekly digest\b",
            r"\bdigest\b",
            r"\brecent developments\b",
            r"\blatest updates\b",
            r"\bwhat happened recently\b",
            r"\bwhat'?s new\b",
        ),
    ),
)


class IntentClassificationUnavailableError(Exception):
    """Raised when the Intent LLM call fails even after retries AND the
    user's own request text isn't explicit enough for deterministic
    keyword recovery (``_INTENT_RECOVERY_PATTERNS``) to confidently name
    a task type. Deliberately never resolved to a guessed specific
    intent (e.g. defaulting to 'digest') in that case — an
    infrastructure failure must not redefine what the user actually
    asked for. See Orchestrator._classify_intent."""

    def __init__(self, user_goal: str) -> None:
        self.user_goal = user_goal
        super().__init__(
            "Could not determine the type of request this is: intent "
            "classification is temporarily unavailable and the request "
            "isn't explicit enough to recover automatically. Please "
            "retry, or rephrase using an explicit verb like 'compare', "
            "'track', or 'reading plan'."
        )


class ResearchEvidenceUnavailableError(Exception):
    """Raised immediately before the Synthesizer would run, when there
    is zero valid analyzed evidence to synthesize from. Two distinct
    reasons (see Orchestrator._run_research_loop):

    * ``"no_retrieved_evidence"`` — Scout never retrieved any source at
      all; there was nothing to analyze in the first place.
    * ``"analysis_unavailable"`` — sources were retrieved but every
      analysis attempt failed; a processing problem, not a coverage
      gap, so it must never be handed to the Evaluator as "search more".

    Never caught anywhere to paper over with a fabricated "research"
    answer — zero valid evidence means no research synthesis.
    """

    _MESSAGES = {
        "no_retrieved_evidence": (
            "No relevant sources could be found for this request. Try "
            "rephrasing your query, or retry later."
        ),
        "analysis_unavailable": (
            "The system retrieved relevant sources but could not "
            "successfully analyze enough evidence to produce a reliable "
            "answer. Please retry later."
        ),
    }

    def __init__(self, reason: str) -> None:
        self.reason = reason
        message = self._MESSAGES.get(
            reason,
            "No usable research evidence was available to answer this "
            "request. Please retry later.",
        )
        super().__init__(message)


PlanCallback = Callable[[ExecutionPlan], bool]
SynthesisCallback = Callable[[VerifiedSynthesis], VerifiedSynthesis]
# Fired the moment the run's session_id/run_id is known (before any
# research happens) and once more with the final run summary dict (see
# _build_run_summary) — lets a caller like a web frontend correlate a
# request with live telemetry (agent_system.observability.
# summarize_llm_calls) for progress polling, without querying storage
# or threading extra state through run_async's return type. Both are
# fire-and-forget notifications (no return value), unlike PlanCallback/
# SynthesisCallback which gate/transform the pipeline.
RunStartedCallback = Callable[[str], None]
RunSummaryCallback = Callable[[dict[str, Any]], None]


@dataclass
class ResearchState:
    """Orchestrator-internal working memory for the research loop.

    Deliberately NOT in ``schemas.py``: that module is the cross-agent
    interface contract (RawPost/AnalyzedPost/...), reviewed as such;
    this is ephemeral bookkeeping for one ``run_async`` call, never
    handed to Scout/Analyst/Synthesizer/Critic directly (they still get
    plain lists/SourcePlans, unchanged). Exists so replan state
    (queries already run, sources already used) lives in one place
    instead of a pile of loop-local variables — the concrete failure
    mode this prevents is a replan round searching the same thing again
    because nothing remembered what round 1 already tried.
    """

    original_user_query: str
    intent: str
    research_goal: str
    max_rounds: int
    # Set once from ExecutionPlan.research_targets and never mutated by
    # the loop below — the whole point is that replanning (which
    # sources/queries to use next) must never feed back into what the
    # answer is scoped to. See schemas.ExecutionPlan.research_targets.
    research_targets: list[str] = field(default_factory=list)
    research_round: int = 0
    executed_queries: list[str] = field(default_factory=list)
    used_sources: list[str] = field(default_factory=list)
    # Cumulative count of genuinely new (deduped) posts analyzed so far
    # across all rounds — the global budget's accounting unit. See
    # settings.max_total_posts.
    total_new_posts: int = 0
    # Cumulative count of raw posts Scout has returned across every
    # round, BEFORE dedup (unlike total_new_posts above) — used only to
    # tell "Scout found nothing at all" (no_retrieved_evidence) apart
    # from "Scout found sources but none could be analyzed"
    # (analysis_unavailable). See _run_research_loop's zero-evidence
    # hard stop.
    total_retrieved_posts: int = 0
    evidence_gaps: list[dict] = field(default_factory=list)
    next_actions: list[dict] = field(default_factory=list)
    stop_reason: str = ""


def _auto_approve_plan(_: ExecutionPlan) -> bool:
    return True


def _auto_accept_synthesis(verified: VerifiedSynthesis) -> VerifiedSynthesis:
    return verified


# ---------------------------------------------------------------------------
# ADK agent factories
# ---------------------------------------------------------------------------


def build_intent_agent(settings: Settings) -> Agent:
    """ADK ``Agent`` (LlmAgent) that classifies user intent.

    Uses Gemini Flash with ``response_mime_type='application/json'`` so the
    raw response stored in ``state[INTENT_OUTPUT_KEY]`` is already parseable
    JSON of the form ``{"intent", "confidence", "reasoning"}``.
    """

    return Agent(
        name="intent_classifier",
        model=settings.model_flash,
        description=(
            "Classify a user goal into exactly one of: digest, tracker, "
            "comparison, reading_plan. Output strict JSON."
        ),
        instruction=load_prompt("orchestrator_intent"),
        output_key=INTENT_OUTPUT_KEY,
        generate_content_config=genai_types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.1,
        ),
    )


def build_planner_agent(settings: Settings) -> Agent:
    """ADK ``Agent`` that produces a concrete ExecutionPlan JSON.

    The list of allowed sources is baked into the system prompt at
    construction time (``load_prompt(... available_sources=...)``); only
    the user goal / intent / profile flow through the user message at
    runtime.
    """

    instruction = load_prompt(
        "orchestrator_planner",
        available_sources=", ".join(AVAILABLE_SOURCES),
    )
    return Agent(
        name="planner",
        model=settings.model_flash,
        description=(
            "Given a classified intent + user goal + profile, produce an "
            "ExecutionPlan: sources to query, time window, filter keywords, "
            "max_posts, synthesis_type. Output strict JSON."
        ),
        instruction=instruction,
        output_key=PLAN_OUTPUT_KEY,
        generate_content_config=genai_types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.2,
        ),
    )


def build_evaluator_agent(settings: Settings) -> Agent:
    """ADK ``Agent`` that judges research sufficiency and — if
    insufficient — proposes the next targeted search action.

    Combines what the task brief calls the "Evidence/Coverage
    Evaluator" and the "Replanner" into one structured-output call
    rather than two: the JSON schema the brief itself specifies for the
    Evaluator already bundles ``evidence_gaps`` with ``next_actions`` in
    one object, and the model best placed to name a gap precisely is
    the same one that just read the evidence — splitting that into a
    second round-trip would add latency/cost without adding any
    information the first call didn't already have. Same reasoning
    pattern as ``intent_classifier``/``planner`` above: single-turn,
    strict JSON, no tool calls.
    """

    instruction = load_prompt(
        "orchestrator_evaluator",
        available_sources=", ".join(AVAILABLE_SOURCES),
    )
    return Agent(
        name="research_evaluator",
        model=settings.model_flash,
        description=(
            "Given a research goal and the evidence gathered so far, judge "
            "whether it's sufficient. If not, name the specific evidence "
            "gap and propose a targeted next search action. Output strict "
            "JSON."
        ),
        instruction=instruction,
        output_key=EVALUATOR_OUTPUT_KEY,
        generate_content_config=genai_types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.2,
        ),
    )


def build_pipeline(
    settings: Settings,
    *,
    intent_agent: Agent,
    planner_agent: Agent,
    scout_agent: Agent | None = None,
    analyst_agent: Agent | None = None,
    synthesizer_agent: Agent | None = None,
    critic_agent: Agent | None = None,
) -> SequentialAgent:
    """Compose the full ADK pipeline.

    Always includes the two orchestrator-owned agents. The four
    downstream slots are appended only when the caller passes an
    ADK-wrapped version of that sub-agent; the production Orchestrator
    never does, and drives Scout/Analyst/Synthesizer/Critic directly
    via plain Python over typed dataclasses instead (see the module
    docstring for why).
    """

    sub_agents: list[Any] = [intent_agent, planner_agent]
    if scout_agent is not None:
        sub_agents.append(scout_agent)
    if analyst_agent is not None:
        sub_agents.append(analyst_agent)
    if synthesizer_agent is not None and critic_agent is not None:
        critic_loop = LoopAgent(
            name="critic_revision_loop",
            description=(
                "Synthesize, then critique. Loop until the Critic accepts "
                "(escalates) or max_iterations is reached."
            ),
            sub_agents=[synthesizer_agent, critic_agent],
            max_iterations=int(settings.max_revisions) + 1,
        )
        sub_agents.append(critic_loop)
    return SequentialAgent(
        name="orchestrator_pipeline",
        description="End-to-end frontier literature synthesis pipeline.",
        sub_agents=sub_agents,
    )


# ---------------------------------------------------------------------------
# Orchestrator class
# ---------------------------------------------------------------------------


class Orchestrator:
    """Coordinates the multi-agent pipeline.

    Sub-agents are injected via the constructor so unit tests can hand
    in mocks; in production each defaults to its real implementation.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        scout: Scout | None = None,
        analyst: Analyst | None = None,
        synthesizer: Synthesizer | None = None,
        critic: Critic | None = None,
        on_plan_ready: PlanCallback | None = None,
        on_synthesis_ready: SynthesisCallback | None = None,
        on_run_started: RunStartedCallback | None = None,
        on_run_summary: RunSummaryCallback | None = None,
        intent_agent: Agent | None = None,
        planner_agent: Agent | None = None,
        evaluator_agent: Agent | None = None,
        session_service: Any | None = None,
    ) -> None:
        self.settings = settings if settings is not None else get_settings()
        self.scout = scout if scout is not None else Scout(self.settings)
        self.analyst = analyst if analyst is not None else Analyst(self.settings)
        self.synthesizer = (
            synthesizer if synthesizer is not None else Synthesizer(self.settings)
        )
        self.critic = critic if critic is not None else Critic(self.settings)
        self.on_plan_ready = on_plan_ready or _auto_approve_plan
        self.on_synthesis_ready = on_synthesis_ready or _auto_accept_synthesis
        self.on_run_started = on_run_started
        self.on_run_summary = on_run_summary
        # Lazy Gemini client (same pattern as Analyst/Synthesizer/Critic)
        # — used only by comparison-target recovery, the Orchestrator's
        # one direct (non-ADK) Gemini call. See _recover_research_targets.
        self._client: Any = None

        self.intent_agent = intent_agent or build_intent_agent(self.settings)
        self.planner_agent = planner_agent or build_planner_agent(self.settings)
        self.evaluator_agent = evaluator_agent or build_evaluator_agent(self.settings)
        self.pipeline = build_pipeline(
            self.settings,
            intent_agent=self.intent_agent,
            planner_agent=self.planner_agent,
        )
        self._session_service = session_service or InMemorySessionService()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, user_goal: str, profile: UserProfile) -> VerifiedSynthesis:
        """Synchronous facade — internally drives the async ADK runner."""

        return asyncio.run(self.run_async(user_goal, profile))

    async def run_async(
        self, user_goal: str, profile: UserProfile
    ) -> VerifiedSynthesis:
        """Execute the full pipeline and return a :class:`VerifiedSynthesis`."""

        with trace_span("orchestrator.run"):
            run_started = time.perf_counter()
            sanitized = self._guard_input(user_goal)
            session_id = await self._create_session(profile, sanitized)
            # Reuse the ADK session_id as the telemetry run_id (already
            # unique per run) — every LLM-call record produced anywhere
            # in this run gets tagged with it. See
            # agent_system.llm_retry.set_current_run_id.
            set_current_run_id(session_id)
            if self.on_run_started:
                self.on_run_started(session_id)

            intent, intent_reason = await self._classify_intent(
                sanitized, profile, session_id
            )
            plan = await self._generate_plan(
                sanitized, intent, profile, intent_reason, session_id
            )
            plan = self._recover_research_targets_if_needed(plan, sanitized)

            if not self._approve_plan(plan):
                raise RuntimeError("User rejected the execution plan; aborting run.")

            analyzed, used_sources, evidence_gaps, loop_stop_reason, research_rounds = (
                await self._run_research_loop(
                    sanitized, intent, profile, plan, session_id
                )
            )
            # Final fail-safe — belt-and-suspenders on top of the
            # research loop's own hard stop below: the Synthesizer must
            # never run with zero valid evidence, no matter which
            # upstream path produced an empty list. Generating a
            # "research answer" with nothing to cite risks the model
            # filling the gap from parametric knowledge dressed up as a
            # finding. See ResearchEvidenceUnavailableError.
            if not analyzed:
                raise ResearchEvidenceUnavailableError(
                    loop_stop_reason or "no_analyzed_evidence"
                )
            draft = self._run_synthesizer(
                analyzed,
                profile,
                plan.synthesis_type,
                research_targets=plan.research_targets,
                evidence_gaps=evidence_gaps,
            )
            verified = self._critic_loop(draft, analyzed)
            verified = self._approve_synthesis(verified)
            verified = self._guard_output(verified)
            verified = self._attach_citations(verified, analyzed)
            run_summary = self._build_run_summary(
                run_id=session_id,
                intent=intent,
                research_rounds=research_rounds,
                used_sources=used_sources,
                stop_reason=loop_stop_reason,
                analyzed_posts=len(analyzed),
                latency_ms=int((time.perf_counter() - run_started) * 1000),
            )
            if self.on_run_summary:
                self.on_run_summary(run_summary)
            self._persist(verified, profile, plan, sanitized, run_summary=run_summary)
            return verified

    # ------------------------------------------------------------------
    # ADK invocation
    # ------------------------------------------------------------------

    async def _create_session(self, profile: UserProfile, user_goal: str) -> str:
        session_id = f"sess-{uuid.uuid4().hex[:10]}"
        await self._session_service.create_session(
            app_name=APP_NAME,
            user_id=profile.user_id,
            session_id=session_id,
            state={
                "user_goal": user_goal,
                "user_profile_summary": self._summarise_profile(profile),
            },
        )
        return session_id

    async def _invoke_agent(
        self,
        agent: Agent,
        user_text: str,
        profile: UserProfile,
        session_id: str,
        output_key: str,
        *,
        component: str,
    ) -> dict[str, Any]:
        """Drive one ADK Agent end-to-end and return its parsed JSON output.

        Retries the whole call (a fresh ``Runner.run_async`` + session
        read) on a transient infra error via the shared
        ``agent_system.llm_retry`` policy — the actual Gemini call
        happens deep inside the ADK's own flow, so there's no narrower
        seam to wrap. Re-asking the same question in the same session on
        retry is a known, accepted simplification (see this module's own
        docstring / the P0-1 report's limitations) rather than minting a
        fresh session per attempt, which the single-turn classifier
        prompts here (Intent/Planner/Evaluator) don't need in practice.

        Raises the final exception when every attempt fails or the
        first failure isn't retryable — callers (``_classify_intent``,
        ``_generate_plan``, ``_evaluate_research_state``) each decide
        their own module-specific fallback.
        """

        async def _once() -> dict[str, Any]:
            runner = Runner(
                app_name=APP_NAME,
                agent=agent,
                session_service=self._session_service,
            )
            message = genai_types.Content(
                role="user", parts=[genai_types.Part(text=user_text)]
            )
            async for event in runner.run_async(
                user_id=profile.user_id,
                session_id=session_id,
                new_message=message,
            ):
                # Drain all events; the agent's final response is
                # captured by ADK into session.state[output_key] thanks
                # to ``output_key=`` on the Agent. The one thing worth
                # reading off an event directly is its usage_metadata
                # (google.adk.events.Event extends LlmResponse, which
                # carries the same usage_metadata a direct
                # generate_content() response does) — real token usage
                # for this ADK-driven call, same as the direct-call
                # sites via note_usage_from_gemini_response.
                usage = getattr(event, "usage_metadata", None)
                if usage is not None:
                    note_usage(
                        getattr(usage, "prompt_token_count", None),
                        getattr(usage, "candidates_token_count", None),
                    )

            session = await self._session_service.get_session(
                app_name=APP_NAME, user_id=profile.user_id, session_id=session_id
            )
            raw = session.state.get(output_key, "")
            if isinstance(raw, dict):
                return raw
            return _parse_json_or_raise(str(raw))

        model = getattr(agent, "model", "") or "unknown"
        return await async_call_with_retry(component, model, _once)

    async def _classify_intent(
        self, user_goal: str, profile: UserProfile, session_id: str
    ) -> tuple[str, str]:
        """Classify intent, with a deterministic *recovery* — not a
        blanket default — if the LLM call fails even after retries.

        A transient Gemini outage must not silently redefine what the
        user actually asked for: if the request's own wording is
        explicit enough (e.g. "compare X and Y"), that's recovered via
        :meth:`_recover_intent_deterministically`; otherwise the
        pipeline refuses to guess and raises
        :class:`IntentClassificationUnavailableError` instead of
        defaulting to a specific intent like 'digest'.
        """

        with trace_span("orchestrator.intent"):
            user_text = (
                f"User goal: {user_goal}\n"
                f"User profile: {self._summarise_profile(profile)}"
            )
            try:
                payload = await self._invoke_agent(
                    self.intent_agent,
                    user_text,
                    profile,
                    session_id,
                    INTENT_OUTPUT_KEY,
                    component="intent",
                )
            except Exception:
                logger.error(
                    "component=intent final_status=intent_llm_unavailable "
                    "— attempting deterministic recovery from the "
                    "request's own wording before giving up."
                )
                return self._recover_intent_deterministically(user_goal)

            intent = str(payload.get("intent", "digest")).strip().lower()
            if intent not in VALID_INTENTS:
                logger.warning(
                    "Unknown intent %r — falling back to 'digest'.", intent
                )
                intent = "digest"
            logger.info("component=intent intent_source=llm intent=%s", intent)
            return intent, str(payload.get("reasoning", "")).strip()

    @staticmethod
    def _recover_intent_from_text(user_goal: str) -> str | None:
        """Deterministic, high-confidence-only intent match against the
        raw user text — see ``_INTENT_RECOVERY_PATTERNS``. Returns
        ``None`` when nothing matches; callers must not guess a specific
        intent in that case."""

        text = user_goal.lower()
        for intent, patterns in _INTENT_RECOVERY_PATTERNS:
            for pattern in patterns:
                if re.search(pattern, text):
                    return intent
        return None

    def _recover_intent_deterministically(self, user_goal: str) -> tuple[str, str]:
        """Called only when the Intent LLM call is unavailable even
        after retries (see :meth:`_classify_intent`). Only ever returns
        an intent it can identify with high confidence from explicit
        phrasing in the user's own request — never from anything
        retrieved later (nothing has been retrieved yet at this point
        in the pipeline regardless). Raises
        :class:`IntentClassificationUnavailableError` rather than
        guessing when nothing matches confidently.
        """

        recovered = self._recover_intent_from_text(user_goal)
        if recovered is not None:
            logger.warning(
                "component=intent intent_source=deterministic_recovery "
                "intent=%s — LLM classification unavailable; recovered "
                "from explicit phrasing in the request.",
                recovered,
            )
            return recovered, "intent_source=deterministic_recovery"

        logger.error(
            "component=intent intent_source=unavailable "
            "final_status=intent_classification_unavailable — LLM "
            "classification unavailable and the request isn't explicit "
            "enough to recover deterministically; refusing to guess a "
            "task type."
        )
        raise IntentClassificationUnavailableError(user_goal)

    async def _generate_plan(
        self,
        user_goal: str,
        intent: str,
        profile: UserProfile,
        intent_reasoning: str,
        session_id: str,
    ) -> ExecutionPlan:
        """Generate the execution plan, with a deterministic fallback
        SearchPlan if the Planner LLM call fails even after retries —
        the user still gets a result, just with degraded planning
        quality (broad defaults) instead of no result at all. See
        :meth:`_fallback_plan`."""

        with trace_span("orchestrator.plan"):
            user_text = (
                f"User goal: {user_goal}\n"
                f"Classified intent: {intent}\n"
                f"Intent reasoning: {intent_reasoning}\n"
                f"User profile: {self._summarise_profile(profile)}"
            )
            try:
                payload = await self._invoke_agent(
                    self.planner_agent,
                    user_text,
                    profile,
                    session_id,
                    PLAN_OUTPUT_KEY,
                    component="planner",
                )
            except Exception:
                return self._fallback_plan(user_goal, intent, profile)
            return self._payload_to_plan(payload, intent, profile, intent_reasoning)

    def _fallback_plan(
        self, user_goal: str, intent: str, profile: UserProfile
    ) -> ExecutionPlan:
        """Deterministic plan used when the Planner LLM call fails even
        after retries. Broad default sources, a reasonable default time
        window, keywords derived straight from the raw user query (no
        LLM needed — reuses ``Scout._extract_keywords``), and a default
        max_posts. ``research_targets`` stays empty here — a fully
        failed Planner call has no reliable signal to invent targets
        from; if intent is comparison, target recovery
        (``_recover_research_targets_if_needed``) still gets a chance to
        resolve them from the original query, independent of why
        research_targets ended up empty.

        Marked in the plan's own ``reasoning`` field (there's no
        separate "is this a fallback" field on ExecutionPlan, and adding
        one for a single boolean would be more schema than this needs)
        plus an explicit ``planner_fallback`` log line — both are
        equally greppable trace signals.
        """

        keywords = Scout._extract_keywords(user_goal)
        source_plan = SourcePlan(
            sources_to_query=list(AVAILABLE_SOURCES[:5]),
            time_window_days=14,
            filter_keywords=keywords,
            max_posts=20,
        )
        logger.error(
            "component=planner final_status=planner_fallback — using a "
            "deterministic default plan; planning quality is degraded "
            "but the pipeline continues."
        )
        return ExecutionPlan(
            intent=intent,
            source_plan=source_plan,
            synthesis_type=intent,
            user_profile_summary=self._summarise_profile(profile),
            reasoning=(
                "planner_fallback: deterministic default plan used because "
                "the Planner LLM call failed after retries."
            ),
            research_targets=[],
        )

    def _payload_to_plan(
        self,
        payload: dict[str, Any],
        intent: str,
        profile: UserProfile,
        intent_reasoning: str,
    ) -> ExecutionPlan:
        sp_payload = payload.get("source_plan") or {}
        sources = self._coerce_sources(sp_payload.get("sources_to_query"))
        time_window = self._coerce_int(
            sp_payload.get("time_window_days"), default=7, lo=1
        )
        keywords = [
            str(k).strip()
            for k in (sp_payload.get("filter_keywords") or [])
            if str(k).strip()
        ]
        max_posts = self._coerce_int(
            sp_payload.get("max_posts"), default=20, lo=1, hi=100
        )
        source_plan = SourcePlan(
            sources_to_query=sources,
            time_window_days=time_window,
            filter_keywords=keywords,
            max_posts=max_posts,
        )
        synthesis_type = str(payload.get("synthesis_type", intent)).strip().lower()
        if synthesis_type not in VALID_INTENTS:
            synthesis_type = intent
        research_targets = self._coerce_research_targets(payload.get("research_targets"))
        plan = ExecutionPlan(
            intent=intent,
            source_plan=source_plan,
            synthesis_type=synthesis_type,
            user_profile_summary=self._summarise_profile(profile),
            reasoning=str(payload.get("reasoning", intent_reasoning)).strip(),
            research_targets=research_targets,
        )
        logger.info(
            "ExecutionPlan: intent=%s synthesis=%s sources=%s targets=%s "
            "window=%dd posts<=%d",
            plan.intent,
            plan.synthesis_type,
            sources,
            research_targets,
            time_window,
            max_posts,
        )
        return plan

    @staticmethod
    def _coerce_research_targets(raw: Any) -> list[str]:
        """Clean the Planner's research_targets list: strip whitespace,
        drop empties, dedupe case-insensitively (keeping the first
        casing seen). Deliberately does NOT validate against
        AVAILABLE_SOURCES or fall back to sources_to_query on empty —
        these are free-text entity names, a different vocabulary from
        source ids, and an empty result here must stay empty (see
        ExecutionPlan.research_targets: never reverse-derived from
        sources). Capped at 6 — a "research target" list this long
        stopped being a targeted comparison."""

        if not raw:
            return []
        cleaned: list[str] = []
        seen_lower: set[str] = set()
        for item in raw:
            name = str(item).strip()
            if not name or name.lower() in seen_lower:
                continue
            seen_lower.add(name.lower())
            cleaned.append(name)
        return cleaned[:6]

    # ------------------------------------------------------------------
    # P0-2: Comparison research-target recovery
    #
    # Runs once, right after the plan exists (Planner output or
    # fallback), before anything else touches it. Comparison-only: for
    # every other synthesis_type, a missing/thin research_targets list
    # is not this step's concern (Synthesizer's own posts-based fallback
    # already covers those, unaffected by this fix).
    # ------------------------------------------------------------------

    def _recover_research_targets_if_needed(
        self, plan: ExecutionPlan, user_goal: str
    ) -> ExecutionPlan:
        """Comparison's safety net when the Planner didn't resolve
        research_targets (missing, too few, or something that's
        actually a retrieval-source id rather than an entity name — see
        :meth:`_research_targets_look_valid`). Recovers from the
        ORIGINAL user query only — never from retrieved posts or
        sources, which is exactly the mixup this exists to prevent (a
        comparison's targets must come from user intent, not from
        whatever organizations the search happened to surface).

        If recovery also can't determine targets, ``research_targets``
        is left empty rather than guessed — see
        ``Synthesizer._labs_for_comparison``'s comparison-specific
        branch for how that renders honestly in the final answer
        instead of silently promoting retrieved organizations.
        """

        if plan.synthesis_type != "comparison":
            return plan
        if self._research_targets_look_valid(plan.research_targets):
            return plan

        logger.warning(
            "component=target_recovery synthesis_type=comparison "
            "research_targets=%r invalid/missing — attempting recovery "
            "from the original user query.",
            plan.research_targets,
        )
        recovered = self._recover_research_targets(user_goal)
        if self._research_targets_look_valid(recovered):
            logger.info(
                "component=target_recovery final_status=recovered targets=%r",
                recovered,
            )
            return self._with_research_targets(plan, recovered)

        logger.error(
            "component=target_recovery final_status=comparison_targets_undetermined "
            "— comparison targets could not be reliably determined from the "
            "user's request; the Synthesizer will say so rather than "
            "guessing from retrieved sources."
        )
        return self._with_research_targets(plan, [])

    @staticmethod
    def _research_targets_look_valid(targets: list[str]) -> bool:
        """False when there are fewer than 2 targets (the minimum for a
        comparison), or any of them normalizes to a retrieval-source id
        that isn't itself an organization (arxiv_cs_ai/semantic_scholar/
        openalex — see _NON_ORGANIZATION_SOURCE_IDS) — a strong signal
        the Planner confused "where to search" with "who to compare".
        Lab-specific source ids (openai/anthropic/...) are NOT
        disqualifying: "OpenAI" is a perfectly legitimate target even
        though it happens to share a slug with a source id."""

        if len(targets) < 2:
            return False
        normalized = {
            t.strip().lower().replace(" ", "_").replace("-", "_")
            for t in targets
            if t.strip()
        }
        return not (normalized & _NON_ORGANIZATION_SOURCE_IDS)

    @staticmethod
    def _with_research_targets(
        plan: ExecutionPlan, research_targets: list[str]
    ) -> ExecutionPlan:
        return ExecutionPlan(
            intent=plan.intent,
            source_plan=plan.source_plan,
            synthesis_type=plan.synthesis_type,
            user_profile_summary=plan.user_profile_summary,
            reasoning=plan.reasoning,
            research_targets=research_targets,
        )

    def _get_client(self) -> Any:
        """Lazy Gemini client for the one direct (non-ADK) call the
        Orchestrator makes itself — see _recover_research_targets."""

        if self._client is None:
            from google.genai import Client

            self._client = Client()
        return self._client

    def _recover_research_targets(self, user_goal: str) -> list[str]:
        """Lightweight entity extraction — NOT a new Agent: a single
        direct Gemini call (same pattern Analyst/Synthesizer/Critic
        already use for their own Gemini calls), reusing the shared
        retry policy. Reads only the user's own words
        (``prompts/comparison_target_recovery.md``); deliberately never
        given the retrieved posts or the source list.
        """

        import json as _json

        from google.genai import types as genai_types

        prompt = load_prompt("comparison_target_recovery", user_goal=user_goal)

        def _once() -> list[str]:
            client = self._get_client()
            response = client.models.generate_content(
                model=self.settings.model_flash,
                contents=prompt,
                config=genai_types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.1,
                ),
            )
            note_usage_from_gemini_response(response)
            payload = _json.loads(response.text)
            return list(payload.get("targets") or [])

        try:
            raw = call_with_retry(
                "target_recovery", self.settings.model_flash, _once
            )
        except Exception:
            logger.error(
                "component=target_recovery final_status=recovery_call_failed"
            )
            return []
        return self._coerce_research_targets(raw)

    # ------------------------------------------------------------------
    # HITL + guardrails + persist
    # ------------------------------------------------------------------

    def _guard_input(self, user_goal: str) -> str:
        with trace_span("orchestrator.guardrail_input"):
            result = run_guardrails(user_goal, "input")
            if not result.safe:
                raise ValueError(f"Input rejected by guardrails: {result.reason}")
            return result.sanitized_text or user_goal

    def _approve_plan(self, plan: ExecutionPlan) -> bool:
        with trace_span("orchestrator.hitl_plan"):
            return bool(self.on_plan_ready(plan))

    def _approve_synthesis(self, verified: VerifiedSynthesis) -> VerifiedSynthesis:
        with trace_span("orchestrator.hitl_synthesis"):
            edited = self.on_synthesis_ready(verified)
            return edited if edited is not None else verified

    def _guard_output(self, verified: VerifiedSynthesis) -> VerifiedSynthesis:
        """Output Guard — the final fail-safe before returning to the user.

        Two independent checks:

        1. PII redaction on the rendered text (unchanged).
        2. Citation completeness on the draft's claims — a *structural*
           check ("does this claim carry a citation at all"), not a
           semantic one (whether the citation actually supports the
           claim is the Critic's job, upstream of here).

        Deterministic only — see :meth:`_enforce_citation_completeness`:
        no LLM call happens in this method or anything it calls. The
        Critic's own revision loop (:meth:`_critic_loop`) already treats
        an uncited claim as "unsupported" and, at the default threshold,
        revises and re-verifies it before this method ever runs — an LLM
        "repair" here would be new factual content generated after the
        Critic already signed off, and never itself re-checked by the
        Critic. A gap surviving to here (max_revisions exhausted — already
        reflected in ``verified.final=False`` — or introduced by a human
        edit via ``on_synthesis_ready``) is handled by dropping the
        claim, never by asking an LLM to rewrite it.
        """

        with trace_span("orchestrator.guardrail_output"):
            text = self._render_for_guardrail(verified.draft)
            pii_result = run_guardrails(text, "output")
            if not pii_result.safe:
                raise ValueError(f"Output rejected by guardrails: {pii_result.reason}")

            return self._enforce_citation_completeness(verified)

    def _enforce_citation_completeness(
        self, verified: VerifiedSynthesis
    ) -> VerifiedSynthesis:
        draft = verified.draft
        missing = find_missing_citations(draft)
        if not missing:
            return verified

        total = self._count_claims(draft)
        ratio = len(missing) / total if total else 1.0
        if ratio > SYSTEMIC_CITATION_GAP_RATIO:
            raise ValueError(
                f"Output rejected by guardrails: {len(missing)}/{total} claims "
                "lack a citation — the core answer is systemically unsupported, "
                "not a single isolated claim."
            )

        logger.warning(
            "Output guard: %d/%d claim(s) missing citation — dropping them "
            "(deterministic fail-safe only; see _guard_output docstring for "
            "why no LLM rewrite happens here).",
            len(missing),
            total,
        )
        cleaned = strip_claims(draft, missing)
        return self._with_draft(verified, cleaned)

    @staticmethod
    def _count_claims(draft: DraftSynthesis) -> int:
        return sum(len(section.get("claims", []) or []) for section in draft.sections)

    @staticmethod
    def _with_draft(verified: VerifiedSynthesis, draft: DraftSynthesis) -> VerifiedSynthesis:
        """Swap in a new draft, keeping the rest of the VerifiedSynthesis
        (critic_report / revision_count / final) untouched — citation
        repair/cleanup at the output-guard stage is a distinct, later
        safety net from the Critic's own revision loop and shouldn't be
        conflated with it."""

        return VerifiedSynthesis(
            draft=draft,
            critic_report=verified.critic_report,
            revision_count=verified.revision_count,
            final=verified.final,
        )

    def _attach_citations(
        self, verified: VerifiedSynthesis, analyzed: list[AnalyzedPost]
    ) -> VerifiedSynthesis:
        """Attach the post_id -> {title, source, organization,
        published_at, url} map so a claim's bare ``[post_id]`` citation
        is resolvable to something a reader can act on, without a second
        storage lookup. Applied last (after HITL synthesis review and
        the output guard) rather than threaded through those steps: both
        can replace the draft/VerifiedSynthesis wholesale (see
        ``test_synthesis_callback_invoked_and_can_edit``), and building
        this map fresh from ``analyzed`` — which doesn't change after
        the research loop — is simpler and more robust than trying to
        keep a citations dict in sync through every intermediate
        transformation.
        """

        return VerifiedSynthesis(
            draft=verified.draft,
            critic_report=verified.critic_report,
            revision_count=verified.revision_count,
            final=verified.final,
            citations=self._build_citations(analyzed),
        )

    @staticmethod
    def _build_citations(analyzed: list[AnalyzedPost]) -> dict[str, dict[str, str]]:
        return {
            p.post_id: {
                "title": p.title,
                "source": p.source,
                "organization": p.organization,
                "published_at": p.published_at,
                "url": p.url,
            }
            for p in analyzed
        }

    def _persist(
        self,
        verified: VerifiedSynthesis,
        profile: UserProfile,
        plan: ExecutionPlan,
        user_goal: str,
        *,
        run_summary: dict[str, Any] | None = None,
    ) -> None:
        with trace_span("orchestrator.persist"):
            try:
                storage_db.save_synthesis(verified, self.settings)
            except Exception:
                # Storage failure must not kill the run — surface via logs only.
                logger.exception("Failed to persist VerifiedSynthesis")
            feedback_record = {
                "user_id": profile.user_id,
                "user_goal": user_goal,
                "intent": plan.intent,
                "synthesis_type": plan.synthesis_type,
                "revision_count": verified.revision_count,
                "final": verified.final,
                "logged_at": datetime.now(UTC).isoformat(),
            }
            # Eval-readiness fields (run_id, research_rounds, token/call
            # telemetry, ...) reuse this same flexible payload_json
            # column — see _build_run_summary — rather than a separate
            # table/schema.
            if run_summary:
                feedback_record.update(run_summary)
            try:
                storage_db.save_user_feedback(feedback_record, self.settings)
            except Exception:
                logger.exception("Failed to persist user_feedback row")

    def _build_run_summary(
        self,
        *,
        run_id: str,
        intent: str,
        research_rounds: int,
        used_sources: list[str],
        stop_reason: str,
        analyzed_posts: int,
        latency_ms: int,
    ) -> dict[str, Any]:
        """Assemble the per-run Eval summary: pipeline facts this class
        already tracks, plus real LLM-call telemetry aggregated from
        this run's own records (see ``agent_system.observability.
        tracing.summarize_llm_calls`` — every call through
        ``agent_system.llm_retry`` recorded one automatically). No
        dashboard, no cost estimate beyond what's already configured —
        just the JSON-serializable facts a future Eval Harness needs,
        persisted via the existing user_feedback row (see _persist).
        """

        from agent_system.observability.tracing import summarize_llm_calls

        llm_summary = summarize_llm_calls(run_id)
        return {
            "run_id": run_id,
            "research_rounds": research_rounds,
            "replan_triggered": research_rounds > 1,
            "used_sources": list(used_sources),
            "stop_reason": stop_reason,
            "analyzed_posts": analyzed_posts,
            "latency_ms": latency_ms,
            "agentic_replan_enabled": bool(self.settings.agentic_replan_enabled),
            **llm_summary,
        }

    # ------------------------------------------------------------------
    # Sub-agent calls (Python — typed schemas)
    # ------------------------------------------------------------------

    def _run_scout_initial(
        self, user_goal: str, profile: UserProfile, plan: ExecutionPlan
    ) -> tuple[list[RawPost], SourcePlan]:
        """Research round 1 — executes the Planner's own ``source_plan``
        exactly as given, the same way round 2+ (:meth:`_run_scout_targeted`)
        already executes the Evaluator's targeted selection unmodified.

        Deliberately does NOT call ``Scout.plan_sources()`` or union its
        heuristic defaults back in: that used to silently widen an
        explicit Planner choice (e.g. ``[anthropic, semantic_scholar]``)
        back out to Scout's own hardcoded default list, so "the Agent
        dynamically selected sources" wasn't actually true at
        execution time. ``plan.source_plan`` already carries its own
        fallback — ``_coerce_sources``/``_fallback_plan`` fall back to
        ``AVAILABLE_SOURCES`` when the Planner gave nothing valid — so
        there is nothing left for a second, Scout-level fallback to add.
        ``user_goal``/``profile`` are unused now that there's no Scout
        heuristic plan to build from them, but stay in the signature so
        callers don't need to change and a future per-goal Scout
        refinement has an obvious place to plug back in.
        """

        with trace_span("scout.plan_and_fetch"):
            return list(self.scout.fetch(plan.source_plan)), plan.source_plan

    def _run_scout_targeted(
        self,
        next_actions: list[dict[str, Any]],
        *,
        max_posts: int,
        time_window_days: int,
    ) -> tuple[list[RawPost], SourcePlan]:
        """Research round 2+ — a replan round driven by the Evaluator's
        ``next_actions``. Same fidelity contract as round 1 above: the
        Evaluator explicitly chose e.g. "Anthropic + Semantic Scholar"
        for a named gap, and that choice must reach Scout unmodified,
        never widened back out to a default source list.
        ``_coerce_sources`` still applies its existing
        fallback-to-default safety net for the degenerate case of an
        empty/invalid selection.
        """

        with trace_span("scout.targeted_fetch"):
            raw_sources = [
                s
                for action in next_actions
                for s in (action.get("preferred_sources") or [])
            ]
            sources = self._coerce_sources(raw_sources)
            raw_query = " ".join(str(a.get("query", "")) for a in next_actions).strip()
            keywords = Scout._extract_keywords(raw_query) if raw_query else []
            targeted_plan = SourcePlan(
                sources_to_query=sources,
                time_window_days=max(1, int(time_window_days)),
                filter_keywords=keywords,
                max_posts=max(1, int(max_posts)),
            )
            return list(self.scout.fetch(targeted_plan)), targeted_plan

    def _run_analyst(self, raw_posts: list[RawPost]) -> list[AnalyzedPost]:
        with trace_span("analyst.analyze_batch"):
            return list(self.analyst.analyze_batch(raw_posts))

    async def _run_research_loop(
        self,
        user_goal: str,
        intent: str,
        profile: UserProfile,
        plan: ExecutionPlan,
        session_id: str,
    ) -> tuple[list[AnalyzedPost], list[str], list[dict], str, int]:
        """Plan -> Act -> Observe -> Evaluate -> Replan/Stop.

        Every round (Scout fetch + Analyst analyze) is followed by an
        Evaluator call: an LLM judgment of whether the evidence gathered
        so far is enough for the goal, and if not, exactly what gap to
        target next (see ``prompts/orchestrator_evaluator.md``). The
        only thing the Evaluator does NOT get to decide is whether
        another round is allowed to happen at all —
        ``settings.max_research_rounds`` is a deterministic hard cap the
        workflow enforces regardless of what the Evaluator wants, so the
        loop can never run away. ``analyst.analyze_batch`` is only ever
        given posts not already seen this run (by post_id and by URL),
        so a replan round that re-surfaces something round 1 already
        found never re-analyzes it.

        Two independent deterministic hard caps, either of which
        overrides the Evaluator regardless of what it wants:
        ``settings.max_research_rounds`` (how many rounds) and
        ``settings.max_total_posts`` (total genuinely-new posts
        analyzed across every round combined — without this,
        max_research_rounds rounds could each spend their own
        ``max_posts`` and add up unboundedly). Budget is charged only
        for posts that survive dedup (by post_id and by canonical URL,
        see ``scout.url_utils.canonicalize_url``) — a re-surfaced
        duplicate costs nothing.

        Zero-evidence hard stop: after every round, if zero valid
        evidence has been analyzed across every round attempted so far
        (this round included), the loop stops right here — WITHOUT
        calling the Evaluator, which would have nothing to evaluate and
        risks the loop misreading an infrastructure failure as "search
        coverage insufficient -> replan". Two distinct reasons (see
        ``ResearchEvidenceUnavailableError``): ``"no_retrieved_evidence"``
        when Scout has never returned any post at all, vs.
        ``"analysis_unavailable"`` when posts were retrieved but every
        analysis attempt failed. A round that has SOME prior evidence
        (from an earlier round) never trips this, even if the current
        round's analysis fully fails — that prior evidence is kept and
        whether to keep researching stays governed by the normal
        Evaluator / hard-budget logic below, unchanged.

        Returns ``(all analyzed posts, cumulative sources actually
        queried, unresolved evidence gaps, stop_reason, research_rounds)``.
        Evidence gaps are non-empty only when the loop stopped at a hard
        cap without the Evaluator confirming sufficiency, and are
        threaded into the Synthesizer so the final answer names the
        limitation instead of writing around it (see
        ``Synthesizer._format_gap_notice``). ``stop_reason`` carries
        either a normal stop reason (evaluator-provided,
        "max_rounds_reached", "global_post_budget_exhausted",
        "agentic_replan_disabled") or one of the two zero-evidence
        reasons above; ``run_async``'s fail-safe only consults it when
        ``analyzed`` is empty, to raise
        ``ResearchEvidenceUnavailableError`` with the right reason.
        ``research_rounds`` is how many rounds actually ran (1 unless a
        replan happened) — used for the run-level Eval telemetry
        summary, see ``run_async``.
        """

        max_rounds = max(1, int(getattr(self.settings, "max_research_rounds", 1)))
        max_total_posts = max(0, int(getattr(self.settings, "max_total_posts", 30)))
        state = ResearchState(
            original_user_query=user_goal,
            intent=intent,
            research_goal=user_goal,
            max_rounds=max_rounds,
            research_targets=list(plan.research_targets),
        )

        all_analyzed: list[AnalyzedPost] = []
        seen_post_ids: set[str] = set()
        seen_urls: set[str] = set()
        sufficient = False

        while True:
            state.research_round += 1
            remaining_before_round = max(0, max_total_posts - state.total_new_posts)
            if state.research_round == 1:
                raw_posts, executed_plan = self._run_scout_initial(
                    user_goal, profile, plan
                )
            else:
                raw_posts, executed_plan = self._run_scout_targeted(
                    state.next_actions,
                    max_posts=min(plan.source_plan.max_posts, remaining_before_round),
                    time_window_days=plan.source_plan.time_window_days,
                )

            state.total_retrieved_posts += len(raw_posts)

            new_posts = [
                p
                for p in raw_posts
                if p.post_id not in seen_post_ids
                and canonicalize_url(p.url) not in seen_urls
            ]
            for p in new_posts:
                seen_post_ids.add(p.post_id)
                canonical = canonicalize_url(p.url)
                if canonical:
                    seen_urls.add(canonical)

            # Global budget is charged in *new unique* posts, uniformly
            # for every round including round 1 — this is the actual
            # enforcement; capping a round's requested max_posts above
            # is only an efficiency nicety (don't over-fetch when the
            # remaining budget is already known to be small).
            if len(new_posts) > remaining_before_round:
                logger.info(
                    "research_loop: round=%d truncating %d new post(s) to "
                    "remaining_budget=%d",
                    state.research_round,
                    len(new_posts),
                    remaining_before_round,
                )
                new_posts = new_posts[:remaining_before_round]

            new_analyzed = self._run_analyst(new_posts)
            all_analyzed.extend(new_analyzed)
            state.total_new_posts += len(new_posts)

            state.executed_queries.append(
                " ".join(executed_plan.filter_keywords) or "(default query)"
            )
            for s in executed_plan.sources_to_query:
                if s not in state.used_sources:
                    state.used_sources.append(s)

            analyzed_failed_this_round = len(new_posts) - len(new_analyzed)
            logger.info(
                "research_loop: round=%d/%d query=%r sources=%s new_posts=%d "
                "analyzed=%d analysis_failed=%d total_posts=%d/%d",
                state.research_round,
                max_rounds,
                state.executed_queries[-1],
                executed_plan.sources_to_query,
                len(new_posts),
                len(new_analyzed),
                analyzed_failed_this_round,
                state.total_new_posts,
                max_total_posts,
            )

            if not all_analyzed:
                # Zero valid evidence across every round attempted so far
                # (this one included) — hard stop before the Evaluator,
                # which would have nothing to evaluate. See this method's
                # docstring and ResearchEvidenceUnavailableError.
                if state.total_retrieved_posts == 0:
                    state.stop_reason = "no_retrieved_evidence"
                    logger.error(
                        "component=research_loop final_status=no_retrieved_evidence "
                        "— Scout did not retrieve any source for this goal; "
                        "stopping before the Evaluator/Synthesizer."
                    )
                else:
                    state.stop_reason = "analysis_unavailable"
                    logger.error(
                        "component=research_loop final_status=analysis_unavailable "
                        "— %d source(s) were retrieved but no article could "
                        "be analyzed; stopping before the Evaluator/"
                        "Synthesizer rather than treating this as a "
                        "search-coverage gap.",
                        state.total_retrieved_posts,
                    )
                return (
                    all_analyzed,
                    state.used_sources,
                    [],
                    state.stop_reason,
                    state.research_round,
                )

            if not self.settings.agentic_replan_enabled:
                # Fixed Workflow baseline (Eval ablation): the Evaluator
                # is never consulted at all — not called-and-ignored,
                # literally zero calls — so a Workflow-vs-Agent
                # benchmark isn't comparing "Agent capped at round 1"
                # against itself under a different name. No replan-
                # specific evidence gap is generated either.
                state.stop_reason = "agentic_replan_disabled"
                logger.info(
                    "research_loop: round=%d agentic_replan_enabled=False "
                    "— fixed workflow baseline, stopping after round 1 "
                    "without consulting the Evaluator.",
                    state.research_round,
                )
                return (
                    all_analyzed,
                    state.used_sources,
                    [],
                    state.stop_reason,
                    state.research_round,
                )

            evaluation = await self._evaluate_research_state(
                state, all_analyzed, profile, session_id
            )
            state.evidence_gaps = evaluation["evidence_gaps"]
            state.next_actions = evaluation["next_actions"]
            sufficient = evaluation["is_sufficient"]
            at_round_cap = state.research_round >= max_rounds
            budget_exhausted = state.total_new_posts >= max_total_posts
            hard_stop = at_round_cap or budget_exhausted

            logger.info(
                "research_loop: round=%d evaluator_wants=%s is_sufficient=%s "
                "gaps=%d at_round_cap=%s budget_exhausted=%s stop_reason=%r",
                state.research_round,
                "continue" if evaluation["continue_research"] else "stop",
                sufficient,
                len(state.evidence_gaps),
                at_round_cap,
                budget_exhausted,
                evaluation["stop_reason"],
            )

            if hard_stop:
                if sufficient:
                    state.stop_reason = evaluation["stop_reason"]
                elif budget_exhausted:
                    state.stop_reason = "global_post_budget_exhausted"
                else:
                    state.stop_reason = "max_rounds_reached"
                break
            if not evaluation["continue_research"]:
                state.stop_reason = evaluation["stop_reason"] or "evaluator_sufficient"
                break

        unresolved_gaps = [] if sufficient else state.evidence_gaps
        return (
            all_analyzed,
            state.used_sources,
            unresolved_gaps,
            state.stop_reason,
            state.research_round,
        )

    async def _evaluate_research_state(
        self,
        state: ResearchState,
        analyzed: list[AnalyzedPost],
        profile: UserProfile,
        session_id: str,
    ) -> dict[str, Any]:
        with trace_span("research_loop.evaluate"):
            user_text = self._build_evaluator_user_text(state, analyzed)
            try:
                payload = await self._invoke_agent(
                    self.evaluator_agent,
                    user_text,
                    profile,
                    session_id,
                    EVALUATOR_OUTPUT_KEY,
                    component="evaluator",
                )
            except Exception:
                # Unable to judge whether to continue -> the conservative
                # choice is to stop and hand whatever evidence exists so
                # far to the Synthesizer, not to guess and keep searching
                # blindly with no signal for what to look for next.
                logger.error(
                    "component=evaluator final_status=evaluator_unavailable "
                    "— stopping the research loop; current evidence "
                    "continues to the Synthesizer."
                )
                return {
                    "is_sufficient": False,
                    "covered_dimensions": [],
                    "evidence_gaps": [],
                    "continue_research": False,
                    "next_actions": [],
                    "stop_reason": "evaluator_unavailable",
                }
            return self._normalize_evaluation_payload(payload)

    @staticmethod
    def _build_evaluator_user_text(
        state: ResearchState, analyzed: list[AnalyzedPost]
    ) -> str:
        lines = [
            f"Research goal: {state.research_goal}",
            f"Synthesis intent: {state.intent}",
            f"Research targets: {', '.join(state.research_targets) or '(none named — general goal)'}",
            f"Research round: {state.research_round} of max {state.max_rounds}",
            f"Sources used so far: {', '.join(state.used_sources) or '(none)'}",
            "Queries executed so far: "
            f"{' | '.join(state.executed_queries) or '(none)'}",
            "",
            "Evidence gathered so far, grouped by source:",
        ]
        by_source: dict[str, list[AnalyzedPost]] = {}
        for post in analyzed:
            by_source.setdefault(post.source or "unknown", []).append(post)
        if not by_source:
            lines.append("(no evidence gathered yet)")
        for source, posts in sorted(by_source.items()):
            org = posts[0].organization or source
            lines.append(f"- {org} [source={source}] — {len(posts)} post(s):")
            for p in posts[:5]:
                lines.append(f'    * [{p.post_id}] "{p.title}": {p.key_claim}')
            if len(posts) > 5:
                lines.append(f"    * ...and {len(posts) - 5} more.")
        return "\n".join(lines)

    def _normalize_evaluation_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Coerce the Evaluator's raw JSON into a safe shape — same
        defensive-parsing role as :meth:`_payload_to_plan` plays for the
        Planner: an unrecognized source is dropped, an actionless
        "continue" is downgraded to "stop", never trusted verbatim."""

        raw_actions = payload.get("next_actions") or []
        next_actions: list[dict[str, Any]] = []
        for a in raw_actions:
            if not isinstance(a, dict):
                continue
            query = str(a.get("query", "")).strip()
            if not query:
                continue
            preferred = [
                str(s).strip().lower().replace(" ", "_")
                for s in (a.get("preferred_sources") or [])
                if str(s).strip()
            ]
            preferred = [s for s in preferred if s in AVAILABLE_SOURCES]
            next_actions.append(
                {
                    "query": query,
                    "preferred_sources": preferred,
                    "reason": str(a.get("reason", "")).strip(),
                }
            )

        continue_research = bool(payload.get("continue_research", False)) and bool(
            next_actions
        )
        is_sufficient = bool(payload.get("is_sufficient", not continue_research))
        evidence_gaps = [
            g for g in (payload.get("evidence_gaps") or []) if isinstance(g, dict)
        ]
        covered = [
            d for d in (payload.get("covered_dimensions") or []) if isinstance(d, dict)
        ]
        stop_reason = str(payload.get("stop_reason", "")).strip()

        return {
            "is_sufficient": is_sufficient,
            "covered_dimensions": covered,
            "evidence_gaps": evidence_gaps,
            "continue_research": continue_research,
            "next_actions": next_actions,
            "stop_reason": stop_reason,
        }

    def _run_synthesizer(
        self,
        analyzed: list[AnalyzedPost],
        profile: UserProfile,
        synthesis_type: str,
        *,
        research_targets: list[str] | None = None,
        evidence_gaps: list[dict] | None = None,
    ) -> DraftSynthesis:
        with trace_span("synthesizer.synthesize"):
            return self.synthesizer.synthesize(
                analyzed,
                profile,
                synthesis_type,
                research_targets=research_targets,
                evidence_gaps=evidence_gaps,
            )

    def _critic_loop(
        self, draft: DraftSynthesis, analyzed: list[AnalyzedPost]
    ) -> VerifiedSynthesis:
        with trace_span("critic.loop"):
            report = self.critic.review(draft, analyzed)
            revisions = 0
            cap = max(0, int(self.settings.max_revisions))
            while report.revision_needed and revisions < cap:
                with trace_span(f"synthesizer.revise.{revisions + 1}"):
                    draft = self.synthesizer.revise(draft, report.revision_notes)
                report = self.critic.review(draft, analyzed)
                revisions += 1
            final = not report.revision_needed
            if not final:
                logger.warning(
                    "Critic still flagged %d unsupported claim(s) after %d revision(s); "
                    "surfacing to user with final=False.",
                    report.num_unsupported,
                    revisions,
                )
            return VerifiedSynthesis(
                draft=draft,
                critic_report=report,
                revision_count=revisions,
                final=final,
            )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _summarise_profile(self, profile: UserProfile) -> str:
        parts = [
            f"role_target={profile.role_target}",
            f"seniority={profile.seniority}",
            f"interests={', '.join(profile.interests)}",
        ]
        if profile.reading_history:
            parts.append(f"posts_read={len(profile.reading_history)}")
        return "; ".join(parts)

    def _coerce_sources(self, raw: Any) -> list[str]:
        if not raw:
            return list(AVAILABLE_SOURCES[:3])
        cleaned: list[str] = []
        for item in raw:
            s = str(item).strip().lower().replace(" ", "_")
            if s and s in AVAILABLE_SOURCES and s not in cleaned:
                cleaned.append(s)
        return cleaned or list(AVAILABLE_SOURCES[:3])

    @staticmethod
    def _coerce_int(
        value: Any, *, default: int, lo: int = 0, hi: int | None = None
    ) -> int:
        try:
            n = int(value) if value is not None else default
        except (TypeError, ValueError):
            n = default
        if n < lo:
            n = lo
        if hi is not None and n > hi:
            n = hi
        return n

    def _render_for_guardrail(self, draft: DraftSynthesis) -> str:
        chunks: list[str] = [draft.title or ""]
        for section in draft.sections or []:
            if not isinstance(section, dict):
                continue
            heading = section.get("heading", "")
            prose = section.get("prose", "")
            chunks.append(f"## {heading}\n{prose}")
        return "\n\n".join(chunks).strip()


def _parse_json_or_raise(text: str) -> dict[str, Any]:
    """Decode a JSON object from raw LLM text, tolerating fenced markdown."""

    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned[:4].lower() == "json":
            cleaned = cleaned[4:]
        cleaned = cleaned.strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(cleaned[start : end + 1])
        raise
