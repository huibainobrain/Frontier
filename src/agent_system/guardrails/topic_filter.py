"""Topic / scope guardrail — runs on the *input* stage.

Judges whether a user's goal is within FrontierLit's research scope
(frontier AI lab publications, papers, literature monitoring and
synthesis), not whether the final rendered output happens to contain
specific keywords — that check used to live on the output stage and
over-rejected legitimate on-topic requests (it flagged this project's
own happy-path test fixture as "off-topic"); see
``agent_system.guardrails.__init__.run_guardrails`` for how the two
stages are wired.

Default-permissive by design: an ambiguous or terse goal ("what's new
this week?") passes. Only two things reject:

1. A small set of explicitly unsafe patterns (unchanged from before).
2. A small set of explicit *different-domain* task requests (write me a
   poem, help with my taxes, ...) — a genuine signal the user is trying
   to repurpose the agent, not just a goal that happens not to mention
   "AI" by name.

Anything else — including a goal that matches neither list — passes.
Over-rejecting a plausible-but-terse research request is worse than
occasionally letting one truly off-topic request continue to a
downstream stage that can still catch it.
"""

from __future__ import annotations

from agent_system.guardrails.result import GuardrailResult

_UNSAFE_PATTERNS = [
    "steal an api key",
    "steal api key",
    "phishing",
    "write a phishing email",
    "malware",
    "exfiltrate",
    "bypass safety",
    "hack into",
    "credential theft",
]

# On-topic signal — if any of these appear, that's a confident pass.
_DEFAULT_ALLOWED_TOPICS = [
    "ai",
    "artificial intelligence",
    "frontier",
    "literature",
    "research",
    "paper",
    "blog",
    "anthropic",
    "openai",
    "deepmind",
    "google research",
    "meta ai",
    "hugging face",
    "arxiv",
    "embodied",
    "robotics",
    "agent",
    "rag",
    "model",
    "evaluation",
]

# Off-topic signal — an explicit ask for a different kind of task
# entirely. Not exhaustive on purpose: a miss here just falls through to
# the default-permissive "pass" below, which is the intended safe side.
_CLEARLY_OFF_TOPIC_PATTERNS = [
    "write me a poem",
    "write a poem",
    "tell me a joke",
    "write a joke",
    "help me file my taxes",
    "plan my vacation",
    "plan a trip",
    "write a love letter",
    "give me a recipe",
    "translate this to",
    "what's the weather",
    "play a game",
]


def is_on_topic(query: str, allowed_topics: list[str] | None = None) -> GuardrailResult:
    lowered = query.lower().strip()

    for pattern in _UNSAFE_PATTERNS:
        if pattern in lowered:
            return GuardrailResult(
                safe=False,
                reason=f"Unsafe request detected: {pattern}",
                sanitized_text=None,
            )

    if not lowered:
        # Empty goal isn't this guard's problem to solve — let it through
        # to whatever downstream validation handles empty input.
        return GuardrailResult(safe=True, sanitized_text=query)

    topics = allowed_topics or _DEFAULT_ALLOWED_TOPICS
    if any(topic.lower() in lowered for topic in topics):
        return GuardrailResult(safe=True, sanitized_text=query)

    for pattern in _CLEARLY_OFF_TOPIC_PATTERNS:
        if pattern in lowered:
            return GuardrailResult(
                safe=False,
                reason=(
                    "This looks like a different kind of task than frontier AI "
                    "literature monitoring — try asking about a lab, paper, "
                    "concept, or research topic instead."
                ),
                sanitized_text=None,
            )

    # Ambiguous: no topic keyword hit, but no off-topic pattern either.
    # Default-permissive — pass rather than risk rejecting a legitimate
    # terse request.
    return GuardrailResult(safe=True, sanitized_text=query)
