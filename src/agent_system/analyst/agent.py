"""Analyst agent — structured extraction from research articles.

Takes a :class:`RawPost` (produced by the Scout) and returns an
:class:`AnalyzedPost` with category, key claim, practitioner takeaway,
concepts introduced, confidence, and evidence quotes.

Uses Gemini Pro via the Google GenAI SDK with JSON response mode for
reliable structured output. Optionally prepends RAG context from
previously analyzed articles for cross-referencing, and persists every
successful analysis back into that same corpus (see
:func:`agent_system.analyst.rag.store_analysis`) so later posts can
retrieve it.

Two guardrails around the Gemini call itself:

* **Cache-first** — a post is only ever billed to Gemini once. If
  ``post.post_id`` already has a saved :class:`AnalyzedPost`, ``analyze``
  returns that instead of re-analyzing.
* **Content guardrail** — ``post.content`` is checked against the
  ``"content"`` guardrail stage (prompt-injection patterns) *before* it
  is ever sent to Gemini. A post that fails raises
  :class:`ContentBlockedError`; :meth:`Analyst.analyze_batch` catches
  that and simply omits the post rather than failing the whole batch.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from agent_system.config import Settings, get_settings
from agent_system.guardrails import run_guardrails
from agent_system.llm_retry import call_with_retry, note_usage_from_gemini_response
from agent_system.prompts import load_prompt
from agent_system.schemas import AnalyzedPost, RawPost, now_iso
from agent_system.scout.agent import organization_for_source

logger = logging.getLogger(__name__)


class ContentBlockedError(Exception):
    """Raised when a post's content fails the ``"content"`` guardrail stage.

    The post is never sent to Gemini when this is raised. Carries
    ``post_id`` and ``reason`` so a caller (or the orchestrator's logs)
    can say which post was withheld and why.
    """

    def __init__(self, post_id: str, reason: str) -> None:
        self.post_id = post_id
        self.reason = reason
        super().__init__(f"post_id={post_id!r} blocked by content guardrail: {reason}")


class Analyst:
    """Gemini-backed Analyst that extracts structured analysis from articles."""

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

    def analyze(self, post: RawPost, *, use_cache: bool = True) -> AnalyzedPost:
        """Analyze a single article and return structured extraction.

        Args:
            post: the article to analyze.
            use_cache: when true (default), reuse a previously persisted
                ``AnalyzedPost`` for ``post.post_id`` instead of paying for
                another Gemini call. Pass ``False`` to force a fresh
                analysis (e.g. after fixing a bad prompt).

        Raises:
            ContentBlockedError: if ``post.content`` fails the ``"content"``
                guardrail stage. Gemini is never called in that case.
        """

        if use_cache:
            cached = self._get_cached(post.post_id)
            if cached is not None:
                logger.info("Analyst cache hit for post_id=%s", post.post_id)
                return cached

        guard = run_guardrails(post.content, "content")
        if not guard.safe:
            logger.warning(
                "Content guardrail blocked post_id=%s before analysis: %s",
                post.post_id,
                guard.reason,
            )
            raise ContentBlockedError(post.post_id, guard.reason)

        from agent_system.analyst.rag import build_rag_prompt, retrieve_context, store_analysis

        context_chunks = retrieve_context(
            f"{post.title} {post.content[:500]}", k=3, settings=self.settings
        )
        rag_context = build_rag_prompt(context_chunks)

        prompt = load_prompt(
            "analyst",
            title=post.title,
            source=post.source,
            content_type=post.content_type,
            content=post.content,
            rag_context=rag_context,
        )

        result = self._call_gemini(prompt)

        analyzed = AnalyzedPost(
            post_id=post.post_id,
            category=result.get("category", "capability"),
            key_claim=result.get("key_claim", ""),
            practitioner_takeaway=result.get("practitioner_takeaway", ""),
            ships_in_product=None,
            concepts_introduced=result.get("concepts_introduced", []),
            relation_to_prior=[],
            confidence=float(result.get("confidence", 0.5)),
            evidence_quotes=result.get("evidence_quotes", []),
            analyzed_at=now_iso(),
            # Provenance copied verbatim from the RawPost — never
            # re-derived from the LLM response, so it can't drift from
            # what Scout actually fetched.
            title=post.title,
            source=post.source,
            url=post.url,
            authors=list(post.authors),
            published_at=post.published_at,
            content_type=post.content_type,
            organization=organization_for_source(post.source),
        )
        store_analysis(analyzed, self.settings)
        return analyzed

    def analyze_batch(self, posts: list[RawPost]) -> list[AnalyzedPost]:
        """Analyze multiple articles sequentially.

        Failure isolation, per article:

        * A post blocked by the content guardrail is logged and left
          out of the returned list — it never reaches the Synthesizer.
        * A post whose Gemini call fails even after retrying transient
          errors (``agent_system.llm_retry``) is logged as
          ``analysis_failed`` and skipped — one bad article must never
          take down the rest of the batch, since Analyst processes
          articles one at a time and has nothing structurally in common
          across them.

        Only if every single post in the batch fails is that escalated
        to an ``insufficient_analyzed_evidence`` log line — still not an
        exception: an empty result list already correctly propagates
        "no evidence this round" to the research loop's Evaluator and
        to the Synthesizer, which can act on that (see
        ``Synthesizer._format_gap_notice``) without this method itself
        needing to decide what "not enough" means.
        """

        results: list[AnalyzedPost] = []
        for post in posts:
            try:
                results.append(self.analyze(post))
            except ContentBlockedError as exc:
                logger.warning("Skipping post_id=%s: %s", exc.post_id, exc.reason)
            except Exception:
                logger.exception(
                    "component=analyst post_id=%s final_status=analysis_failed "
                    "— skipping this article, other articles continue.",
                    post.post_id,
                )

        if posts and not results:
            logger.error(
                "component=analyst final_status=insufficient_analyzed_evidence "
                "— all %d article(s) in this batch failed to analyze.",
                len(posts),
            )
        return results

    def _get_cached(self, post_id: str) -> AnalyzedPost | None:
        """Look up a previously persisted AnalyzedPost for *post_id*, if any."""

        from agent_system.storage import db as storage_db

        try:
            return storage_db.get_analyzed_post(post_id, settings=self.settings)
        except Exception:
            logger.exception(
                "Cache lookup failed for post_id=%s; analyzing fresh", post_id
            )
            return None

    # ------------------------------------------------------------------
    # Gemini invocation
    # ------------------------------------------------------------------

    def _call_gemini(self, prompt: str) -> dict[str, Any]:
        """Call Gemini Pro with JSON response mode and return parsed dict.

        Retries transient infra errors via the shared
        ``agent_system.llm_retry`` policy; the final exception (if every
        attempt fails, or the first failure isn't retryable) propagates
        to ``analyze()`` and from there to ``analyze_batch()``'s
        per-article failure isolation (see its docstring) — this method
        itself doesn't decide what "exhausted" means for the batch.
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

        return call_with_retry("analyst", self.settings.model_pro, _once)