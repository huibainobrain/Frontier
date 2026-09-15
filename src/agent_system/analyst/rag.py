"""Retrieval-augmented generation helpers for the Analyst.

Wraps ``storage/vectors.py`` and ``storage/db.py`` to retrieve top-k
similar past articles and build a context string for the Analyst prompt,
and to persist newly analyzed articles back into that same corpus so
later retrievals can find them.

Embeddings are computed via Gemini's ``text-embedding-004`` model.
"""

from __future__ import annotations

import logging

from agent_system.config import Settings, get_settings
from agent_system.schemas import AnalyzedPost

logger = logging.getLogger(__name__)


def compute_embedding(text: str, settings: Settings | None = None) -> list[float]:
    """Compute a text embedding vector using Gemini's embedding model."""

    from google.genai import Client

    s = settings if settings is not None else get_settings()
    client = Client()
    response = client.models.embed_content(
        model=s.embedding_model,
        contents=[text],
    )
    return list(response.embeddings[0].values)


def retrieve_context(
    query: str,
    k: int = 5,
    settings: Settings | None = None,
) -> list[dict[str, str]]:
    """Find the top-k most relevant past articles for *query*.

    1. Compute an embedding for *query* via Gemini.
    2. Search the ChromaDB vector store for similar :class:`AnalyzedPost` records.
    3. Return the research-relevant fields of each hit directly — no
       second SQLite lookup needed, since ``AnalyzedPost`` itself now
       carries its own provenance (title/source/published_at/url).

    Returns the analysis (key claim, takeaway, concepts, evidence),
    not raw article text — this is meant as prior-research memory the
    Analyst can reason over, not another copy of the source content.
    """

    from agent_system.storage.vectors import search_similar

    s = settings if settings is not None else get_settings()
    try:
        embedding = compute_embedding(query, s)
    except Exception:
        return []

    similar = search_similar(query, k=k, embedding=embedding, settings=s)

    results: list[dict[str, str]] = []
    for post in similar:
        results.append(
            {
                "post_id": post.post_id,
                "title": post.title,
                "source": post.source,
                "organization": post.organization,
                "published_at": post.published_at,
                "category": post.category,
                "key_claim": post.key_claim,
                "practitioner_takeaway": post.practitioner_takeaway,
                "concepts": ", ".join(post.concepts_introduced),
                "evidence": " ".join(post.evidence_quotes[:2]),
                "confidence": str(post.confidence),
            }
        )
    return results


def build_rag_prompt(context_chunks: list[dict[str, str]]) -> str:
    """Format retrieved context into a string block for prompt injection.

    Returns empty string when *context_chunks* is empty so the Analyst
    prompt degrades gracefully. Renders the actual analysis (claim,
    takeaway, concepts, evidence) rather than a raw content excerpt, so
    this "long-term memory" is useful for cross-referencing rather than
    just re-showing the first few hundred characters of prior articles.
    """

    if not context_chunks:
        return ""
    parts = ["Previously analyzed related articles for context:\n"]
    for i, chunk in enumerate(context_chunks, 1):
        title = chunk.get("title") or "Untitled"
        org = chunk.get("organization") or chunk.get("source") or "unknown source"
        date = chunk.get("published_at") or "date unknown"
        claim = chunk.get("key_claim", "")
        takeaway = chunk.get("practitioner_takeaway", "")
        concepts = chunk.get("concepts", "")
        parts.append(
            f"{i}. [{chunk.get('post_id', '')}] \"{title}\" — {org} ({date})\n"
            f"   Key claim: {claim}\n"
            f"   Takeaway: {takeaway}\n"
            f"   Concepts: {concepts}\n"
        )
    return "\n".join(parts)


def store_analysis(post: AnalyzedPost, settings: Settings | None = None) -> None:
    """Persist a freshly analyzed post so future :func:`retrieve_context`
    calls (and the Synthesizer's own future corpus search) can find it.

    Two writes happen here:

    1. SQLite — the row of record, hydrated back out by ``search_similar``.
    2. ChromaDB — the embedding, computed over the same text
       (:func:`agent_system.storage.vectors.document_for`) that will later
       be matched against a query embedding.

    Without this, ``retrieve_context`` always returns ``[]``: nothing is
    ever added to the corpus it searches. Both writes are best-effort —
    a storage or embedding failure is logged, not raised, so a hiccup
    here never invalidates the analysis that already succeeded.
    """

    from agent_system.storage import db as storage_db
    from agent_system.storage.vectors import add_analyzed_post, document_for

    s = settings if settings is not None else get_settings()

    try:
        storage_db.save_analyzed_post(post, settings=s)
    except Exception:
        logger.exception("Failed to save AnalyzedPost %s to SQLite", post.post_id)

    try:
        embedding = compute_embedding(document_for(post), s)
        add_analyzed_post(post, embedding=embedding, settings=s)
    except Exception:
        logger.exception(
            "Failed to index AnalyzedPost %s in the vector store "
            "(RAG retrieval will not see it)",
            post.post_id,
        )