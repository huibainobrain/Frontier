"""ChromaDB persistent vector store for AnalyzedPost embeddings.

The Synthesizer's RAG step calls
:func:`search_similar` to retrieve semantically related prior
:class:`AnalyzedPost` records from the corpus. The Analyst calls
:func:`add_analyzed_post` after every successful analysis.

Embedding strategy:

* If the caller passes ``embedding=...`` (a precomputed vector) it is
  stored directly and Chroma never invokes its embedding function.
  This is the path the Analyst uses once it has computed the embedding
  from Gemini's ``text-embedding-004`` model.
* If no embedding is passed, Chroma's default embedding function is
  used. Tests should always pass embeddings explicitly so they never
  hit the network.
"""

from __future__ import annotations

import logging

from agent_system.config import Settings, get_settings
from agent_system.schemas import AnalyzedPost
from agent_system.storage import db as _db

logger = logging.getLogger(__name__)

COLLECTION_NAME = "analyzed_posts_embeddings"


def _settings(settings: Settings | None = None) -> Settings:
    return settings if settings is not None else get_settings()


def _client(settings: Settings | None = None):
    """Construct a persistent Chroma client. Imported lazily so test
    environments without chromadb installed can still import this module."""

    import chromadb  # type: ignore

    s = _settings(settings)
    s.chroma_dir.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(s.chroma_dir))


def get_collection(settings: Settings | None = None):
    return _client(settings).get_or_create_collection(name=COLLECTION_NAME)


def document_for(post: AnalyzedPost) -> str:
    """Text representation of an AnalyzedPost used for embedding + storage.

    Public (not ``_``-prefixed) because the Analyst needs the exact same
    text when it computes the embedding to pass into :func:`add_analyzed_post`
    — the embedding and the stored document must describe the same content.
    """

    return f"{post.key_claim}\n\n{post.practitioner_takeaway}\n\n" + " ".join(
        post.concepts_introduced
    )


def _metadata_for(post: AnalyzedPost) -> dict:
    return {
        "post_id": post.post_id,
        "category": post.category,
        "confidence": float(post.confidence),
        "analyzed_at": post.analyzed_at,
        # Provenance, so a future filtered query (e.g. "only Anthropic
        # hits") doesn't need a round-trip to SQLite just to check
        # source/org. Chroma metadata values must be str/int/float/bool
        # (no None), hence the "" fallbacks.
        "title": post.title or "",
        "source": post.source or "",
        "organization": post.organization or "",
        "published_at": post.published_at or "",
        # Not itself the enforcement mechanism — search_similar hydrates
        # every hit through storage.db.get_analyzed_post, which already
        # filters stale rows (see its docstring). Stored here too only
        # so a future Chroma-side filtered query doesn't need a second
        # round-trip to check.
        "schema_version": int(post.schema_version),
    }


def add_analyzed_post(
    post: AnalyzedPost,
    embedding: list[float] | None = None,
    settings: Settings | None = None,
) -> None:
    """Insert (or upsert) an AnalyzedPost into the vector store.

    Pass ``embedding=`` to store a precomputed vector and skip Chroma's
    default embedding function entirely (recommended in production —
    the Analyst already has the vector at hand).
    """

    coll = get_collection(settings)
    document = document_for(post)
    metadata = _metadata_for(post)
    if embedding is not None:
        coll.upsert(
            ids=[post.post_id],
            embeddings=[embedding],
            documents=[document],
            metadatas=[metadata],
        )
    else:
        coll.upsert(
            ids=[post.post_id],
            documents=[document],
            metadatas=[metadata],
        )


def search_similar(
    query: str,
    k: int = 5,
    embedding: list[float] | None = None,
    settings: Settings | None = None,
) -> list[AnalyzedPost]:
    """Return the top-k AnalyzedPosts most similar to ``query``.

    Hydrates each hit from the SQLite store so callers receive fully
    typed dataclass instances. Misses (e.g. a vector hit with no
    matching SQL row) are silently skipped.
    """

    coll = get_collection(settings)
    if embedding is not None:
        result = coll.query(query_embeddings=[embedding], n_results=k)
    else:
        result = coll.query(query_texts=[query], n_results=k)
    ids: list[str] = []
    raw_ids = result.get("ids") if isinstance(result, dict) else None
    if raw_ids and len(raw_ids) > 0:
        ids = list(raw_ids[0])
    posts: list[AnalyzedPost] = []
    for pid in ids:
        post = _db.get_analyzed_post(pid, settings=settings)
        if post is not None:
            posts.append(post)
    return posts


def rebuild_index(settings: Settings | None = None) -> None:
    """Drop and recreate the collection — useful after schema changes."""

    client = _client(settings)
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        logger.exception("delete_collection failed (ok if collection didn't exist)")
    client.get_or_create_collection(COLLECTION_NAME)
