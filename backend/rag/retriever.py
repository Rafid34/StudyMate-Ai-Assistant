"""Query-time retrieval for StudyMate's RAG pipeline.

Public API
----------
retrieve(query, session_id="default", k=5) -> list[RetrievedChunk]
    Embed *query* using the same ``gemini-embedding-001`` model (with the
    same ``output_dimensionality=768``) used at ingest time, then run a
    cosine-similarity top-k search against the per-session ChromaDB
    collection created by ``ingest.py``.

    Returns a list of :class:`RetrievedChunk` objects ordered by descending
    similarity score, each carrying the chunk text and the source filename
    the chunk was extracted from.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from backend.config import CHROMA_DB_PATH, GEMINI_API_KEY

logger = logging.getLogger(__name__)

# Collection naming mirrors ingest.py exactly so retrieval always targets
# the same collection that was written during ingestion.
_COLLECTION_PREFIX = "studymate_"
_MAX_COLLECTION_LEN = 63
_INVALID_CHAR_RE = re.compile(r"[^a-zA-Z0-9_-]")


def _collection_name(session_id: str) -> str:
    safe_id = _INVALID_CHAR_RE.sub("_", session_id)
    name = f"{_COLLECTION_PREFIX}{safe_id}"
    return name[:_MAX_COLLECTION_LEN]


@dataclass
class RetrievedChunk:
    """A single chunk returned from the vector store.

    Attributes
    ----------
    text:
        Raw chunk text as stored during ingestion.
    source:
        Filename of the source document (e.g. ``"lecture_notes.pdf"``).
    score:
        Cosine relevance score in [0, 1] — higher means more similar.
        ``None`` when the underlying store does not expose a score.
    """

    text: str
    source: str
    score: float | None = None


def retrieve(
    query: str,
    session_id: str = "default",
    k: int = 5,
) -> list[RetrievedChunk]:
    """Return the top-k most-relevant chunks for *query*.

    Parameters
    ----------
    query:
        The student's question or search string.
    session_id:
        Identifies which ChromaDB collection to search — must match the
        ``session_id`` used when the documents were ingested.
    k:
        Number of chunks to retrieve. Defaults to 5 per the project spec.

    Returns
    -------
    list[RetrievedChunk]
        Ordered by descending similarity (best match first).
        Returns an empty list if the collection is empty or does not exist.

    Raises
    ------
    ValueError
        If *query* is an empty string.
    """
    if not query or not query.strip():
        raise ValueError("query must be a non-empty string")

    collection = _collection_name(session_id)
    logger.info(
        "Retrieving top-%d chunks for query %r from collection '%s'",
        k,
        query,
        collection,
    )

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=GEMINI_API_KEY,
        output_dimensionality=768,
    )

    # Open the existing persisted collection — does NOT create a new one.
    vector_store = Chroma(
        collection_name=collection,
        embedding_function=embeddings,
        persist_directory=CHROMA_DB_PATH,
    )

    # Retrieve with relevance scores so downstream agents can apply thresholds.
    try:
        results = vector_store.similarity_search_with_relevance_scores(query, k=k)
    except Exception as exc:
        # ChromaDB raises if the collection has zero documents; treat as empty.
        logger.warning(
            "Similarity search failed for collection '%s': %s", collection, exc
        )
        return []

    chunks: list[RetrievedChunk] = []
    for doc, score in results:
        source = doc.metadata.get("source", "unknown")
        chunks.append(RetrievedChunk(text=doc.page_content, source=source, score=score))
        logger.debug(
            "  score=%.4f  source=%r  snippet=%r",
            score,
            source,
            doc.page_content[:80],
        )

    logger.info("Retrieved %d chunk(s).", len(chunks))
    return chunks
