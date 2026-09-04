"""RAG Agent — retrieves relevant chunks from ChromaDB for a given query.

Spec reference: Section 6 (Agent Responsibilities — RAG Agent).

The agent embeds *query* using the same ``text-embedding-004`` model used at
ingest time, then runs a cosine-similarity top-k search against the
per-session ChromaDB collection created by ``rag/ingest.py``.  Results are
returned as :class:`~backend.agents.synthesis_agent.RagChunk` objects, which
the Synthesis Agent consumes directly.

LangSmith tracing
-----------------
``run_rag()`` is decorated with ``@traceable`` so it appears as a discrete
child step under the ``studymate_pipeline`` parent span in LangSmith.  The
ChromaDB similarity search is a fast local operation (no LLM call), so the
step reflects the embedding API latency plus the vector-search time.

Usage::

    from backend.agents.rag_agent import run_rag
    from backend.agents.synthesis_agent import RagChunk

    chunks: list[RagChunk] = run_rag(
        query="Explain Newton's third law",
        session_id="student-session-42",
    )
    for chunk in chunks:
        print(chunk.source, chunk.score, chunk.content[:80])
"""

from __future__ import annotations

import logging

from langsmith import traceable

# RagChunk is defined in synthesis_agent to avoid a separate shared-types
# module; rag_agent imports it from there so both sides use the same class.
from backend.agents.synthesis_agent import RagChunk
from backend.rag.retriever import retrieve

logger = logging.getLogger(__name__)

# Default top-k: spec says 4-6, so 5 is used as the midpoint.
_DEFAULT_K = 5


@traceable(name="rag_agent")
def run_rag(
    query: str,
    session_id: str = "default",
    k: int = _DEFAULT_K,
) -> list[RagChunk]:
    """Return the top-k most-relevant chunks for *query* from ChromaDB.

    Converts the :class:`~backend.rag.retriever.RetrievedChunk` objects
    returned by :func:`~backend.rag.retriever.retrieve` into the
    :class:`~backend.agents.synthesis_agent.RagChunk` type consumed by
    :func:`~backend.agents.synthesis_agent.synthesize`.

    Args:
        query:      The student's question (must not be empty).
        session_id: Identifies the ChromaDB collection to search — must match
                    the ``session_id`` used when the documents were ingested.
        k:          Number of chunks to retrieve (default 5, per spec §7).

    Returns:
        List of :class:`RagChunk` objects ordered by descending relevance
        score.  Returns an empty list if no documents have been ingested for
        the session yet (safe — the Synthesis Agent handles empty RAG input).

    Raises:
        ValueError: If *query* is empty or whitespace only.
    """
    query = query.strip()
    if not query:
        raise ValueError("run_rag received an empty query.")

    logger.info(
        "RAG agent: retrieving top-%d chunks for query %r (session='%s')",
        k,
        query,
        session_id,
    )

    raw_chunks = retrieve(query=query, session_id=session_id, k=k)

    chunks: list[RagChunk] = [
        RagChunk(
            content=rc.text,
            source=rc.source,
            score=rc.score if rc.score is not None else 0.0,
        )
        for rc in raw_chunks
    ]

    logger.info(
        "RAG agent: retrieved %d chunk(s) from session '%s'.",
        len(chunks),
        session_id,
    )
    return chunks
