"""Synthesis Agent — combines OCR text, RAG chunks, and search results into one answer.

Spec reference: Section 6 (Agent Responsibilities — Synthesis Agent).

The synthesis agent is the final stage of the StudyMate pipeline.  It receives
whatever context the upstream agents produced — any combination of OCR text,
retrieved RAG chunks, and a search result — and uses Gemini 2.0 Flash to write
a single, coherent answer that clearly attributes every fact to its source.

Attribution labels used in the generated answer
------------------------------------------------
``"From your uploaded image:"``
    Content derived from OCR text extracted by the OCR agent.
``"From your notes:"``
    Content from a retrieved ChromaDB chunk (the student's own documents).
``"From the web:"``
    Content from the Search agent's summary.

Data types
----------
:class:`RagChunk`
    Lightweight container for one retrieved chunk from ChromaDB.  Defined here
    so that ``rag_agent.py`` imports and reuses this type rather than defining
    its own incompatible struct.

:class:`SynthesisResult`
    Return type of :func:`synthesize`.  ``answer`` is the final prose response;
    ``sources`` is the flat list of all :class:`~backend.models.schemas.Source`
    objects collected from every active agent, ready to pass directly into
    :class:`~backend.models.schemas.AskResponse`.

LangSmith tracing
-----------------
``synthesize()`` is decorated with ``@traceable``.  The inner
``ChatGoogleGenerativeAI`` call is auto-traced by LangChain because
``LANGCHAIN_TRACING_V2`` is set in ``backend/config.py``.

Usage::

    from backend.agents.synthesis_agent import RagChunk, SynthesisResult, synthesize
    from backend.agents.search_agent import SearchResult

    result: SynthesisResult = synthesize(
        query="Explain Newton's third law",
        ocr_text="For every action there is an equal and opposite reaction.",
        rag_chunks=[
            RagChunk(content="Newton's 3rd law: ...", source="physics_notes.pdf"),
        ],
        search_result=SearchResult(summary="Newton's third law means ...", sources=[]),
    )
    print(result.answer)
    for source in result.sources:
        print(source.type, source.title)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langsmith import traceable

from backend.agents.search_agent import SearchResult
from backend.config import GEMINI_API_KEY
from backend.models.schemas import Source
from backend.prompts.templates import (
    SYNTHESIS_CONTEXT_NO_CONTEXT,
    SYNTHESIS_CONTEXT_OCR_HEADER,
    SYNTHESIS_CONTEXT_SEARCH_HEADER,
    SYNTHESIS_SYSTEM_PROMPT,
    build_synthesis_user_message,
    synthesis_rag_header,
)

logger = logging.getLogger(__name__)

_MODEL_NAME = "gemini-3.6-flash"

# Maximum characters stored in a Source.content_snippet sent back to the UI.
_SNIPPET_MAX = 300


@dataclass
class RagChunk:
    """A single chunk retrieved from the ChromaDB vector store.

    This type is the contract between ``rag_agent.py`` (producer) and
    ``synthesis_agent.py`` (consumer).  ``rag_agent.py`` should import
    this class and return ``list[RagChunk]`` from its public function.

    Attributes:
        content: The raw chunk text (may span several paragraphs).
        source:  The source document filename, e.g. ``"lecture_03.pdf"``.
        score:   Cosine-similarity score from the retrieval step.  Informational
                 only; not sent to the LLM.
    """

    content: str
    source: str
    score: float = 0.0


@dataclass
class SynthesisResult:
    """Final output of :func:`synthesize`.

    Attributes:
        answer:  The LLM-generated answer combining all context sources.
        sources: Flat list of every :class:`~backend.models.schemas.Source`
                 from all active agents.  Pass this directly to
                 :class:`~backend.models.schemas.AskResponse`.
    """

    answer: str
    sources: list[Source] = field(default_factory=list)


def _build_llm() -> ChatGoogleGenerativeAI:
    """Instantiate Gemini 2.0 Flash for synthesis.

    temperature=0.2 allows fluent, natural prose while keeping the answer
    grounded in the provided context.  A new instance is created per call
    so LangSmith records each invocation as a discrete, named step.
    """
    return ChatGoogleGenerativeAI(
        model=_MODEL_NAME,
        google_api_key=GEMINI_API_KEY,
        temperature=0.2,
    )


def _build_context_blocks(
    ocr_text: str | None,
    rag_chunks: list[RagChunk] | None,
    search_result: SearchResult | None,
) -> str:
    """Assemble a clearly labelled context string for the synthesis prompt.

    Each source type gets its own delimited section.  The LLM uses the
    section headers as anchors for in-text attribution ("From your notes:",
    "From the web:", etc.).

    The section order — OCR → RAG → Search — places the most personal,
    student-specific context first, so the LLM weighs it appropriately.

    Returns:
        A multi-line string ready to embed in the user prompt, or a note
        that no external context is available (answered from general knowledge).
    """
    blocks: list[str] = []

    if ocr_text and ocr_text.strip():
        blocks.append(
            SYNTHESIS_CONTEXT_OCR_HEADER + "\n"
            + ocr_text.strip()
        )

    for chunk in (rag_chunks or []):
        if chunk.content and chunk.content.strip():
            blocks.append(
                synthesis_rag_header(chunk.source) + "\n"
                + chunk.content.strip()
            )

    if search_result and search_result.summary and search_result.summary.strip():
        blocks.append(
            SYNTHESIS_CONTEXT_SEARCH_HEADER + "\n"
            + search_result.summary.strip()
        )

    if not blocks:
        return SYNTHESIS_CONTEXT_NO_CONTEXT

    return "\n\n".join(blocks)


def _collect_sources(
    ocr_text: str | None,
    rag_chunks: list[RagChunk] | None,
    search_result: SearchResult | None,
) -> list[Source]:
    """Build the flat :class:`Source` list for the API response.

    Mapping:
    - OCR text   → one ``type="notes"`` Source titled ``"Uploaded Image (OCR)"``.
    - RAG chunks → one ``type="notes"`` Source per chunk, titled by filename.
    - Search     → the ``Source`` objects from :class:`SearchResult` (``type="web"``)
                   passed through as-is.
    """
    sources: list[Source] = []

    if ocr_text and ocr_text.strip():
        sources.append(
            Source(
                type="notes",
                title="Uploaded Image (OCR)",
                content_snippet=ocr_text.strip()[:_SNIPPET_MAX],
                url=None,
            )
        )

    for chunk in (rag_chunks or []):
        if chunk.content and chunk.content.strip():
            sources.append(
                Source(
                    type="notes",
                    title=chunk.source,
                    content_snippet=chunk.content.strip()[:_SNIPPET_MAX],
                    url=None,
                )
            )

    if search_result:
        sources.extend(search_result.sources)

    return sources


@traceable(name="synthesis_agent")
def synthesize(
    query: str,
    ocr_text: str | None = None,
    rag_chunks: list[RagChunk] | None = None,
    search_result: SearchResult | None = None,
) -> SynthesisResult:
    """Combine all agent outputs into one coherent, attributed answer.

    Any combination of the three optional context arguments is accepted,
    including all ``None`` (the LLM answers from general knowledge and notes
    that no sources were consulted).

    Args:
        query:         The original student question (must not be empty).
        ocr_text:      Raw text from the OCR agent, or ``None``.
        rag_chunks:    Retrieved ChromaDB chunks from the RAG agent, or ``None``.
        search_result: Summary and sources from the Search agent, or ``None``.

    Returns:
        :class:`SynthesisResult` containing the final ``answer`` string and a
        ``sources`` list covering all active agents, ready for
        :class:`~backend.models.schemas.AskResponse`.

    Raises:
        ValueError: If *query* is empty or whitespace only.
        Exception:  Propagates any Gemini API or network error to the caller
                    so the FastAPI endpoint can return an appropriate HTTP error.
    """
    query = query.strip()
    if not query:
        raise ValueError("synthesize() received an empty query.")

    context_blocks = _build_context_blocks(ocr_text, rag_chunks, search_result)
    sources = _collect_sources(ocr_text, rag_chunks, search_result)

    logger.info(
        "Synthesis agent: query='%s', ocr=%s, rag_chunks=%d, search=%s.",
        query,
        "yes" if ocr_text else "no",
        len(rag_chunks) if rag_chunks else 0,
        "yes" if search_result else "no",
    )

    # Build the user message with an f-string rather than str.format() so that
    # curly braces inside student notes or OCR text do not cause a KeyError.
    user_message = build_synthesis_user_message(query, context_blocks)

    llm = _build_llm()
    response = llm.invoke(
        [
            SystemMessage(content=SYNTHESIS_SYSTEM_PROMPT),
            HumanMessage(content=user_message),
        ]
    )

    # langchain-google-genai 4.x returns content as a list of block dicts;
    # older versions returned a plain string — handle both.
    raw = response.content
    if isinstance(raw, list):
        answer: str = "".join(
            block.get("text", "") if isinstance(block, dict) else str(block)
            for block in raw
        )
    else:
        answer = str(raw)
    logger.info(
        "Synthesis agent: done — %d chars, %d source(s).",
        len(answer),
        len(sources),
    )

    return SynthesisResult(answer=answer, sources=sources)
