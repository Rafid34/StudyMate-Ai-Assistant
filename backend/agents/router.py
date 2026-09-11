"""Router Agent — decides which agents to invoke for a given query.

Spec reference: Section 6 (Agent Responsibilities — Router Agent).

Decision logic
--------------
``use_ocr``
    Always ``True`` when ``has_image=True``.  This is a structural rule —
    an uploaded image must be processed regardless of query content.

``use_rag`` and ``use_search``
    Determined by Gemini 2.0 Flash via ``with_structured_output``.  The LLM
    is given the query text and session context (whether the student has
    uploaded course documents) and returns a typed decision.

    If the LLM call fails for any reason (missing key, network error, parse
    failure) the router falls back to a fast keyword-heuristic so the
    pipeline never hard-errors at the routing step.

LangSmith tracing
-----------------
``route()`` is decorated with ``@traceable``.  The inner
``ChatGoogleGenerativeAI`` invocation is auto-traced by LangChain because
``LANGCHAIN_TRACING_V2`` is set in ``backend/config.py``.

Usage::

    from backend.agents.router import route, RouterDecision

    decision: RouterDecision = route(
        query="Explain Newton's laws of motion",
        has_image=False,
        has_docs=True,
    )
    # RouterDecision(use_ocr=False, use_rag=True, use_search=False, reasoning="...")
"""

from __future__ import annotations

import logging

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langsmith import traceable
from pydantic import BaseModel, Field

from backend.config import GEMINI_API_KEY
from backend.prompts.templates import ROUTER_DECISION_PROMPT, ROUTER_SYSTEM_PROMPT

logger = logging.getLogger(__name__)

_MODEL_NAME = "gemini-2.0-flash"

# Keywords that strongly suggest the query needs live web information.
_SEARCH_SIGNALS: frozenset[str] = frozenset(
    {
        "latest", "current", "recent", "now", "today", "news", "update",
        "new", "trending", "who won", "what happened", "when did",
        "2024", "2025", "2026", "this year", "last year",
    }
)


# ---------------------------------------------------------------------------
# Public return type
# ---------------------------------------------------------------------------


class RouterDecision(BaseModel):
    """Structured routing decision produced by :func:`route`.

    Attributes:
        use_ocr:    Run the OCR agent — set when an image was uploaded.
        use_rag:    Query the ChromaDB vector store of uploaded course docs.
        use_search: Fetch live web results via the Search agent.
        reasoning:  One-sentence explanation; visible in the LangSmith trace.
    """

    use_ocr: bool
    use_rag: bool
    use_search: bool
    reasoning: str = ""


# ---------------------------------------------------------------------------
# Internal Pydantic schema for structured LLM output
# (excludes use_ocr — that flag is determined deterministically)
# ---------------------------------------------------------------------------


class _LLMDecision(BaseModel):
    """Schema passed to ``with_structured_output`` for the routing LLM call."""

    use_rag: bool = Field(
        description=(
            "True if the query can likely be answered from the student's uploaded "
            "course documents (PDFs/slides) in the vector store. "
            "Must be False when has_docs is False."
        )
    )
    use_search: bool = Field(
        description=(
            "True if the query needs current or external web information not "
            "typically found in course notes — e.g. recent events, live statistics, "
            "or up-to-date facts."
        )
    )
    reasoning: str = Field(
        description="One concise sentence explaining the routing decision."
    )


# ---------------------------------------------------------------------------
# LLM builder
# ---------------------------------------------------------------------------


def _build_llm() -> ChatGoogleGenerativeAI:
    """Instantiate Gemini 2.0 Flash for routing.

    temperature=0 makes routing decisions deterministic.  A new instance per
    call lets LangSmith record each invocation as a discrete, named step.
    """
    return ChatGoogleGenerativeAI(
        model=_MODEL_NAME,
        google_api_key=GEMINI_API_KEY,
        temperature=0,
    )


# ---------------------------------------------------------------------------
# Heuristic fallback (no API calls — always succeeds)
# ---------------------------------------------------------------------------


def _heuristic_route(query: str, has_docs: bool) -> tuple[bool, bool, str]:
    """Return ``(use_rag, use_search, reasoning)`` using keyword matching.

    Intentionally conservative: when in doubt, enable both sources so the
    student receives the most complete answer possible.
    """
    q_lower = query.lower()

    has_search_signal = any(token in q_lower for token in _SEARCH_SIGNALS)
    use_rag = has_docs
    use_search = has_search_signal or not has_docs

    # Safety: at least one path must be active.
    if not use_rag and not use_search:
        use_search = True

    parts: list[str] = []
    if use_rag:
        parts.append("course docs (RAG)")
    if use_search:
        parts.append("web search")
    reasoning = f"Heuristic fallback: consulting {' + '.join(parts)}."
    return use_rag, use_search, reasoning


# ---------------------------------------------------------------------------
# LLM routing path
# ---------------------------------------------------------------------------


def _llm_route(
    query: str,
    has_image: bool,
    has_docs: bool,
) -> tuple[bool, bool, str]:
    """Ask Gemini to classify the query and return ``(use_rag, use_search, reasoning)``.

    Raises:
        Exception: Propagates any LLM or network error to :func:`route` so
                   the heuristic fallback can be activated.
    """
    structured_llm = _build_llm().with_structured_output(_LLMDecision)

    user_prompt = ROUTER_DECISION_PROMPT.format(
        has_docs=has_docs,
        has_image=has_image,
        query=query,
    )

    messages = [
        SystemMessage(content=ROUTER_SYSTEM_PROMPT),
        HumanMessage(content=user_prompt),
    ]

    logger.info(
        "Router agent (LLM): classifying query (has_docs=%s, has_image=%s) — '%s'",
        has_docs,
        has_image,
        query,
    )

    decision: _LLMDecision = structured_llm.invoke(messages)

    # Enforce hard constraint: RAG requires docs to exist.
    if decision.use_rag and not has_docs:
        logger.warning(
            "Router agent: LLM set use_rag=True but has_docs=False — overriding to False."
        )
        decision.use_rag = False

    # Safety: at least one path must be active.
    if not decision.use_rag and not decision.use_search:
        logger.warning(
            "Router agent: LLM set both flags to False — forcing use_search=True."
        )
        decision.use_search = True

    return decision.use_rag, decision.use_search, decision.reasoning


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------


@traceable(name="router_agent")
def route(
    query: str,
    has_image: bool = False,
    has_docs: bool = False,
) -> RouterDecision:
    """Decide which agents to invoke for *query* given the session context.

    ``use_ocr`` is set deterministically from ``has_image``.
    ``use_rag`` and ``use_search`` are determined by a Gemini LLM call, with
    an automatic keyword-heuristic fallback on any LLM failure.

    Args:
        query:     The student's question text (must not be empty).
        has_image: ``True`` if the user uploaded a fresh image in this request.
        has_docs:  ``True`` if the session's ChromaDB collection has at least
                   one ingested document.

    Returns:
        :class:`RouterDecision` with all three routing flags and a short
        ``reasoning`` string that is visible in the LangSmith trace.

    Raises:
        ValueError: If *query* is empty or whitespace only.
    """
    query = query.strip()
    if not query:
        raise ValueError("route() received an empty query.")

    # OCR is always activated by the presence of an image — no LLM needed.
    use_ocr = has_image

    # Attempt LLM routing; fall back to heuristics on any failure.
    try:
        use_rag, use_search, reasoning = _llm_route(query, has_image, has_docs)
    except Exception as exc:
        logger.warning(
            "Router agent: LLM routing failed (%s) — using heuristic fallback.", exc
        )
        use_rag, use_search, reasoning = _heuristic_route(query, has_docs)

    decision = RouterDecision(
        use_ocr=use_ocr,
        use_rag=use_rag,
        use_search=use_search,
        reasoning=reasoning,
    )

    logger.info(
        "Router agent: decision — use_ocr=%s, use_rag=%s, use_search=%s | %s",
        decision.use_ocr,
        decision.use_rag,
        decision.use_search,
        decision.reasoning,
    )

    return decision
