"""Search Agent — web search grounding via Gemini's native google_search tool or Tavily.

Spec reference: Section 6 (Agent Responsibilities — Search Agent).

Backend selection
-----------------
Configured by the ``SEARCH_BACKEND`` environment variable:

``gemini`` (default)
    Uses ``gemini-2.0-flash`` with its native Google Search grounding tool
    via the new ``google-genai`` SDK.
    Requires ``GEMINI_API_KEY``.
    On error, automatically falls back to Tavily when ``TAVILY_API_KEY`` is set.

``tavily``
    Uses the Tavily Search API directly.
    Requires ``TAVILY_API_KEY``.

LangSmith tracing
-----------------
Both paths are decorated with ``@traceable`` so they appear as discrete steps
in the LangSmith run tree alongside the LangChain-native agents.

Usage::

    from backend.agents.search_agent import run_search, SearchResult

    result: SearchResult = run_search("What are the key differences between RAM and ROM?")
    print(result.summary)
    for source in result.sources:
        print(source.url, source.title)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from langsmith import traceable

from backend.config import GEMINI_API_KEY, SEARCH_BACKEND, TAVILY_API_KEY
from backend.models.schemas import Source
from backend.prompts.templates import SEARCH_SUMMARY_SYSTEM_INSTRUCTION

logger = logging.getLogger(__name__)

_MODEL_NAME = "gemini-2.0-flash"


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------


@dataclass
class SearchResult:
    """Structured output from :func:`run_search`.

    Attributes:
        summary: LLM-generated (Gemini) or pre-built (Tavily) answer to the query.
        sources: Web sources with URLs, titles, and optional content snippets.
    """

    summary: str
    sources: list[Source] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Gemini-native Google Search grounding path (google-genai SDK 2.x)
# ---------------------------------------------------------------------------


def _extract_gemini_sources(response) -> list[Source]:
    """Pull web source URLs out of a Gemini ``grounding_metadata`` object.

    Returns an empty list (rather than raising) when metadata is absent so
    that a valid summary can still be returned to the caller.
    """
    sources: list[Source] = []
    try:
        grounding_meta = response.candidates[0].grounding_metadata
        for chunk in grounding_meta.grounding_chunks:
            web = getattr(chunk, "web", None)
            if web and getattr(web, "uri", None):
                sources.append(
                    Source(
                        type="web",
                        title=getattr(web, "title", None) or web.uri,
                        content_snippet="",
                        url=web.uri,
                    )
                )
    except (AttributeError, IndexError, TypeError):
        logger.warning(
            "Search agent (Gemini): grounding_metadata unavailable — sources will be empty."
        )
    return sources


@traceable(name="search_agent:gemini_grounding")
def _run_gemini_search(query: str) -> SearchResult:
    """Call Gemini with the native ``google_search`` grounding tool.

    Uses the new ``google-genai`` SDK (``google-genai`` package).

    Args:
        query: The natural-language search question.

    Returns:
        :class:`SearchResult` with a generated summary and grounded source URLs.

    Raises:
        ValueError: If ``GEMINI_API_KEY`` is not configured.
        Exception:  Propagates any Gemini API or network error to the caller
                    so that :func:`run_search` can attempt the Tavily fallback.
    """
    from google import genai  # google-genai SDK 2.x
    from google.genai import types

    if not GEMINI_API_KEY:
        raise ValueError(
            "GEMINI_API_KEY is not set; cannot use Gemini search grounding."
        )

    client = genai.Client(api_key=GEMINI_API_KEY)

    logger.info("Search agent (Gemini): sending query — '%s'", query)

    response = client.models.generate_content(
        model=_MODEL_NAME,
        contents=query,
        config=types.GenerateContentConfig(
            tools=[types.Tool(google_search=types.GoogleSearch())],
            system_instruction=SEARCH_SUMMARY_SYSTEM_INSTRUCTION,
        ),
    )

    summary: str = response.text or ""
    sources = _extract_gemini_sources(response)

    logger.info(
        "Search agent (Gemini): done — summary=%d chars, sources=%d.",
        len(summary),
        len(sources),
    )
    return SearchResult(summary=summary, sources=sources)


# ---------------------------------------------------------------------------
# Tavily API path
# ---------------------------------------------------------------------------


@traceable(name="search_agent:tavily")
def _run_tavily_search(query: str) -> SearchResult:
    """Call the Tavily Search API and return a summary with source list.

    Uses Tavily's built-in ``include_answer=True`` option for the primary
    summary.  If Tavily returns no synthesised answer, a brief summary is
    assembled from the top two result snippets instead.

    Args:
        query: The natural-language search question.

    Returns:
        :class:`SearchResult` with a Tavily-generated summary and source URLs.

    Raises:
        ValueError: If ``TAVILY_API_KEY`` is not configured.
        Exception:  Propagates any Tavily API or network error to the caller.
    """
    from tavily import TavilyClient  # lazy — listed in requirements.txt

    if not TAVILY_API_KEY:
        raise ValueError(
            "TAVILY_API_KEY is not set; cannot use Tavily search."
        )

    client = TavilyClient(api_key=TAVILY_API_KEY)

    logger.info("Search agent (Tavily): sending query — '%s'", query)
    response = client.search(
        query=query,
        max_results=5,
        include_answer=True,
        search_depth="advanced",
    )

    # Prefer the pre-built answer; fall back to joining the top-2 snippets.
    summary: str = response.get("answer") or ""
    if not summary:
        snippets = [r.get("content", "") for r in response.get("results", [])[:2]]
        summary = " ".join(s for s in snippets if s)

    sources: list[Source] = [
        Source(
            type="web",
            title=r.get("title") or r.get("url", "Unknown"),
            content_snippet=(r.get("content") or "")[:300],
            url=r.get("url"),
        )
        for r in response.get("results", [])
    ]

    logger.info(
        "Search agent (Tavily): done — summary=%d chars, sources=%d.",
        len(summary),
        len(sources),
    )
    return SearchResult(summary=summary, sources=sources)


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------


@traceable(name="search_agent")
def run_search(query: str) -> SearchResult:
    """Answer *query* with current web information, returning a summary and sources.

    This is the single entry point used by the router/synthesis pipeline.

    Backend selection (controlled by the ``SEARCH_BACKEND`` env var):

    * ``"gemini"`` *(default)* — Gemini with native Google Search grounding
      via the ``google-genai`` SDK.  Falls back to Tavily automatically on any
      error when ``TAVILY_API_KEY`` is also set.
    * ``"tavily"`` — Tavily API, used directly without touching Gemini.

    Args:
        query: The question to answer using live web search.

    Returns:
        :class:`SearchResult` containing:

        - ``summary`` — answer synthesised from web sources.
        - ``sources`` — list of :class:`~backend.models.schemas.Source` objects
          (``type="web"``) each carrying a ``url``, ``title``, and optional
          ``content_snippet``.

    Raises:
        ValueError:   If *query* is empty.
        RuntimeError: If the configured backend fails and no fallback is available.
    """
    query = query.strip()
    if not query:
        raise ValueError("run_search received an empty query.")

    if SEARCH_BACKEND == "tavily":
        logger.info("Search agent: using Tavily backend (SEARCH_BACKEND=tavily).")
        return _run_tavily_search(query)

    # Default path: Gemini grounding with automatic Tavily fallback on error.
    logger.info("Search agent: using Gemini grounding backend.")
    try:
        return _run_gemini_search(query)
    except Exception as gemini_exc:
        logger.warning(
            "Search agent: Gemini grounding failed — %s. %s",
            gemini_exc,
            "Falling back to Tavily." if TAVILY_API_KEY else "No Tavily fallback configured.",
        )
        if TAVILY_API_KEY:
            return _run_tavily_search(query)
        raise RuntimeError(
            "Gemini search grounding failed and no Tavily fallback is configured. "
            "Set TAVILY_API_KEY in your .env to enable the fallback."
        ) from gemini_exc
