"""OCR Agent — extracts text from an uploaded image using Gemini Vision.

Spec reference: Section 6 (Agent Responsibilities — OCR Agent).

LangSmith tracing
-----------------
``run_ocr()`` is decorated with ``@traceable(name="ocr_agent")`` so it
appears as a distinct labelled step in LangSmith alongside the router,
rag_agent, search_agent, and synthesis_agent spans.  When called at
``/upload`` time it creates a top-level trace; the inner
``ChatGoogleGenerativeAI`` call is auto-traced as a child step by
LangChain because ``LANGCHAIN_TRACING_V2`` is set in ``backend/config.py``.

Usage::

    from backend.agents.ocr_agent import run_ocr

    extracted_text = run_ocr(image_bytes, mime_type="image/png")
"""

from __future__ import annotations

import base64
import logging

from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langsmith import traceable

from backend.config import GEMINI_API_KEY
from backend.prompts.templates import OCR_EXTRACTION_PROMPT

logger = logging.getLogger(__name__)

_MODEL_NAME = "gemini-3.6-flash"


def _build_llm() -> ChatGoogleGenerativeAI:
    """Instantiate the Gemini Vision LLM.

    A new instance is created per call so that LangSmith traces each
    invocation as a discrete, named step with its own run ID.
    temperature=0 keeps OCR output deterministic.
    """
    return ChatGoogleGenerativeAI(
        model=_MODEL_NAME,
        google_api_key=GEMINI_API_KEY,
        temperature=0,
    )


@traceable(name="ocr_agent")
def run_ocr(image_bytes: bytes, mime_type: str = "image/jpeg") -> str:
    """Send *image_bytes* to Gemini Vision and return the extracted text.

    The image is base64-encoded and embedded in a multimodal ``HumanMessage``
    as a data-URL so that LangChain's ``ChatGoogleGenerativeAI`` can forward
    it to the Gemini Vision API.  This function is decorated with
    ``@traceable(name="ocr_agent")`` so it appears as a labelled step in
    LangSmith alongside the other agents.  The inner LLM call is also
    auto-traced by LangChain because ``LANGCHAIN_TRACING_V2`` is set in
    ``backend/config.py``.

    Args:
        image_bytes: Raw bytes of the uploaded image (JPEG, PNG, WEBP, ...).
        mime_type:   MIME type string for the image, e.g. ``"image/jpeg"``
                     or ``"image/png"``.  Defaults to ``"image/jpeg"``.

    Returns:
        The raw text extracted from the image as a single string.  Structure
        (headings, lists, tables, paragraph breaks) is preserved per the
        prompt in :data:`backend.prompts.templates.OCR_EXTRACTION_PROMPT`.

    Raises:
        ValueError: If *image_bytes* is empty.
        Exception:  Propagates any Gemini API or network error to the caller
                    so that the FastAPI endpoint can return a meaningful HTTP
                    error response.
    """
    if not image_bytes:
        raise ValueError("run_ocr received empty image_bytes.")

    logger.info(
        "OCR agent: processing image (%d bytes, mime_type=%s)",
        len(image_bytes),
        mime_type,
    )

    b64_image = base64.b64encode(image_bytes).decode("utf-8")
    data_url = f"data:{mime_type};base64,{b64_image}"

    message = HumanMessage(
        content=[
            {
                "type": "image_url",
                "image_url": {"url": data_url},
            },
            {
                "type": "text",
                "text": OCR_EXTRACTION_PROMPT,
            },
        ]
    )

    llm = _build_llm()
    response = llm.invoke([message])

    # langchain-google-genai 4.x returns content as a list of block dicts;
    # older versions returned a plain string — handle both.
    raw = response.content
    if isinstance(raw, list):
        extracted_text: str = "".join(
            block.get("text", "") if isinstance(block, dict) else str(block)
            for block in raw
        )
    else:
        extracted_text = str(raw)
    logger.info(
        "OCR agent: extraction complete — %d characters returned.",
        len(extracted_text),
    )

    return extracted_text
