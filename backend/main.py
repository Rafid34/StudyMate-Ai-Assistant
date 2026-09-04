"""StudyMate FastAPI application — Section 5 implementation.

Routes
------
GET  /health  — Liveness check.
POST /upload  — Ingest a PDF into ChromaDB OR run OCR on an image.
POST /ask     — Full Router → Agent(s) → Synthesis pipeline.

LangSmith tracing
-----------------
``backend.config`` is imported first so that the LangChain env vars
(``LANGCHAIN_TRACING_V2``, ``LANGCHAIN_PROJECT``, ``LANGCHAIN_API_KEY``) are
written into ``os.environ`` before any LangChain object is constructed.  This
guarantees that every agent and LLM call — including those triggered by
importing the agent modules — is captured under the correct project.

The ``/ask`` pipeline is wrapped in the ``@traceable`` function
``_run_ask_pipeline``, which creates a single parent span called
``studymate_pipeline`` in LangSmith.  Each agent decorated with its own
``@traceable`` (router, rag, search, synthesis) appears as a labelled child
step under that parent, giving you the full Router → Agents → Synthesis flow
in one trace.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from langsmith import traceable

# ── Import config first ──────────────────────────────────────────────────────
# This must precede all other backend imports so that the LangChain / LangSmith
# env vars are written into os.environ before any LangChain objects are built.
import backend.config  # noqa: F401

from backend.agents.ocr_agent import run_ocr
from backend.agents.rag_agent import run_rag
from backend.agents.router import RouterDecision, route
from backend.agents.search_agent import SearchResult, run_search
from backend.agents.synthesis_agent import RagChunk, SynthesisResult, synthesize
from backend.models.schemas import (
    AskRequest,
    AskResponse,
    HealthResponse,
    Source,
    UploadResponse,
)
from backend.rag.ingest import ingest_pdf

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── FastAPI app ──────────────────────────────────────────────────────────────

app = FastAPI(title="StudyMate API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── In-memory session state ──────────────────────────────────────────────────
# These dicts are process-local and reset on server restart — acceptable for
# the single-session demo described in Section 12 of the spec.

# Maps session_id → extracted OCR text from the most-recently uploaded image.
# Cleared when a new image is uploaded for the same session.
_ocr_cache: dict[str, str] = {}

# Tracks which sessions have at least one ingested PDF in ChromaDB.
# Updated on each successful /upload (PDF path).
_sessions_with_docs: set[str] = set()

# ── File-type helpers ────────────────────────────────────────────────────────

_IMAGE_MIME_TYPES: frozenset[str] = frozenset(
    {
        "image/jpeg",
        "image/jpg",
        "image/png",
        "image/webp",
        "image/gif",
        "image/bmp",
        "image/tiff",
    }
)

_IMAGE_EXTENSIONS: frozenset[str] = frozenset(
    {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".tiff"}
)

# Maps common image extensions to canonical MIME types for the Gemini Vision
# call, used when the browser sends a generic or empty Content-Type header.
_EXT_TO_MIME: dict[str, str] = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
    ".gif": "image/gif",
    ".bmp": "image/bmp",
    ".tiff": "image/tiff",
}


def _is_image(content_type: str | None, filename: str) -> bool:
    """Return True if the file looks like a supported image."""
    if content_type and content_type.lower() in _IMAGE_MIME_TYPES:
        return True
    return Path(filename).suffix.lower() in _IMAGE_EXTENSIONS


def _resolve_mime_type(content_type: str | None, filename: str) -> str:
    """Return the best MIME type string to pass to the OCR agent."""
    if content_type and content_type.lower() in _IMAGE_MIME_TYPES:
        return content_type.lower()
    return _EXT_TO_MIME.get(Path(filename).suffix.lower(), "image/jpeg")


# ── Routes ───────────────────────────────────────────────────────────────────


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Liveness check — always returns ``{"status": "ok"}``."""
    return HealthResponse(status="ok")


@app.post("/upload", response_model=UploadResponse)
async def upload(
    file: UploadFile = File(...),
    session_id: str = Form("default"),
) -> UploadResponse:
    """Accept a PDF or image file and process it for the given session.

    PDF path
    --------
    The file is written to a temporary location, chunked with
    ``RecursiveCharacterTextSplitter`` (~625-token chunks, ~100-token overlap),
    embedded via Gemini ``text-embedding-004``, and stored in a per-session
    ChromaDB collection.  Returns ``doc_count`` (number of chunks stored).

    Image path
    ----------
    The raw bytes are sent to the OCR agent (Gemini Vision).  The extracted
    text is returned in ``extracted_text`` *and* cached in ``_ocr_cache`` so
    the next ``/ask`` call for the same session can include it as context.
    """
    filename = file.filename or "upload"
    content_type = file.content_type or ""

    is_pdf = content_type == "application/pdf" or filename.lower().endswith(".pdf")
    is_img = _is_image(content_type, filename)

    if not is_pdf and not is_img:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported file type '{content_type}' ('{filename}'). "
                "Accepted types: PDF (.pdf) and images (.jpg, .png, .webp, .gif, .bmp, .tiff)."
            ),
        )

    file_bytes = await file.read()

    # ── PDF: chunk + embed + store ──────────────────────────────────────────
    if is_pdf:
        tmp_path: Optional[str] = None
        try:
            # Write to a temp file and close it before PyPDFLoader opens it
            # (required on Windows where open file handles block re-opens).
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(file_bytes)
                tmp_path = tmp.name

            logger.info(
                "/upload PDF: '%s' (%d bytes) → session '%s'",
                filename,
                len(file_bytes),
                session_id,
            )
            chunks_stored = ingest_pdf(tmp_path, session_id)
            logger.info("Ingestion complete: %d chunk(s) stored.", chunks_stored)

            # Mark this session as having at least one ingested document.
            _sessions_with_docs.add(session_id)

            return UploadResponse(
                message="PDF processed and stored successfully.",
                filename=filename,
                file_type="pdf",
                doc_count=chunks_stored,
            )

        except HTTPException:
            raise
        except Exception as exc:
            logger.exception("Unexpected error during PDF ingestion")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to ingest PDF: {exc}",
            ) from exc

        finally:
            if tmp_path is not None:
                Path(tmp_path).unlink(missing_ok=True)

    # ── Image: OCR via Gemini Vision ────────────────────────────────────────
    mime_type = _resolve_mime_type(content_type, filename)
    try:
        logger.info(
            "/upload image: '%s' (%d bytes, mime=%s) → session '%s'",
            filename,
            len(file_bytes),
            mime_type,
            session_id,
        )
        extracted_text = run_ocr(file_bytes, mime_type=mime_type)

        # Cache so /ask can inject the OCR text into the synthesis context.
        _ocr_cache[session_id] = extracted_text
        logger.info(
            "OCR complete: %d chars extracted, cached for session '%s'.",
            len(extracted_text),
            session_id,
        )

        return UploadResponse(
            message="Image processed via OCR successfully.",
            filename=filename,
            file_type="image",
            extracted_text=extracted_text,
        )

    except Exception as exc:
        logger.exception("Unexpected error during OCR")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process image via OCR: {exc}",
        ) from exc


# ── /ask pipeline ────────────────────────────────────────────────────────────


@traceable(name="studymate_pipeline")
def _run_ask_pipeline(
    query: str,
    session_id: str,
    cached_ocr: str | None,
    has_image: bool,
    has_docs: bool,
) -> tuple[str, list[Source], str | None]:
    """Orchestrate the full Router → Agent(s) → Synthesis pipeline.

    Wrapping this in ``@traceable`` creates a single *parent* span called
    ``studymate_pipeline`` in LangSmith.  Every downstream agent decorated
    with its own ``@traceable`` (router, rag_agent, search_agent,
    synthesis_agent) and every LangChain LLM call auto-traced by LangChain
    appear as labelled *child* steps under that parent, giving end-to-end
    visibility of the full pipeline in one trace.

    Args:
        query:       The student's question (already stripped).
        session_id:  Session identifier forwarded to the RAG agent.
        cached_ocr:  OCR text extracted during ``/upload``, or ``None``.
        has_image:   True when ``cached_ocr`` is not None (passed to router).
        has_docs:    True when this session has ingested PDFs (passed to router).

    Returns:
        A 3-tuple of ``(answer, sources, trace_url)``.  ``trace_url`` is the
        LangSmith URL for this run, or ``None`` when tracing is off.
    """
    # ── Step 1: Router ───────────────────────────────────────────────────────
    decision: RouterDecision = route(
        query=query,
        has_image=has_image,
        has_docs=has_docs,
    )
    logger.info(
        "Router decision — use_ocr=%s, use_rag=%s, use_search=%s | %s",
        decision.use_ocr,
        decision.use_rag,
        decision.use_search,
        decision.reasoning,
    )

    ocr_text: str | None = None
    rag_chunks: list[RagChunk] | None = None
    search_result: SearchResult | None = None

    # ── Step 2a: OCR (from cache — already ran at /upload time) ─────────────
    if decision.use_ocr and cached_ocr:
        ocr_text = cached_ocr
        logger.info("Pipeline: injecting cached OCR text (%d chars).", len(ocr_text))

    # ── Step 2b: RAG ─────────────────────────────────────────────────────────
    if decision.use_rag:
        rag_chunks = run_rag(query=query, session_id=session_id)
        logger.info("Pipeline: RAG returned %d chunk(s).", len(rag_chunks))

    # ── Step 2c: Search ──────────────────────────────────────────────────────
    if decision.use_search:
        search_result = run_search(query=query)
        logger.info(
            "Pipeline: search returned summary (%d chars) + %d source(s).",
            len(search_result.summary),
            len(search_result.sources),
        )

    # ── Step 3: Synthesis ────────────────────────────────────────────────────
    result: SynthesisResult = synthesize(
        query=query,
        ocr_text=ocr_text,
        rag_chunks=rag_chunks,
        search_result=search_result,
    )

    # ── Step 4: Grab the LangSmith trace URL (best-effort) ──────────────────
    trace_url: str | None = None
    try:
        from langsmith import get_current_run_tree  # available in langsmith ≥ 0.1

        run = get_current_run_tree()
        if run is not None:
            trace_url = run.get_url()
    except Exception:
        # Non-fatal: tracing may be disabled or the langsmith version may not
        # expose get_current_run_tree yet.
        pass

    return result.answer, result.sources, trace_url


@app.post("/ask", response_model=AskResponse)
async def ask(request: AskRequest) -> AskResponse:
    """Run the full Router → Agent(s) → Synthesis pipeline for *request.query*.

    Pipeline steps
    --------------
    1. Read session state — cached OCR text and whether PDFs have been ingested.
    2. Router agent — decides which of {OCR, RAG, Search} to invoke.
    3. Selected agents run sequentially (OCR is already done; RAG and Search
       run if flagged by the router).
    4. Synthesis agent — combines all context into one attributed answer.
    5. Return ``{ answer, sources, trace_url }`` to the caller.

    The entire pipeline is wrapped in a single ``@traceable`` span so every
    step is visible as a labelled child run in LangSmith.
    """
    session_id = request.session_id
    query = request.query  # already stripped/validated by the Pydantic schema

    # Determine session state from in-memory caches.
    cached_ocr = _ocr_cache.get(session_id)
    has_image = cached_ocr is not None
    has_docs = session_id in _sessions_with_docs

    logger.info(
        "/ask: session='%s', has_image=%s, has_docs=%s, query=%r",
        session_id,
        has_image,
        has_docs,
        query,
    )

    try:
        answer, sources, trace_url = _run_ask_pipeline(
            query=query,
            session_id=session_id,
            cached_ocr=cached_ocr,
            has_image=has_image,
            has_docs=has_docs,
        )
    except Exception as exc:
        logger.exception("Pipeline error for query %r", query)
        raise HTTPException(
            status_code=500,
            detail=f"Pipeline error: {exc}",
        ) from exc

    return AskResponse(answer=answer, sources=sources, trace_url=trace_url)
