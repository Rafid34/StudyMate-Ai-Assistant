"""PDF ingestion pipeline for StudyMate.

Public API
----------
ingest_pdf(pdf_path, session_id="default") -> int
    Load a PDF from *pdf_path*, split it into overlapping text chunks,
    embed each chunk using Google's ``text-embedding-004`` model via the
    Gemini API (model name passed without the ``models/`` prefix, required
    by langchain-google-genai 2.x), and persist the resulting vectors in a
    per-session ChromaDB collection.

    Returns the number of chunks stored.

Chunking strategy (Section 7 of spec)
--------------------------------------
RecursiveCharacterTextSplitter is used with:
  chunk_size    = 2500 chars  ≈ 625 tokens  (mid-point of the 500-800 target)
  chunk_overlap = 400  chars  ≈ 100 tokens
The approximation assumes ~4 characters per token (English prose average).

ChromaDB collection naming
---------------------------
Collections are namespaced per session: ``studymate_<sanitized_session_id>``.
Any character outside ``[a-zA-Z0-9_-]`` is replaced with ``_`` and the total
name is capped at 63 characters (ChromaDB's hard limit).
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from backend.config import CHROMA_DB_PATH, GEMINI_API_KEY

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Chunking constants
# 4 chars per token is a standard rough estimate for English text.
#   chunk_size=2500  → ~625 tokens (mid-range of the 500-800 token target)
#   chunk_overlap=400 → ~100 tokens
# ---------------------------------------------------------------------------
_CHUNK_SIZE = 2500
_CHUNK_OVERLAP = 400
_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

# ---------------------------------------------------------------------------
# Collection naming
# ---------------------------------------------------------------------------
_COLLECTION_PREFIX = "studymate_"
_MAX_COLLECTION_LEN = 63
_INVALID_CHAR_RE = re.compile(r"[^a-zA-Z0-9_-]")


def _collection_name(session_id: str) -> str:
    """Return a ChromaDB-safe collection name for *session_id*."""
    safe_id = _INVALID_CHAR_RE.sub("_", session_id)
    name = f"{_COLLECTION_PREFIX}{safe_id}"
    return name[:_MAX_COLLECTION_LEN]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def ingest_pdf(pdf_path: str | Path, session_id: str = "default") -> int:
    """Chunk, embed, and store a PDF in ChromaDB.

    Parameters
    ----------
    pdf_path:
        Filesystem path to the PDF file to ingest.
    session_id:
        Logical session identifier used to namespace the ChromaDB collection.
        Defaults to ``"default"`` for single-session use.

    Returns
    -------
    int
        Number of document chunks written to the vector store.

    Raises
    ------
    FileNotFoundError
        If *pdf_path* does not exist on disk.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    source_name = pdf_path.name
    collection = _collection_name(session_id)

    # 1. Load pages -----------------------------------------------------------
    logger.info("Loading PDF '%s'", source_name)
    loader = PyPDFLoader(str(pdf_path))
    pages = loader.load()

    if not pages:
        logger.warning("PDF produced no extractable pages: '%s'", source_name)
        return 0

    logger.info("Loaded %d page(s) from '%s'", len(pages), source_name)

    # 2. Split into overlapping chunks ----------------------------------------
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=_CHUNK_SIZE,
        chunk_overlap=_CHUNK_OVERLAP,
        separators=_SEPARATORS,
        length_function=len,
    )
    chunks = splitter.split_documents(pages)

    if not chunks:
        logger.warning("Splitter produced no chunks from '%s'", source_name)
        return 0

    logger.info("Split '%s' into %d chunk(s)", source_name, len(chunks))

    # 3. Stamp metadata on every chunk for downstream source attribution ------
    for chunk in chunks:
        chunk.metadata["source"] = source_name
        chunk.metadata["session_id"] = session_id

    # 4. Embed and persist into ChromaDB --------------------------------------
    logger.info(
        "Embedding %d chunk(s) and storing in ChromaDB collection '%s' at '%s'",
        len(chunks),
        collection,
        CHROMA_DB_PATH,
    )

    embeddings = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-2-preview",
        google_api_key=GEMINI_API_KEY,
    )

    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DB_PATH,
        collection_name=collection,
    )

    logger.info(
        "Ingestion complete — %d chunk(s) stored in collection '%s'.",
        len(chunks),
        collection,
    )
    return len(chunks)
