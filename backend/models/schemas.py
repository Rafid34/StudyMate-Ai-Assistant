"""Pydantic v2 request/response schemas for all StudyMate API endpoints.

Covers:
  - POST /upload  -> UploadResponse
  - POST /ask     -> AskRequest, Source, AskResponse
  - GET  /health  -> HealthResponse
"""

from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, field_validator


class UploadResponse(BaseModel):
    message: str
    filename: str
    file_type: Literal["pdf", "image"]
    doc_count: Optional[int] = None      # chunks stored; populated for PDFs
    extracted_text: Optional[str] = None  # OCR output; populated for images

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "message": "PDF processed and stored successfully.",
                    "filename": "lecture_notes.pdf",
                    "file_type": "pdf",
                    "doc_count": 42,
                    "extracted_text": None,
                },
                {
                    "message": "Image processed via OCR successfully.",
                    "filename": "whiteboard.png",
                    "file_type": "image",
                    "doc_count": None,
                    "extracted_text": "Newton's second law: F = ma ...",
                },
            ]
        }
    )


class AskRequest(BaseModel):
    query: str
    session_id: str

    @field_validator("query", mode="before")
    @classmethod
    def query_must_not_be_blank(cls, v: str) -> str:
        stripped = v.strip()
        if not stripped:
            raise ValueError("query must not be empty or whitespace only")
        return stripped


class Source(BaseModel):
    type: Literal["notes", "web"]  # "notes" = RAG chunk, "web" = search result
    title: str                      # filename for notes, page title for web
    content_snippet: str            # short excerpt to show in UI
    url: Optional[str] = None       # only for web sources


class AskResponse(BaseModel):
    answer: str
    sources: List[Source]
    trace_url: Optional[str] = None  # LangSmith trace URL; None if tracing is off


class HealthResponse(BaseModel):
    status: Literal["ok"]
    version: str = "1.0.0"
