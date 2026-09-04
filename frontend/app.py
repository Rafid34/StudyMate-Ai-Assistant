"""StudyMate — Streamlit frontend.

Single-page app with:
  - Sidebar: file uploader (PDF/image) → calls POST /upload
  - Main area: chat interface → calls POST /ask, renders answer + sources
"""

from __future__ import annotations

import uuid
from typing import Any

import requests
import streamlit as st

# ── Must be the very first Streamlit call ────────────────────────────────────
st.set_page_config(
    page_title="StudyMate — AI Study Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Constants ────────────────────────────────────────────────────────────────
DEFAULT_BACKEND_URL = "http://localhost:8000"
UPLOAD_TIMEOUT_S = 120  # PDFs can take a while to embed
ASK_TIMEOUT_S = 90


# ── Helper: render source cards ──────────────────────────────────────────────
def _render_sources(sources: list[dict[str, Any]]) -> None:
    """Render an expandable source panel beneath an assistant message."""
    if not sources:
        return

    notes = [s for s in sources if s.get("type") == "notes"]
    web = [s for s in sources if s.get("type") == "web"]

    with st.expander(f"📎 {len(sources)} source(s) used", expanded=False):
        if notes:
            st.markdown("**📄 From your notes**")
            for src in notes:
                st.markdown(f"**{src['title']}**")
                st.caption(src.get("content_snippet", ""))
                if notes.index(src) < len(notes) - 1:
                    st.divider()

        if notes and web:
            st.divider()

        if web:
            st.markdown("**🌐 From the web**")
            for src in web:
                url = src.get("url")
                label = f"[{src['title']}]({url})" if url else src["title"]
                st.markdown(f"**{label}**")
                st.caption(src.get("content_snippet", ""))
                if web.index(src) < len(web) - 1:
                    st.divider()


# ── Session state defaults ───────────────────────────────────────────────────
def _init_state() -> None:
    defaults: dict[str, Any] = {
        "session_id": str(uuid.uuid4()),
        # Chat history: list of dicts with keys role, content, sources, trace_url
        "messages": [],
        # Files successfully sent to /upload: list of {filename, file_type, info}
        "uploaded_files": [],
        # Set of "<name>:<size>" strings to avoid re-uploading on rerun
        "processed_file_keys": set(),
        "backend_url": DEFAULT_BACKEND_URL,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


_init_state()

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📚 StudyMate")
    st.caption("AI-powered study assistant")

    st.divider()

    # ── Backend settings ─────────────────────────────────────────────────────
    with st.expander("⚙️ Settings", expanded=False):
        new_url = st.text_input(
            "Backend URL",
            value=st.session_state.backend_url,
            help="Base URL of the running FastAPI server.",
        )
        st.session_state.backend_url = new_url.rstrip("/")

        if st.button("Check connection", use_container_width=True):
            try:
                r = requests.get(
                    f"{st.session_state.backend_url}/health", timeout=5
                )
                if r.ok:
                    st.success("✅ Backend is online")
                else:
                    st.warning(f"Backend returned HTTP {r.status_code}")
            except requests.ConnectionError:
                st.error("❌ Could not reach backend")

    st.divider()

    # ── File uploader ─────────────────────────────────────────────────────────
    st.subheader("📁 Upload Study Materials")
    st.caption("PDF → indexed for RAG  •  Image → OCR text extraction")

    uploaded_file = st.file_uploader(
        "Upload a PDF or image",
        type=["pdf", "jpg", "jpeg", "png", "webp", "gif", "bmp", "tiff"],
        label_visibility="collapsed",
        help="Supported: .pdf, .jpg, .png, .webp, .gif, .bmp, .tiff",
    )

    if uploaded_file is not None:
        file_key = f"{uploaded_file.name}:{uploaded_file.size}"

        if file_key not in st.session_state.processed_file_keys:
            with st.spinner(f"Uploading {uploaded_file.name}…"):
                try:
                    resp = requests.post(
                        f"{st.session_state.backend_url}/upload",
                        files={
                            "file": (
                                uploaded_file.name,
                                uploaded_file.getvalue(),
                                uploaded_file.type or "application/octet-stream",
                            )
                        },
                        data={"session_id": st.session_state.session_id},
                        timeout=UPLOAD_TIMEOUT_S,
                    )

                    if resp.ok:
                        data = resp.json()
                        file_type: str = data["file_type"]

                        if file_type == "pdf":
                            chunks = data.get("doc_count", "?")
                            info = f"{chunks} chunk(s) indexed"
                            st.success(f"✅ **{uploaded_file.name}** — {info}")
                        else:
                            info = "OCR complete"
                            extracted: str = data.get("extracted_text") or ""
                            st.success(f"✅ **{uploaded_file.name}** — OCR complete")
                            if extracted:
                                preview = extracted[:300]
                                suffix = "…" if len(extracted) > 300 else ""
                                with st.expander("Preview extracted text"):
                                    st.text(preview + suffix)

                        st.session_state.uploaded_files.append(
                            {
                                "filename": uploaded_file.name,
                                "file_type": file_type,
                                "info": info,
                            }
                        )
                        st.session_state.processed_file_keys.add(file_key)

                    else:
                        try:
                            detail = resp.json().get("detail", resp.text)
                        except Exception:
                            detail = resp.text
                        st.error(f"Upload failed ({resp.status_code}): {detail}")

                except requests.ConnectionError:
                    st.error(
                        "❌ Cannot reach the backend. "
                        "Start it with: `uvicorn backend.main:app --reload`"
                    )
                except requests.Timeout:
                    st.error("⏱️ Upload timed out — the file may be very large.")
                except Exception as exc:
                    st.error(f"Unexpected error: {exc}")

    # ── Uploaded files list ───────────────────────────────────────────────────
    if st.session_state.uploaded_files:
        st.divider()
        st.markdown("**Uploaded this session:**")
        for f in st.session_state.uploaded_files:
            icon = "📄" if f["file_type"] == "pdf" else "🖼️"
            st.markdown(f"{icon} {f['filename']}  \n<small>{f['info']}</small>", unsafe_allow_html=True)

    st.divider()

    # ── Session controls ──────────────────────────────────────────────────────
    col1, col2 = st.columns(2)
    with col1:
        st.caption(f"Session `{st.session_state.session_id[:8]}…`")
    with col2:
        if st.button("🔄 New session", use_container_width=True, help="Clear chat and start fresh"):
            st.session_state.session_id = str(uuid.uuid4())
            st.session_state.messages = []
            st.session_state.uploaded_files = []
            st.session_state.processed_file_keys = set()
            st.rerun()


# ── Main chat area ───────────────────────────────────────────────────────────
st.title("💬 Ask StudyMate")

if not st.session_state.uploaded_files:
    st.info(
        "**Get started:** upload a PDF or image in the sidebar, then ask questions "
        "about it below. You can also ask anything — StudyMate will search the web "
        "when your notes don't cover it.",
        icon="💡",
    )

# ── Render chat history ───────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        if msg["role"] == "assistant":
            if msg.get("sources"):
                _render_sources(msg["sources"])
            if msg.get("trace_url"):
                st.caption(
                    f"[🔍 View LangSmith trace]({msg['trace_url']})"
                )

# ── Chat input ────────────────────────────────────────────────────────────────
if prompt := st.chat_input("Ask a question about your study materials…"):
    # Append and render the user bubble immediately
    st.session_state.messages.append(
        {"role": "user", "content": prompt, "sources": [], "trace_url": None}
    )
    with st.chat_message("user"):
        st.markdown(prompt)

    # Call the backend and stream the assistant bubble
    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            try:
                resp = requests.post(
                    f"{st.session_state.backend_url}/ask",
                    json={
                        "query": prompt,
                        "session_id": st.session_state.session_id,
                    },
                    timeout=ASK_TIMEOUT_S,
                )

                if resp.ok:
                    data = resp.json()
                    answer: str = data["answer"]
                    sources: list[dict[str, Any]] = data.get("sources", [])
                    trace_url: str | None = data.get("trace_url")

                    st.markdown(answer)

                    if sources:
                        _render_sources(sources)

                    if trace_url:
                        st.caption(f"[🔍 View LangSmith trace]({trace_url})")

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": answer,
                            "sources": sources,
                            "trace_url": trace_url,
                        }
                    )

                else:
                    try:
                        detail = resp.json().get("detail", resp.text)
                    except Exception:
                        detail = resp.text
                    error_msg = f"❌ Backend error ({resp.status_code}): {detail}"
                    st.error(error_msg)
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": error_msg,
                            "sources": [],
                            "trace_url": None,
                        }
                    )

            except requests.ConnectionError:
                error_msg = (
                    "❌ Cannot reach the backend. "
                    "Start it with: `uvicorn backend.main:app --reload`"
                )
                st.error(error_msg)
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": error_msg,
                        "sources": [],
                        "trace_url": None,
                    }
                )

            except requests.Timeout:
                error_msg = "⏱️ The request timed out — the pipeline is taking longer than expected."
                st.error(error_msg)
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": error_msg,
                        "sources": [],
                        "trace_url": None,
                    }
                )

            except Exception as exc:
                error_msg = f"❌ Unexpected error: {exc}"
                st.error(error_msg)
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": error_msg,
                        "sources": [],
                        "trace_url": None,
                    }
                )
