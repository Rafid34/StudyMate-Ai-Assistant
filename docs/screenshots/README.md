# Screenshots

Add app screenshots here after running the demo. The README references these files:

| File | What to capture |
|---|---|
| `chat_pdf.png` | The main chat window after uploading a PDF — show the sidebar confirming "*N chunk(s) indexed*" and at least one answered question with the **📎 Sources** panel expanded to reveal a "From your notes:" result |
| `sources_panel.png` | The expandable **Sources** panel with both notes results (📄) and web results (🌐) visible in the same response |
| `ocr_result.png` | The sidebar after uploading a photograph — show the **Preview extracted text** expander with OCR output, then a chat answer referencing "From your uploaded image:" |
| `langsmith_trace.png` | A LangSmith trace view showing the `studymate_pipeline` parent run with `router_agent`, `rag_agent` (or `search_agent`), and `synthesis_agent` as labelled child steps |

## Suggested capture workflow

1. Start the backend: `uvicorn backend.main:app --reload`
2. Start the frontend: `streamlit run frontend/app.py`
3. Upload a PDF → take `chat_pdf.png`
4. Ask a question that pulls from both notes and the web → take `sources_panel.png`
5. Upload a JPG photograph → take `ocr_result.png`
6. Open [smith.langchain.com](https://smith.langchain.com/), navigate to the **studymate** project, click any run → take `langsmith_trace.png`
7. Save all files to this directory and remove this note.
