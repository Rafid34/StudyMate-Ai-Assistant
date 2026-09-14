import os
import warnings
from pathlib import Path

from dotenv import load_dotenv

# Resolve studymate/.env relative to this file regardless of working directory.
_ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=_ENV_PATH)

GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
LANGCHAIN_API_KEY: str = os.getenv("LANGCHAIN_API_KEY", "")

LANGCHAIN_TRACING_V2: str = os.getenv("LANGCHAIN_TRACING_V2", "true")
LANGCHAIN_PROJECT: str = os.getenv("LANGCHAIN_PROJECT", "studymate")

TAVILY_API_KEY: str | None = os.getenv("TAVILY_API_KEY") or None
SEARCH_BACKEND: str = os.getenv("SEARCH_BACKEND", "gemini").lower()

CHROMA_DB_PATH: str = os.getenv("CHROMA_DB_PATH", "./data/chroma_db")

# Propagate into os.environ so LangChain auto-detects these settings.
os.environ["LANGCHAIN_TRACING_V2"] = LANGCHAIN_TRACING_V2
os.environ["LANGCHAIN_PROJECT"] = LANGCHAIN_PROJECT
os.environ["LANGCHAIN_API_KEY"] = LANGCHAIN_API_KEY

# Warn rather than raise so the server can still start without full config.
_missing = [
    name
    for name, value in [
        ("GEMINI_API_KEY", GEMINI_API_KEY),
        ("LANGCHAIN_API_KEY", LANGCHAIN_API_KEY),
    ]
    if not value
]

if _missing:
    warnings.warn(
        f"StudyMate: the following required environment variables are not set: "
        f"{', '.join(_missing)}. "
        "Copy .env.example to .env and fill in the missing values.",
        stacklevel=1,
    )
