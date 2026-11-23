"""Central configuration for ResearchMate.

All tunable values live here and are read from environment variables (or a
local ``.env`` file). Nothing secret is hardcoded. On Streamlit Community
Cloud, values come from the "Secrets" panel; ``app.py`` copies them into
``os.environ`` before ``Settings.from_env()`` is called.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_DOCS_DIR = PROJECT_ROOT / "data" / "sample_documents"
EVAL_DATASET_PATH = PROJECT_ROOT / "data" / "eval_dataset.json"

# load_dotenv never overrides variables that are already set in the real
# environment, so deployment settings always win over a stray .env file.
load_dotenv(PROJECT_ROOT / ".env")


def _env_str(name: str, default: str) -> str:
    return os.environ.get(name, default).strip()


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"Environment variable {name} must be an integer, got {raw!r}") from exc


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"Environment variable {name} must be a number, got {raw!r}") from exc


@dataclass(frozen=True)
class Settings:
    """Immutable application settings."""

    # --- LLM (Groq) ---
    groq_api_key: str = ""
    groq_model: str = "llama-3.3-70b-versatile"
    llm_temperature: float = 0.1
    llm_timeout_seconds: float = 30.0
    llm_max_retries: int = 2

    # --- Embeddings (local) ---
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # --- Chunking and retrieval ---
    chunk_size: int = 1000
    chunk_overlap: int = 200
    top_k: int = 5
    min_relevance_score: float = 0.0  # cosine similarity; 0 disables filtering

    # --- Upload limits ---
    max_file_size_mb: int = 20
    max_files: int = 10
    max_pages_per_file: int = 300
    min_text_chars: int = 100  # below this a PDF is treated as scanned/empty

    # --- Safety / budgets ---
    max_question_chars: int = 1000
    analysis_max_context_chars: int = 12000  # evidence budget for compare/summary/gaps

    @classmethod
    def from_env(cls) -> "Settings":
        settings = cls(
            groq_api_key=_env_str("GROQ_API_KEY", ""),
            groq_model=_env_str("GROQ_MODEL", cls.groq_model),
            llm_temperature=_env_float("LLM_TEMPERATURE", cls.llm_temperature),
            llm_timeout_seconds=_env_float("LLM_TIMEOUT_SECONDS", cls.llm_timeout_seconds),
            llm_max_retries=_env_int("LLM_MAX_RETRIES", cls.llm_max_retries),
            embedding_model=_env_str("EMBEDDING_MODEL", cls.embedding_model),
            chunk_size=_env_int("CHUNK_SIZE", cls.chunk_size),
            chunk_overlap=_env_int("CHUNK_OVERLAP", cls.chunk_overlap),
            top_k=_env_int("TOP_K", cls.top_k),
            min_relevance_score=_env_float("MIN_RELEVANCE_SCORE", cls.min_relevance_score),
            max_file_size_mb=_env_int("MAX_FILE_SIZE_MB", cls.max_file_size_mb),
            max_files=_env_int("MAX_FILES", cls.max_files),
            max_pages_per_file=_env_int("MAX_PAGES_PER_FILE", cls.max_pages_per_file),
            min_text_chars=_env_int("MIN_TEXT_CHARS", cls.min_text_chars),
            max_question_chars=_env_int("MAX_QUESTION_CHARS", cls.max_question_chars),
            analysis_max_context_chars=_env_int(
                "ANALYSIS_MAX_CONTEXT_CHARS", cls.analysis_max_context_chars
            ),
        )
        settings.validate()
        return settings

    def validate(self) -> None:
        """Raise ValueError for settings that would break the pipeline."""
        if self.chunk_size < 100:
            raise ValueError("CHUNK_SIZE must be at least 100 characters")
        if not 0 <= self.chunk_overlap < self.chunk_size:
            raise ValueError("CHUNK_OVERLAP must be >= 0 and smaller than CHUNK_SIZE")
        if not 1 <= self.top_k <= 20:
            raise ValueError("TOP_K must be between 1 and 20")
        if self.max_file_size_mb < 1 or self.max_files < 1:
            raise ValueError("MAX_FILE_SIZE_MB and MAX_FILES must be positive")

    @property
    def max_file_size_bytes(self) -> int:
        return self.max_file_size_mb * 1024 * 1024

    @property
    def has_llm_key(self) -> bool:
        return bool(self.groq_api_key)


def get_settings() -> Settings:
    """Build settings from the current environment (not cached on purpose)."""
    return Settings.from_env()
