"""
config.py
Central configuration for the AI Financial Intelligence Copilot.

Design: module-level constants (importable anywhere) + a frozen AppConfig
dataclass (CONFIG singleton) so both styles work side-by-side.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ── Root paths ────────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / os.getenv("DATA_DIR", "data")

RAW_DIR = DATA_DIR / "raw"
FILINGS_DIR = RAW_DIR / "filings"
TRANSCRIPTS_DIR = RAW_DIR / "transcripts"
MACRO_DIR = RAW_DIR / "macro"

PROCESSED_DIR = DATA_DIR / "processed"
CLEANED_DOCS_DIR = PROCESSED_DIR / "cleaned_docs"
CHUNKS_DIR = PROCESSED_DIR / "chunks"
METADATA_DIR = PROCESSED_DIR / "metadata"

EVAL_DIR = DATA_DIR / "eval"
CHROMA_PERSIST_DIR = DATA_DIR / os.getenv("CHROMA_PERSIST_DIR", "vectorstore/chroma")

LOG_DIR = ROOT_DIR / "logs"

# ── LLM — API keys ────────────────────────────────────────────────────────────
ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")

# ── LLM — provider / model ────────────────────────────────────────────────────
# Supported providers: "anthropic" | "openai" | "gemini"
LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "anthropic")


# Default models
_DEFAULT_MODELS: dict[str, str] = {
    "anthropic": "claude-sonnet-4.6",
    "openai": "gpt-5.4-mini",
    "gemini": "gemini-3-flash-preview",
}

LLM_MODEL_NAME: str = os.getenv(
    "LLM_MODEL_NAME",
    _DEFAULT_MODELS.get(LLM_PROVIDER.lower(), "gemini-3-flash-preview"),
)
LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.0"))
LLM_MAX_TOKENS: int = int(os.getenv("LLM_MAX_TOKENS", "2048"))

# ── Embeddings ────────────────────────────────────────────────────────────────
EMBEDDING_MODEL_NAME: str = os.getenv(
    "EMBEDDING_MODEL_NAME",
    "sentence-transformers/all-MiniLM-L6-v2",
)
EMBEDDING_DEVICE: str = os.getenv("EMBEDDING_DEVICE", "cpu")

# ── Chroma ────────────────────────────────────────────────────────────────────
CHROMA_COLLECTION_NAME: str = os.getenv("CHROMA_COLLECTION_NAME", "sec_filings")

# ── Retrieval ─────────────────────────────────────────────────────────────────
DEFAULT_TOP_K: int = int(os.getenv("DEFAULT_TOP_K", "5"))
RETRIEVAL_SCORE_THRESHOLD: float = float(os.getenv("RETRIEVAL_SCORE_THRESHOLD", "0.0"))

# Smart retrieval routing flags
ENABLE_RETRIEVAL_UPGRADES: bool = os.getenv("ENABLE_RETRIEVAL_UPGRADES", "true").lower() == "true"
ENABLE_MULTI_QUERY: bool = os.getenv("ENABLE_MULTI_QUERY", "true").lower() == "true"
ENABLE_PARENT_DOCUMENT_RETRIEVAL: bool = os.getenv("ENABLE_PARENT_DOCUMENT_RETRIEVAL", "true").lower() == "true"
MULTI_QUERY_MAX_VARIANTS: int = int(os.getenv("MULTI_QUERY_MAX_VARIANTS", "4"))
PARENT_CONTEXT_WINDOW: int = int(os.getenv("PARENT_CONTEXT_WINDOW", "1"))
PARENT_CONTEXT_TRIGGER_TOKENS: int = int(os.getenv("PARENT_CONTEXT_TRIGGER_TOKENS", "220"))

# ── Memo generation ───────────────────────────────────────────────────────────
MEMO_VALIDATION_RETRIES: int = int(os.getenv("MEMO_VALIDATION_RETRIES", "2"))

# ── Chunking ──────────────────────────────────────────────────────────────────
CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "150"))

# ── EDGAR ─────────────────────────────────────────────────────────────────────
EDGAR_USER_AGENT: str = os.getenv("EDGAR_USER_AGENT", "AI Copilot research@example.com")
EDGAR_BASE_URL: str = "https://data.sec.gov"
EDGAR_RATE_LIMIT_DELAY: float = float(os.getenv("EDGAR_RATE_LIMIT_DELAY", "0.5"))

DEFAULT_TICKERS: list[str] = ["AAPL", "MSFT", "GOOGL"]
DEFAULT_FORM_TYPES: list[str] = ["10-K", "10-Q"]
DEFAULT_FILING_COUNT: int = 2

PRIORITY_SECTIONS: list[str] = [
    "Risk Factors",
    "Management's Discussion and Analysis",
    "Business",
    "Notes to Financial Statements",
]

# ── App env ───────────────────────────────────────────────────────────────────
APP_ENV: str = os.getenv("APP_ENV", "development")
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
VERSION: str = "1.0.0"


# ── AppConfig dataclass singleton ─────────────────────────────────────────────

@dataclass(frozen=True)
class AppConfig:
    """Frozen dataclass exposing every setting as a typed attribute."""

    # paths
    root_dir: Path = ROOT_DIR
    data_dir: Path = DATA_DIR
    raw_dir: Path = RAW_DIR
    filings_dir: Path = FILINGS_DIR
    transcripts_dir: Path = TRANSCRIPTS_DIR
    macro_dir: Path = MACRO_DIR
    processed_dir: Path = PROCESSED_DIR
    cleaned_docs_dir: Path = CLEANED_DOCS_DIR
    chunks_dir: Path = CHUNKS_DIR
    metadata_dir: Path = METADATA_DIR
    eval_dir: Path = EVAL_DIR
    chroma_persist_dir: Path = CHROMA_PERSIST_DIR
    log_dir: Path = LOG_DIR

    # LLM
    anthropic_api_key: str = ANTHROPIC_API_KEY
    openai_api_key: str = OPENAI_API_KEY
    google_api_key: str = GOOGLE_API_KEY
    llm_provider: str = LLM_PROVIDER
    llm_model_name: str = LLM_MODEL_NAME
    llm_temperature: float = LLM_TEMPERATURE
    llm_max_tokens: int = LLM_MAX_TOKENS

    # embeddings
    embedding_model_name: str = EMBEDDING_MODEL_NAME
    embedding_device: str = EMBEDDING_DEVICE

    # chroma
    chroma_collection_name: str = CHROMA_COLLECTION_NAME

    # retrieval
    default_top_k: int = DEFAULT_TOP_K
    retrieval_score_threshold: float = RETRIEVAL_SCORE_THRESHOLD
    enable_retrieval_upgrades: bool = ENABLE_RETRIEVAL_UPGRADES
    enable_multi_query: bool = ENABLE_MULTI_QUERY
    enable_parent_document_retrieval: bool = ENABLE_PARENT_DOCUMENT_RETRIEVAL
    multi_query_max_variants: int = MULTI_QUERY_MAX_VARIANTS
    parent_context_window: int = PARENT_CONTEXT_WINDOW
    parent_context_trigger_tokens: int = PARENT_CONTEXT_TRIGGER_TOKENS

    # memo
    memo_validation_retries: int = MEMO_VALIDATION_RETRIES

    # chunking
    chunk_size: int = CHUNK_SIZE
    chunk_overlap: int = CHUNK_OVERLAP

    # edgar
    edgar_user_agent: str = EDGAR_USER_AGENT
    edgar_rate_limit_delay: float = EDGAR_RATE_LIMIT_DELAY

    # misc
    app_env: str = APP_ENV
    log_level: str = LOG_LEVEL
    version: str = VERSION


CONFIG = AppConfig()


def ensure_directories() -> None:
    """Create all required data directories."""
    for path in [
        FILINGS_DIR, TRANSCRIPTS_DIR, MACRO_DIR,
        CLEANED_DOCS_DIR, CHUNKS_DIR, METADATA_DIR,
        EVAL_DIR, CHROMA_PERSIST_DIR, LOG_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)


# Auto-create on import so nothing fails at startup
ensure_directories()
