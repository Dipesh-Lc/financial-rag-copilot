"""
hf_embeddings.py
LangChain-compatible Hugging Face embedding wrapper.
Loads the model once; exposes embed_documents / embed_query.
"""

from __future__ import annotations

from langchain_huggingface import HuggingFaceEmbeddings

from app.config import EMBEDDING_MODEL_NAME, EMBEDDING_DEVICE
from app.logging_config import get_logger

logger = get_logger(__name__)

_INSTANCE: HuggingFaceEmbeddings | None = None


def get_embeddings(model_name: str = EMBEDDING_MODEL_NAME) -> HuggingFaceEmbeddings:
    """Return a singleton embedding model instance."""
    global _INSTANCE
    if _INSTANCE is None or _INSTANCE.model_name != model_name:
        logger.info("Loading embedding model: %s (device=%s)", model_name, EMBEDDING_DEVICE)
        _INSTANCE = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": EMBEDDING_DEVICE},
            encode_kwargs={"normalize_embeddings": True},
        )
    return _INSTANCE
