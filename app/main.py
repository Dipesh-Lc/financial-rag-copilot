"""
main.py
Entry point: validate LLM config then launch the Gradio app.
"""

from __future__ import annotations

import json
import os
import sys

from app.config import CONFIG, SEED_DIR
from app.logging_config import get_logger

logger = get_logger(__name__)


def _check_llm() -> None:
    """Warn (not crash) if no API key is set for the configured provider."""
    from app.llm.factory import detect_available_providers
    available = detect_available_providers()
    if not available:
        logger.warning(
            "No LLM API key detected. Set ANTHROPIC_API_KEY, OPENAI_API_KEY, or GOOGLE_API_KEY in .env"
        )
    elif CONFIG.llm_provider not in available:
        logger.warning(
            "LLM_PROVIDER=%s but no key found for it. Available: %s",
            CONFIG.llm_provider, available,
        )
    else:
        logger.info("LLM provider: %s / %s ✓", CONFIG.llm_provider, CONFIG.llm_model_name)


def _load_seed_docs() -> list[dict]:
    docs = []
    if not SEED_DIR.exists():
        return docs
    for json_file in sorted(SEED_DIR.rglob("*.json")):
        try:
            docs.append(json.loads(json_file.read_text(encoding="utf-8")))
        except Exception as exc:
            logger.warning("Could not read seed file %s: %s", json_file, exc)
    return docs


def ensure_index_built() -> None:
    """Build the Chroma index from seed docs if the collection is empty."""
    from app.vectorstore.chroma_store import get_vector_store, add_chunks, collection_stats
    from app.processing.section_splitter import SectionAwareChunker

    store = get_vector_store()
    stats = collection_stats(store)
    if stats.get("total_chunks", 0) > 0:
        logger.info("Index already has %d chunks — skipping seed build.", stats["total_chunks"])
        return

    docs = _load_seed_docs()
    if not docs:
        logger.warning("Seed corpus not found at %s — index will be empty.", SEED_DIR)
        return

    logger.info("Index is empty. Building from %d seed documents...", len(docs))
    chunker = SectionAwareChunker()
    chunks = chunker.chunk_and_save(docs)
    add_chunks(chunks, store=store)
    logger.info("Seed index built: %d chunks indexed.", len(chunks))


if __name__ == "__main__":
    _check_llm()
    ensure_index_built()
    from app.ui.gradio_app import build_ui
    demo = build_ui()
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "7860"))
    demo.launch(server_name=host, server_port=port, share=False)
