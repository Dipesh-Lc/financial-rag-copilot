#!/usr/bin/env python

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import CHROMA_PERSIST_DIR, CLEANED_DOCS_DIR, DEFAULT_TICKERS
from app.logging_config import get_logger
from app.processing.section_splitter import SectionAwareChunker
from app.vectorstore.chroma_store import add_chunks, collection_stats, get_vector_store

logger = get_logger("build_index")


def load_cleaned_docs(tickers: list[str]) -> list[dict]:
    docs = []
    for ticker in tickers:
        ticker_dir = CLEANED_DOCS_DIR / ticker.upper()
        if not ticker_dir.exists():
            logger.warning("No cleaned docs for %s at %s", ticker, ticker_dir)
            continue
        for json_file in ticker_dir.rglob("*.json"):
            docs.append(json.loads(json_file.read_text(encoding="utf-8")))
    return docs


def reset_chroma_dir() -> None:
    if CHROMA_PERSIST_DIR.exists():
        shutil.rmtree(CHROMA_PERSIST_DIR)
        logger.info("Deleted existing Chroma directory: %s", CHROMA_PERSIST_DIR)
    CHROMA_PERSIST_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Chroma index from cleaned docs")
    parser.add_argument("--tickers", nargs="+", default=DEFAULT_TICKERS)
    parser.add_argument("--reset", action="store_true", help="Delete existing Chroma index before rebuilding")
    args = parser.parse_args()

    if args.reset:
        reset_chroma_dir()

    docs = load_cleaned_docs(args.tickers)
    logger.info("Loaded %d documents", len(docs))

    if not docs:
        logger.warning("No cleaned documents found. Run build_corpus.py first.")
        return

    chunker = SectionAwareChunker()
    chunks = chunker.chunk_and_save(docs)
    logger.info("Created %d chunks", len(chunks))

    store = get_vector_store()
    add_chunks(chunks, store=store)

    stats = collection_stats(store)
    logger.info("Index stats: %s", stats)


if __name__ == "__main__":
    main()
