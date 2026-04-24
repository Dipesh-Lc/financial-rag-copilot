"""
ingestion_service.py
End-to-end ingestion: download → parse → clean → chunk → embed → index.
Accepts an existing Chroma store so it shares the same collection as the
QueryService and MemoService (no duplicate store objects).
"""

from __future__ import annotations

from langchain_chroma import Chroma

from app.ingestion.loaders import FilingLoader
from app.processing.section_splitter import SectionAwareChunker
from app.processing.doc_registry import DocRegistry
from app.vectorstore.chroma_store import get_vector_store, add_chunks
from app.logging_config import get_logger

logger = get_logger(__name__)


class IngestionService:
    def __init__(self, store: Chroma | None = None):
        self.loader = FilingLoader()
        self.chunker = SectionAwareChunker()
        self.registry = DocRegistry()
        self.store = store or get_vector_store()

    def ingest_ticker(
        self,
        ticker: str,
        form_types: list[str] | None = None,
        max_filings: int = 2,
    ) -> dict:
        """
        Full pipeline for one ticker.
        Returns stats dict: {ticker, docs, chunks, already_indexed}.
        """
        docs = self.loader.load_ticker(
            ticker, form_types=form_types, max_filings=max_filings
        )

        # Register docs and filter out already-indexed ones
        new_docs = []
        for doc in docs:
            if self.registry.get_document(doc["document_id"]) is None:
                self.registry.register_document(doc)
                new_docs.append(doc)
            else:
                logger.debug("Already registered: %s", doc["document_id"])

        chunks = self.chunker.chunk_and_save(new_docs) if new_docs else []

        for chunk in chunks:
            self.registry.register_chunk(chunk)

        if chunks:
            add_chunks(chunks, store=self.store)

        logger.info(
            "Ingested %s: %d docs (%d new) → %d chunks → indexed",
            ticker, len(docs), len(new_docs), len(chunks),
        )
        return {
            "ticker": ticker,
            "docs": len(docs),
            "new_docs": len(new_docs),
            "chunks": len(chunks),
        }

    def ingest_tickers(self, tickers: list[str], **kwargs) -> list[dict]:
        return [self.ingest_ticker(t, **kwargs) for t in tickers]
