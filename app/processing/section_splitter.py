"""
section_splitter.py
Utility to enforce section-boundary chunking: chunks never span two sections.
"""

from __future__ import annotations

from app.processing.chunker import DocumentChunker
from app.logging_config import get_logger

logger = get_logger(__name__)


class SectionAwareChunker(DocumentChunker):
    """
    Extends DocumentChunker to ensure chunks stay within section boundaries.
    Each document should already contain a single section (as produced by FilingLoader).
    This class documents that intent explicitly.
    """

    def chunk_documents(self, docs: list[dict]) -> list[dict]:
        """
        Group docs by section before chunking so chunks never cross section lines.
        """
        by_section: dict[str, list[dict]] = {}
        for doc in docs:
            key = f"{doc['ticker']}_{doc['filing_date']}_{doc['section_name']}"
            by_section.setdefault(key, []).append(doc)

        all_chunks = []
        for group in by_section.values():
            for doc in group:
                all_chunks.extend(self.chunk_document(doc))

        logger.info("Section-aware chunking: %d chunks from %d docs", len(all_chunks), len(docs))
        return all_chunks
