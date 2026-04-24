"""
chunker.py
Split cleaned document text into overlapping chunks with full metadata.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterator

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import CHUNK_SIZE, CHUNK_OVERLAP, CHUNKS_DIR
from app.logging_config import get_logger

logger = get_logger(__name__)


def _chunk_id(parent_id: str, index: int) -> str:
    raw = f"{parent_id}_chunk_{index}"
    return hashlib.md5(raw.encode()).hexdigest()[:12]


class DocumentChunker:
    def __init__(self, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def chunk_document(self, doc: dict) -> list[dict]:
        """Split a single cleaned document dict into chunk dicts."""
        texts = self.splitter.split_text(doc["text"])
        chunks = []
        for i, text in enumerate(texts):
            chunk = {
                "chunk_id": _chunk_id(doc["document_id"], i),
                "parent_document_id": doc["document_id"],
                "chunk_index": i,
                "ticker": doc["ticker"],
                "company_name": doc.get("company_name", doc["ticker"]),
                "form_type": doc["form_type"],
                "filing_date": doc["filing_date"],
                "section_name": doc["section_name"],
                "source_url": doc.get("source_url", ""),
                "text": text,
            }
            chunks.append(chunk)
        return chunks

    def chunk_documents(self, docs: list[dict]) -> list[dict]:
        all_chunks = []
        for doc in docs:
            all_chunks.extend(self.chunk_document(doc))
        logger.info("Created %d chunks from %d documents", len(all_chunks), len(docs))
        return all_chunks

    def chunk_and_save(self, docs: list[dict]) -> list[dict]:
        chunks = self.chunk_documents(docs)
        for chunk in chunks:
            out_dir = CHUNKS_DIR / chunk["ticker"] / chunk["form_type"]
            out_dir.mkdir(parents=True, exist_ok=True)
            path = out_dir / f"{chunk['chunk_id']}.json"
            path.write_text(json.dumps(chunk, indent=2), encoding="utf-8")
        return chunks
