"""
doc_registry.py
In-memory (and optional JSON-backed) registry of ingested documents and chunks.
"""

from __future__ import annotations

import json
from pathlib import Path

from app.config import METADATA_DIR
from app.logging_config import get_logger

logger = get_logger(__name__)
_REGISTRY_FILE = METADATA_DIR / "doc_registry.json"


class DocRegistry:
    def __init__(self):
        self._docs: dict[str, dict] = {}
        self._chunks: dict[str, dict] = {}
        self._load()

    def register_document(self, doc: dict) -> None:
        self._docs[doc["document_id"]] = doc
        self._save()

    def register_chunk(self, chunk: dict) -> None:
        self._chunks[chunk["chunk_id"]] = chunk

    def get_document(self, doc_id: str) -> dict | None:
        return self._docs.get(doc_id)

    def get_chunk(self, chunk_id: str) -> dict | None:
        return self._chunks.get(chunk_id)

    def list_documents(self, ticker: str | None = None) -> list[dict]:
        docs = list(self._docs.values())
        if ticker:
            docs = [d for d in docs if d.get("ticker") == ticker.upper()]
        return docs

    def _save(self) -> None:
        METADATA_DIR.mkdir(parents=True, exist_ok=True)
        _REGISTRY_FILE.write_text(json.dumps(self._docs, indent=2), encoding="utf-8")

    def _load(self) -> None:
        if _REGISTRY_FILE.exists():
            self._docs = json.loads(_REGISTRY_FILE.read_text(encoding="utf-8"))
