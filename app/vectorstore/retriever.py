"""
retriever.py
Three retrieval strategies, selectable at call time.

  1. similarity   — plain cosine similarity (baseline)
  2. multi_query  — generate N query variants, union results, dedupe
  3. parent_doc   — retrieve child chunks, return full parent-section text

Uses P2's typed RetrievedChunk dataclass while preserving P1's battle-tested
strategy implementations and keeping dict compatibility for the UI layer.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Literal

from langchain_chroma import Chroma

from app.config import DEFAULT_TOP_K, MULTI_QUERY_MAX_VARIANTS
from app.vectorstore.filters import combined_filter
from app.logging_config import get_logger

logger = get_logger(__name__)

RetrievalStrategy = Literal["similarity", "multi_query", "parent_doc"]


# ── Typed result ──────────────────────────────────────────────────────────────

@dataclass
class RetrievedChunk:
    chunk_id: str
    text: str
    score: float | None
    metadata: dict[str, Any]
    parent_document_id: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "score": self.score,
            "metadata": self.metadata,
            "chunk_id": self.chunk_id,
            "parent_document_id": self.parent_document_id,
            "section_name": (self.metadata or {}).get("section_name", ""),
            "ticker": (self.metadata or {}).get("ticker", ""),
            "filing_date": (self.metadata or {}).get("filing_date", ""),
            "form_type": (self.metadata or {}).get("form_type", ""),
            "source_url": (self.metadata or {}).get("source_url", ""),
        }


# ── Query expansion (no LLM dependency) ──────────────────────────────────────

def _expand_query(query: str, max_variants: int = MULTI_QUERY_MAX_VARIANTS) -> list[str]:
    q = query.strip().rstrip("?")
    variants = [query]

    if re.search(r"\bwhat\b", q, re.I):
        variants.append(re.sub(r"\bwhat\b", "describe", q, flags=re.I))
    if re.search(r"\bhow\b", q, re.I):
        variants.append(re.sub(r"\bhow\b", "explain the method for", q, flags=re.I))
    if re.search(r"\brisk\b", q, re.I):
        variants.append(q + " uncertainty exposure")
    if re.search(r"\brevenue|sales|growth\b", q, re.I):
        variants.append(q + " financial performance")

    stopwords = {"what", "is", "are", "the", "a", "an", "of", "in", "for",
                 "does", "did", "how", "why", "when", "its", "their"}
    keywords = [w for w in q.lower().split() if w not in stopwords]
    if keywords:
        variants.append(" ".join(keywords))

    seen: set[str] = set()
    unique: list[str] = []
    for v in variants:
        key = v.lower().strip()
        if key not in seen:
            seen.add(key)
            unique.append(v)
    return unique[:max_variants]


# ── Main retriever ────────────────────────────────────────────────────────────

class FilingRetriever:
    def __init__(self, store: Chroma, top_k: int = DEFAULT_TOP_K) -> None:
        self.store = store
        self.top_k = top_k

    def retrieve(
        self,
        query: str,
        ticker: str | None = None,
        form_type: str | None = None,
        section_name: str | None = None,
        filing_date: str | None = None,
        top_k: int | None = None,
        strategy: RetrievalStrategy = "similarity",
        **_kwargs: Any,
    ) -> list[dict[str, Any]]:
        """
        Retrieve filing chunks using the specified strategy.
        Returns list[dict] for backwards compatibility with the UI.
        """
        k = top_k or self.top_k
        where = combined_filter(
            ticker=ticker, form_type=form_type,
            section_name=section_name, filing_date=filing_date,
        )

        if strategy == "multi_query":
            results = self._multi_query(query, k=k, where=where)
        elif strategy == "parent_doc":
            results = self._parent_doc(query, k=k, where=where)
        else:
            results = self._similarity(query, k=k, where=where)

        logger.info("[%s] Retrieved %d chunks for: %s", strategy, len(results), query[:70])
        return results

    def retrieve_typed(
        self,
        query: str,
        **kwargs: Any,
    ) -> list[RetrievedChunk]:
        """Return typed RetrievedChunk objects instead of dicts."""
        raw = self.retrieve(query, **kwargs)
        return [
            RetrievedChunk(
                chunk_id=r.get("chunk_id", ""),
                text=r.get("text", ""),
                score=r.get("score"),
                metadata=r.get("metadata", {}),
                parent_document_id=r.get("parent_document_id"),
            )
            for r in raw
        ]

    # ── Strategies ────────────────────────────────────────────────────────────

    def _similarity(self, query: str, k: int, where: dict | None) -> list[dict]:
        kwargs: dict[str, Any] = {"k": k}
        if where:
            kwargs["filter"] = where
        docs_scores = self.store.similarity_search_with_score(query, **kwargs)
        return [_doc_to_dict(d, s) for d, s in docs_scores]

    def _multi_query(self, query: str, k: int, where: dict | None) -> list[dict]:
        variants = _expand_query(query)
        seen: dict[str, dict] = {}
        per_k = max(k, 3)
        kwargs: dict[str, Any] = {"k": per_k}
        if where:
            kwargs["filter"] = where

        for variant in variants:
            try:
                docs_scores = self.store.similarity_search_with_score(variant, **kwargs)
            except Exception as exc:
                logger.warning("Multi-query variant failed: %s", exc)
                continue
            for doc, score in docs_scores:
                cid = doc.metadata.get("chunk_id", doc.page_content[:40])
                if cid not in seen or score < seen[cid]["score"]:
                    seen[cid] = _doc_to_dict(doc, score)

        ranked = sorted(seen.values(), key=lambda r: r["score"])
        return ranked[:k]

    def _parent_doc(self, query: str, k: int, where: dict | None) -> list[dict]:
        child_results = self._similarity(query, k=k, where=where)
        if not child_results:
            return child_results

        parent_ids = {r["parent_document_id"] for r in child_results if r["parent_document_id"]}
        if not parent_ids:
            return child_results

        expanded: list[dict] = []
        for parent_id in parent_ids:
            try:
                siblings = self.store.similarity_search(
                    query, k=50, filter={"parent_document_id": parent_id}
                )
                if not siblings:
                    continue
                siblings.sort(key=lambda d: d.metadata.get("chunk_index", 0))
                merged_text = "\n\n".join(d.page_content for d in siblings)
                best_child = next(
                    (r for r in child_results if r["parent_document_id"] == parent_id),
                    child_results[0],
                )
                expanded.append({**best_child, "text": merged_text})
            except Exception as exc:
                logger.warning("Parent-doc fetch failed for %s: %s", parent_id, exc)

        return expanded if expanded else child_results


# ── Helper ────────────────────────────────────────────────────────────────────

def _doc_to_dict(doc: Any, score: float = 0.0) -> dict[str, Any]:
    return {
        "text": doc.page_content,
        "score": round(float(score), 4),
        "metadata": doc.metadata,
        "chunk_id": doc.metadata.get("chunk_id", ""),
        "parent_document_id": doc.metadata.get("parent_document_id", ""),
        "section_name": doc.metadata.get("section_name", ""),
        "ticker": doc.metadata.get("ticker", ""),
        "filing_date": doc.metadata.get("filing_date", ""),
        "form_type": doc.metadata.get("form_type", ""),
        "source_url": doc.metadata.get("source_url", ""),
    }


__all__ = ["FilingRetriever", "RetrievedChunk", "RetrievalStrategy", "_expand_query"]
