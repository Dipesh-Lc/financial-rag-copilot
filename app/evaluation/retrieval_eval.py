from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class RetrievalEvalResult:
    question_id: str
    retrieved_chunk_ids: list[str]
    retrieved_sections: list[str]
    retrieved_parent_document_ids: list[str]
    hit_at_k: bool
    section_hit: bool
    parent_hit: bool
    precision_at_k: float
    recall_at_k: float
    details: dict[str, Any] = field(default_factory=dict)


class RetrievalEvaluator:
    def evaluate(
        self,
        *,
        question_id: str,
        retrieved_chunks: list[dict[str, Any]],
        expected_chunk_ids: list[str] | None = None,
        expected_parent_document_ids: list[str] | None = None,
        expected_sections: list[str] | None = None,
    ) -> RetrievalEvalResult:
        expected_chunk_ids = expected_chunk_ids or []
        expected_parent_document_ids = expected_parent_document_ids or []
        expected_sections = expected_sections or []

        retrieved_chunk_ids = [str(chunk.get("chunk_id", "")) for chunk in retrieved_chunks if chunk.get("chunk_id")]
        retrieved_parent_ids = [
            str(chunk.get("parent_document_id") or chunk.get("metadata", {}).get("parent_document_id", ""))
            for chunk in retrieved_chunks
            if chunk.get("parent_document_id") or chunk.get("metadata", {}).get("parent_document_id")
        ]
        retrieved_sections = [
            str(chunk.get("metadata", {}).get("section_name") or chunk.get("section_name", ""))
            for chunk in retrieved_chunks
            if chunk.get("metadata", {}).get("section_name") or chunk.get("section_name")
        ]

        expected_chunk_set = set(expected_chunk_ids)
        expected_parent_set = set(expected_parent_document_ids)
        expected_section_set = {section.lower() for section in expected_sections}

        retrieved_chunk_set = set(retrieved_chunk_ids)
        retrieved_parent_set = set(retrieved_parent_ids)
        retrieved_section_set = {section.lower() for section in retrieved_sections}

        chunk_matches = expected_chunk_set.intersection(retrieved_chunk_set)
        parent_matches = expected_parent_set.intersection(retrieved_parent_set)
        section_matches = expected_section_set.intersection(retrieved_section_set)

        precision = len(chunk_matches) / len(retrieved_chunk_ids) if retrieved_chunk_ids and expected_chunk_ids else 0.0
        recall = len(chunk_matches) / len(expected_chunk_ids) if expected_chunk_ids else 0.0
        hit = bool(chunk_matches) if expected_chunk_ids else bool(parent_matches or section_matches)

        return RetrievalEvalResult(
            question_id=question_id,
            retrieved_chunk_ids=retrieved_chunk_ids,
            retrieved_sections=retrieved_sections,
            retrieved_parent_document_ids=retrieved_parent_ids,
            hit_at_k=hit,
            section_hit=bool(section_matches),
            parent_hit=bool(parent_matches),
            precision_at_k=precision,
            recall_at_k=recall,
            details={
                "matching_chunk_ids": sorted(chunk_matches),
                "matching_parent_document_ids": sorted(parent_matches),
                "matching_sections": sorted(section_matches),
            },
        )

    def aggregate(self, results: list[RetrievalEvalResult]) -> dict[str, Any]:
        if not results:
            return {
                "count": 0,
                "retrieval_hit_rate": 0.0,
                "section_hit_rate": 0.0,
                "parent_hit_rate": 0.0,
                "mean_precision_at_k": 0.0,
                "mean_recall_at_k": 0.0,
            }
        count = len(results)
        return {
            "count": count,
            "retrieval_hit_rate": sum(1 for item in results if item.hit_at_k) / count,
            "section_hit_rate": sum(1 for item in results if item.section_hit) / count,
            "parent_hit_rate": sum(1 for item in results if item.parent_hit) / count,
            "mean_precision_at_k": sum(item.precision_at_k for item in results) / count,
            "mean_recall_at_k": sum(item.recall_at_k for item in results) / count,
        }


__all__ = ["RetrievalEvaluator", "RetrievalEvalResult"]
