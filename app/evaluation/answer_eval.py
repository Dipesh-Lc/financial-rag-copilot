from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import re


@dataclass
class AnswerEvalResult:
    question_id: str
    answer_relevance: float
    key_point_coverage: float
    insufficient_context_compliance: bool
    hallucination_flag: bool
    matched_key_points: list[str] = field(default_factory=list)
    missing_key_points: list[str] = field(default_factory=list)
    notes: dict[str, Any] = field(default_factory=dict)


class AnswerEvaluator:
    def evaluate(
        self,
        *,
        question_id: str,
        answer: str,
        reference_answer: str | None = None,
        key_points: list[str] | None = None,
        unacceptable_claims: list[str] | None = None,
        expects_insufficient_context: bool = False,
    ) -> AnswerEvalResult:
        answer_norm = self._normalize(answer)
        reference_norm = self._normalize(reference_answer or "")
        key_points = key_points or []
        unacceptable_claims = unacceptable_claims or []

        matched_key_points = [point for point in key_points if self._normalize(point) in answer_norm]
        missing_key_points = [point for point in key_points if point not in matched_key_points]
        key_point_coverage = len(matched_key_points) / len(key_points) if key_points else 0.0

        answer_tokens = set(answer_norm.split())
        reference_tokens = set(reference_norm.split())
        answer_relevance = (
            len(answer_tokens.intersection(reference_tokens)) / len(reference_tokens)
            if reference_tokens
            else (1.0 if answer_norm else 0.0)
        )

        insufficient_markers = [
            "insufficient",
            "not enough evidence",
            "not enough context",
            "cannot determine",
            "unable to determine",
        ]
        insufficient_context_compliance = (
            any(marker in answer_norm for marker in insufficient_markers)
            if expects_insufficient_context
            else True
        )

        hallucination_flag = any(self._normalize(claim) in answer_norm for claim in unacceptable_claims)

        return AnswerEvalResult(
            question_id=question_id,
            answer_relevance=answer_relevance,
            key_point_coverage=key_point_coverage,
            insufficient_context_compliance=insufficient_context_compliance,
            hallucination_flag=hallucination_flag,
            matched_key_points=matched_key_points,
            missing_key_points=missing_key_points,
            notes={
                "answer_length": len(answer),
                "reference_length": len(reference_answer or ""),
            },
        )

    def aggregate(self, results: list[AnswerEvalResult]) -> dict[str, Any]:
        if not results:
            return {
                "count": 0,
                "mean_answer_relevance": 0.0,
                "mean_key_point_coverage": 0.0,
                "insufficient_context_compliance_rate": 0.0,
                "hallucination_rate": 0.0,
            }
        count = len(results)
        return {
            "count": count,
            "mean_answer_relevance": sum(item.answer_relevance for item in results) / count,
            "mean_key_point_coverage": sum(item.key_point_coverage for item in results) / count,
            "insufficient_context_compliance_rate": sum(1 for item in results if item.insufficient_context_compliance) / count,
            "hallucination_rate": sum(1 for item in results if item.hallucination_flag) / count,
        }

    @staticmethod
    def _normalize(text: str) -> str:
        text = text.lower()
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text


__all__ = ["AnswerEvaluator", "AnswerEvalResult"]
