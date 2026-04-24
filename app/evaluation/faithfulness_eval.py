from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import re


@dataclass
class FaithfulnessEvalResult:
    question_id: str
    supported_sentence_ratio: float
    unsupported_sentences: list[str] = field(default_factory=list)
    supported_sentences: list[str] = field(default_factory=list)
    citation_present: bool = False
    notes: dict[str, Any] = field(default_factory=dict)


class FaithfulnessEvaluator:
    def evaluate(
        self,
        *,
        question_id: str,
        answer: str,
        retrieved_chunks: list[dict[str, Any]],
        citations: list[dict[str, Any]] | None = None,
    ) -> FaithfulnessEvalResult:
        sentences = self._split_sentences(answer)
        context = " ".join(str(chunk.get("text", "")) for chunk in retrieved_chunks)
        context_norm = self._normalize(context)

        supported: list[str] = []
        unsupported: list[str] = []
        for sentence in sentences:
            norm_sentence = self._normalize(sentence)
            if not norm_sentence:
                continue
            token_overlap = self._token_overlap_ratio(norm_sentence, context_norm)
            if token_overlap >= 0.45 or norm_sentence in context_norm:
                supported.append(sentence)
            else:
                unsupported.append(sentence)

        considered = len(supported) + len(unsupported)
        ratio = len(supported) / considered if considered else 1.0
        return FaithfulnessEvalResult(
            question_id=question_id,
            supported_sentence_ratio=ratio,
            unsupported_sentences=unsupported,
            supported_sentences=supported,
            citation_present=bool(citations),
            notes={
                "sentence_count": considered,
                "retrieved_chunk_count": len(retrieved_chunks),
            },
        )

    def aggregate(self, results: list[FaithfulnessEvalResult]) -> dict[str, Any]:
        if not results:
            return {
                "count": 0,
                "mean_supported_sentence_ratio": 0.0,
                "citation_presence_rate": 0.0,
                "unsupported_answer_rate": 0.0,
            }
        count = len(results)
        return {
            "count": count,
            "mean_supported_sentence_ratio": sum(item.supported_sentence_ratio for item in results) / count,
            "citation_presence_rate": sum(1 for item in results if item.citation_present) / count,
            "unsupported_answer_rate": sum(1 for item in results if item.unsupported_sentences) / count,
        }

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        return [segment.strip() for segment in re.split(r"(?<=[.!?])\s+", text.strip()) if segment.strip()]

    @staticmethod
    def _normalize(text: str) -> str:
        text = text.lower()
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    @classmethod
    def _token_overlap_ratio(cls, sentence: str, context: str) -> float:
        sentence_tokens = set(sentence.split())
        context_tokens = set(context.split())
        if not sentence_tokens:
            return 0.0
        return len(sentence_tokens.intersection(context_tokens)) / len(sentence_tokens)


__all__ = ["FaithfulnessEvaluator", "FaithfulnessEvalResult"]
