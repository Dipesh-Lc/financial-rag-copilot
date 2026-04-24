from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.config import CONFIG
from app.utils.json_utils import load_json


@dataclass
class EvalQuestion:
    question_id: str
    question: str
    ticker: str | None = None
    form_type: str | None = None
    filing_date: str | None = None
    section_name: str | None = None
    category: str = "factual"
    expects_insufficient_context: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class GoldAnswer:
    question_id: str
    reference_answer: str
    key_points: list[str] = field(default_factory=list)
    unacceptable_claims: list[str] = field(default_factory=list)


@dataclass
class RetrievalLabel:
    question_id: str
    relevant_chunk_ids: list[str] = field(default_factory=list)
    relevant_parent_document_ids: list[str] = field(default_factory=list)
    relevant_sections: list[str] = field(default_factory=list)


@dataclass
class EvalExample:
    question: EvalQuestion
    gold_answer: GoldAnswer | None = None
    retrieval_label: RetrievalLabel | None = None


class EvalDataset:
    def __init__(self, examples: list[EvalExample]) -> None:
        self.examples = examples

    @classmethod
    def from_directory(cls, directory: str | Path | None = None) -> "EvalDataset":
        eval_dir = Path(directory or CONFIG.eval_data_dir)
        questions_payload = load_json(eval_dir / "questions.json") or []
        answers_payload = load_json(eval_dir / "gold_answers.json") or []
        labels_payload = load_json(eval_dir / "retrieval_labels.json") or []

        answers_by_id = {
            str(item.get("question_id")): GoldAnswer(
                question_id=str(item.get("question_id")),
                reference_answer=str(item.get("reference_answer", "")),
                key_points=list(item.get("key_points", []) or []),
                unacceptable_claims=list(item.get("unacceptable_claims", []) or []),
            )
            for item in answers_payload
        }
        labels_by_id = {
            str(item.get("question_id")): RetrievalLabel(
                question_id=str(item.get("question_id")),
                relevant_chunk_ids=list(item.get("relevant_chunk_ids", []) or []),
                relevant_parent_document_ids=list(item.get("relevant_parent_document_ids", []) or []),
                relevant_sections=list(item.get("relevant_sections", []) or []),
            )
            for item in labels_payload
        }

        examples: list[EvalExample] = []
        for item in questions_payload:
            question = EvalQuestion(
                question_id=str(item.get("question_id")),
                question=str(item.get("question", "")),
                ticker=item.get("ticker"),
                form_type=item.get("form_type"),
                filing_date=item.get("filing_date"),
                section_name=item.get("section_name"),
                category=str(item.get("category", "factual")),
                expects_insufficient_context=bool(item.get("expects_insufficient_context", False)),
                metadata=dict(item.get("metadata", {}) or {}),
            )
            qid = question.question_id
            examples.append(
                EvalExample(
                    question=question,
                    gold_answer=answers_by_id.get(qid),
                    retrieval_label=labels_by_id.get(qid),
                )
            )
        return cls(examples)

    def to_dict(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for example in self.examples:
            rows.append(
                {
                    "question_id": example.question.question_id,
                    "question": example.question.question,
                    "ticker": example.question.ticker,
                    "form_type": example.question.form_type,
                    "filing_date": example.question.filing_date,
                    "section_name": example.question.section_name,
                    "category": example.question.category,
                    "expects_insufficient_context": example.question.expects_insufficient_context,
                    "gold_answer": example.gold_answer.reference_answer if example.gold_answer else None,
                    "key_points": example.gold_answer.key_points if example.gold_answer else [],
                    "relevant_chunk_ids": example.retrieval_label.relevant_chunk_ids if example.retrieval_label else [],
                    "relevant_sections": example.retrieval_label.relevant_sections if example.retrieval_label else [],
                }
            )
        return rows


__all__ = [
    "EvalDataset",
    "EvalExample",
    "EvalQuestion",
    "GoldAnswer",
    "RetrievalLabel",
]
