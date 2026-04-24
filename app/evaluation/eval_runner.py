"""
eval_runner.py
Runs end-to-end evaluation over the labeled eval set.

This version is robust to:
- missing PROCESSED_METADATA_DIR constant in app.config
- multiple retrieval label formats
- QA service responses returned as either typed objects or dicts
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.config import CONFIG, EVAL_DIR
from app.evaluation.answer_eval import AnswerEvaluator
from app.evaluation.faithfulness_eval import FaithfulnessEvaluator
from app.evaluation.retrieval_eval import RetrievalEvaluator
from app.logging_config import get_logger
from app.utils.file_io import write_json

logger = get_logger(__name__)


class EvalRunner:
    def __init__(self, query_service: Any, retriever: Any) -> None:
        self.query_service = query_service
        self.retriever = retriever
        self.retrieval_evaluator = RetrievalEvaluator()
        self.answer_evaluator = AnswerEvaluator()
        self.faithfulness_evaluator = FaithfulnessEvaluator()

    def run(
        self,
        *,
        questions_path: str | Path | None = None,
        gold_answers_path: str | Path | None = None,
        retrieval_labels_path: str | Path | None = None,
        strategy: str = "similarity",
    ) -> dict[str, Any]:
        questions_path = Path(questions_path or (EVAL_DIR / "questions.json"))
        gold_answers_path = Path(gold_answers_path or (EVAL_DIR / "gold_answers.json"))
        retrieval_labels_path = Path(retrieval_labels_path or (EVAL_DIR / "retrieval_labels.json"))

        questions = self._load_json(questions_path, default=[])
        gold_answers = self._load_json(gold_answers_path, default={})
        retrieval_labels = self._load_json(retrieval_labels_path, default={})

        if not questions:
            logger.warning("No evaluation questions found at %s", questions_path)
            return {}

        retrieval_results = []
        answer_results = []
        faithfulness_results = []
        detailed_results = []

        schema_valid_count = 0

        for q in questions:
            qid = q.get("id") or q.get("question_id") or "unknown"
            question = q.get("question", "")
            ticker = q.get("ticker")
            form_type = q.get("form_type")
            expected_section = q.get("expected_section") or q.get("section_name")
            expects_insufficient_context = (
                q.get("category") == "insufficient_context"
                or bool(q.get("expects_insufficient_context", False))
            )

            logger.info("Evaluating [%s]: %s", qid, question[:60])

            retrieved_chunks = self.retriever.retrieve(
                question,
                ticker=ticker,
                form_type=form_type,
                strategy=strategy,
            )

            qa_response = self._run_query_service(
                question=question,
                ticker=ticker,
                form_type=form_type,
                strategy=strategy,
            )

            answer_text = self._extract_answer(qa_response)
            citations = self._extract_citations(qa_response)
            returned_chunks = self._extract_retrieved_chunks(qa_response) or retrieved_chunks

            schema_valid = self._is_schema_valid(qa_response)
            if schema_valid:
                schema_valid_count += 1

            label = retrieval_labels.get(qid, {})
            normalized_label = self._normalize_retrieval_label(
                label,
                expected_section=expected_section,
            )

            retrieval_eval = self.retrieval_evaluator.evaluate(
                question_id=qid,
                retrieved_chunks=returned_chunks,
                expected_chunk_ids=normalized_label["chunk_ids"],
                expected_parent_document_ids=normalized_label["parent_document_ids"],
                expected_sections=normalized_label["sections"],
            )
            retrieval_results.append(retrieval_eval)

            gold = self._normalize_gold_answer(gold_answers, qid)

            answer_eval = self.answer_evaluator.evaluate(
                question_id=qid,
                answer=answer_text,
                reference_answer=gold["reference_answer"],
                key_points=gold["key_points"],
                unacceptable_claims=gold["unacceptable_claims"],
                expects_insufficient_context=expects_insufficient_context,
            )
            answer_results.append(answer_eval)

            faithfulness_eval = self.faithfulness_evaluator.evaluate(
                question_id=qid,
                answer=answer_text,
                retrieved_chunks=returned_chunks,
                citations=citations,
            )
            faithfulness_results.append(faithfulness_eval)

            detailed_results.append(
                {
                    "question_id": qid,
                    "question": question,
                    "ticker": ticker,
                    "form_type": form_type,
                    "strategy": strategy,
                    "answer": answer_text,
                    "citations": citations,
                    "retrieved_chunks": returned_chunks,
                    "retrieval_eval": self._result_to_dict(retrieval_eval),
                    "answer_eval": self._result_to_dict(answer_eval),
                    "faithfulness_eval": self._result_to_dict(faithfulness_eval),
                    "schema_valid": schema_valid,
                }
            )

        summary = {
            "retrieval": self.retrieval_evaluator.aggregate(retrieval_results),
            "answer": self.answer_evaluator.aggregate(answer_results),
            "faithfulness": self.faithfulness_evaluator.aggregate(faithfulness_results),
            "schema_validity_rate": round(schema_valid_count / max(len(questions), 1), 4),
        }

        results = {
            "strategy": strategy,
            "num_questions": len(questions),
            "summary": summary,
            "details": detailed_results,
        }

        out_dir = self._processed_metadata_dir()
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"eval_results_{strategy}.json"
        write_json(results, out_path)
        logger.info("Saved evaluation results to %s", out_path)

        return results

    @staticmethod
    def _load_json(path: Path, default: Any) -> Any:
        if not path.exists():
            return default
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return default

    @staticmethod
    def _normalize_retrieval_label(
        label: Any,
        *,
        expected_section: str | None = None,
    ) -> dict[str, list[str]]:
        if isinstance(label, dict):
            chunk_ids = label.get("chunk_ids") or label.get("expected_chunk_ids") or []
            parent_document_ids = (
                label.get("parent_document_ids")
                or label.get("expected_parent_document_ids")
                or []
            )
            sections = label.get("sections") or label.get("expected_sections") or []

            if expected_section and not sections:
                sections = [expected_section]

            return {
                "chunk_ids": chunk_ids if isinstance(chunk_ids, list) else [],
                "parent_document_ids": parent_document_ids if isinstance(parent_document_ids, list) else [],
                "sections": sections if isinstance(sections, list) else [],
            }

        if isinstance(label, list):
            return {
                "chunk_ids": label,
                "parent_document_ids": [],
                "sections": [expected_section] if expected_section else [],
            }

        return {
            "chunk_ids": [],
            "parent_document_ids": [],
            "sections": [expected_section] if expected_section else [],
        }

    @staticmethod
    def _normalize_gold_answer(gold_answers: Any, qid: str) -> dict[str, Any]:
        """
        Supports:
        - {"q001": ""}
        - {"q001": {"reference_answer": "...", "key_points": [...], ...}}
        - [{"question_id": "q001", "reference_answer": "...", ...}]
        """
        empty = {
            "reference_answer": "",
            "key_points": [],
            "unacceptable_claims": [],
        }

        if isinstance(gold_answers, dict):
            item = gold_answers.get(qid, "")
            if isinstance(item, str):
                result = dict(empty)
                result["reference_answer"] = item
                return result
            if isinstance(item, dict):
                return {
                    "reference_answer": str(item.get("reference_answer", "")),
                    "key_points": list(item.get("key_points", []) or []),
                    "unacceptable_claims": list(item.get("unacceptable_claims", []) or []),
                }
            return empty

        if isinstance(gold_answers, list):
            for item in gold_answers:
                if str(item.get("question_id")) == qid:
                    return {
                        "reference_answer": str(item.get("reference_answer", "")),
                        "key_points": list(item.get("key_points", []) or []),
                        "unacceptable_claims": list(item.get("unacceptable_claims", []) or []),
                    }

        return empty

    def _run_query_service(
        self,
        *,
        question: str,
        ticker: str | None,
        form_type: str | None,
        strategy: str,
    ) -> Any:
        if hasattr(self.query_service, "answer_question"):
            return self.query_service.answer_question(
                question=question,
                ticker=ticker,
                form_type=form_type,
                strategy=strategy,
            )
        if hasattr(self.query_service, "ask"):
            return self.query_service.ask(
                question=question,
                ticker=ticker,
                form_type=form_type,
                strategy=strategy,
            )
        raise AttributeError("QueryService does not expose answer_question() or ask().")

    @staticmethod
    def _extract_answer(response: Any) -> str:
        if hasattr(response, "answer"):
            return str(response.answer)
        if isinstance(response, dict):
            return str(response.get("answer", ""))
        return str(response)

    @staticmethod
    def _extract_citations(response: Any) -> list[dict[str, Any]]:
        if hasattr(response, "citations"):
            value = response.citations
            return value if isinstance(value, list) else []
        if isinstance(response, dict):
            value = response.get("citations", [])
            return value if isinstance(value, list) else []
        return []

    @staticmethod
    def _extract_retrieved_chunks(response: Any) -> list[dict[str, Any]]:
        if hasattr(response, "retrieved_chunks"):
            value = response.retrieved_chunks
            return value if isinstance(value, list) else []
        if isinstance(response, dict):
            value = response.get("retrieved_chunks", [])
            return value if isinstance(value, list) else []
        return []

    @staticmethod
    def _is_schema_valid(response: Any) -> bool:
        if hasattr(response, "answer") and hasattr(response, "citations"):
            return True
        if hasattr(response, "model_dump"):
            return True
        if isinstance(response, dict):
            return "answer" in response
        return False

    @staticmethod
    def _result_to_dict(obj: Any) -> dict[str, Any]:
        if hasattr(obj, "__dict__"):
            return dict(obj.__dict__)
        if isinstance(obj, dict):
            return obj
        return {"value": str(obj)}

    @staticmethod
    def _processed_metadata_dir() -> Path:
        """
        Avoid depending on a module-level constant that may not exist.
        """
        candidates = [
            getattr(CONFIG, "processed_metadata_dir", None),
            getattr(CONFIG, "metadata_dir", None),
            getattr(CONFIG, "processed_metadata_path", None),
        ]
        for candidate in candidates:
            if candidate:
                return Path(candidate)
        return Path("data/processed/metadata")