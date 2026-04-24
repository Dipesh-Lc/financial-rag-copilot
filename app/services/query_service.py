"""
query_service.py
High-level Q&A service — ties retriever + FilingQAChain together.

Returns a typed QueryServiceResponse and logs every run to JSONL.
Supports strategy selection: "similarity" | "multi_query" | "parent_doc".
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, TYPE_CHECKING

from app.config import CONFIG
from app.logging_config import get_logger
from app.rag.chains import FilingQAChain, QAChainResult
from app.rag.structured_outputs import QAResponse
if TYPE_CHECKING:
    from app.vectorstore.retriever import FilingRetriever

logger = get_logger(__name__)


@dataclass
class QueryServiceResponse:
    question: str
    answer: str
    citations: list[dict[str, Any]]
    retrieved_chunks: list[dict[str, Any]]

    def to_qa_response(self) -> QAResponse:
        return QAResponse(
            question=self.question,
            answer=self.answer,
            citations=self.citations,
            retrieved_chunks=self.retrieved_chunks,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class QueryService:
    def __init__(
        self,
        retriever: "FilingRetriever" | None = None,
        llm: Any | None = None,
        log_path: str | Path | None = None,
    ) -> None:
        if retriever is not None:
            self.chain = FilingQAChain(retriever=retriever, llm=llm)
        else:
            self.chain = FilingQAChain(llm=llm)
        self._retriever = retriever
        self.log_path = Path(log_path or CONFIG.metadata_dir / "qa_runs.jsonl")

    def answer(
        self,
        question: str,
        ticker: str | None = None,
        form_type: str | None = None,
        filing_date: str | None = None,
        section_name: str | None = None,
        top_k: int | None = None,
        strategy: str = "similarity",
        extra_filters: dict[str, Any] | None = None,
    ) -> QueryServiceResponse:
        if not question.strip():
            return QueryServiceResponse(
                question=question,
                answer="Please enter a question.",
                citations=[],
                retrieved_chunks=[],
            )

        if self._retriever is None:
            return QueryServiceResponse(
                question=question,
                answer="No vector store configured. Please ingest filings first.",
                citations=[],
                retrieved_chunks=[],
            )

        result: QAChainResult = self.chain.run(
            question,
            ticker=ticker,
            form_type=form_type,
            filing_date=filing_date,
            section_name=section_name,
            top_k=top_k,
            strategy=strategy,
            extra_filters=extra_filters,
        )

        payload = self._to_service_response(result)
        self._append_log(
            {
                "question": payload.question,
                "answer": payload.answer,
                "ticker": ticker,
                "form_type": form_type,
                "strategy": strategy,
                "top_k": top_k,
                "num_chunks": len(payload.retrieved_chunks),
            }
        )
        return payload

    def answer_question(self, question: str, **kwargs: Any) -> QueryServiceResponse:
        return self.answer(question, **kwargs)

    def _to_service_response(self, result: QAChainResult) -> QueryServiceResponse:
        return QueryServiceResponse(
            question=result.question,
            answer=result.answer,
            citations=result.citations,
            retrieved_chunks=[
                c if isinstance(c, dict) else c.to_dict()
                for c in result.retrieved_chunks
            ],
        )

    def _append_log(self, record: dict[str, Any]) -> None:
        try:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as exc:
            logger.warning("Failed to write QA log: %s", exc)


__all__ = ["QueryService", "QueryServiceResponse"]