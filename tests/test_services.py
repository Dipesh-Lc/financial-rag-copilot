"""
test_services.py
Tests for QueryService and MemoService typed response contracts.
"""

from __future__ import annotations

import pytest

from app.services.query_service import QueryService, QueryServiceResponse
from app.services.memo_service import MemoService, MemoServiceResponse
from app.rag.structured_outputs import FinancialMemo, RiskItem


class _FakeLLM:
    def __init__(self, content="Answer with citation [C1]"):
        self.content = content

    def invoke(self, messages, config=None):
        return type("R", (), {"content": self.content})()


class _FakeRetriever:
    CHUNK = {
        "chunk_id": "c1",
        "text": "Apple discloses significant regulatory risks.",
        "score": 0.1,
        "metadata": {"ticker": "AAPL", "section_name": "Risk Factors", "filing_date": "2024-09-28", "form_type": "10-K"},
        "parent_document_id": None,
        "section_name": "Risk Factors",
        "ticker": "AAPL",
        "filing_date": "2024-09-28",
        "form_type": "10-K",
        "source_url": "",
    }

    def retrieve(self, *_, **__):
        return [self.CHUNK]


class TestQueryServiceResponse:
    def _service(self):
        return QueryService(retriever=_FakeRetriever(), llm=_FakeLLM(), log_path="/tmp/qa_test.jsonl")

    def test_returns_typed_response(self):
        svc = self._service()
        result = svc.answer("What are the risks?", ticker="AAPL")
        assert isinstance(result, QueryServiceResponse)

    def test_answer_is_non_empty_string(self):
        svc = self._service()
        result = svc.answer("What are the risks?")
        assert isinstance(result.answer, str)
        assert len(result.answer) > 0

    def test_citations_is_list(self):
        svc = self._service()
        result = svc.answer("What are the risks?")
        assert isinstance(result.citations, list)

    def test_retrieved_chunks_is_list(self):
        svc = self._service()
        result = svc.answer("What are the risks?")
        assert isinstance(result.retrieved_chunks, list)
        assert len(result.retrieved_chunks) > 0

    def test_empty_question_returns_prompt(self):
        svc = self._service()
        result = svc.answer("   ")
        assert "Please enter" in result.answer

    def test_answer_question_alias_works(self):
        svc = self._service()
        result = svc.answer_question("What are the risks?")
        assert isinstance(result, QueryServiceResponse)

    def test_to_qa_response_conversion(self):
        from app.rag.structured_outputs import QAResponse
        svc = self._service()
        result = svc.answer("What are the risks?")
        qa = result.to_qa_response()
        assert isinstance(qa, QAResponse)
        assert qa.question == result.question


class TestMemoServiceResponse:
    def _make_memo(self):
        return FinancialMemo(
            company="Apple Inc.", ticker="AAPL", form_type="10-K",
            filing_date="2024-09-28", summary="Apple faces regulatory risks.",
            key_risks=[RiskItem(risk_title="Regulatory risk", description="Significant regulatory exposure.", severity="high")],
            confidence_score=0.8,
            limitations=["Limited prior filing context."],
        )

    def test_format_display_ok_memo(self):
        from dataclasses import asdict
        memo = self._make_memo()
        resp = MemoServiceResponse(
            ok=True, memo=memo.model_dump(),
            executive_summary=memo.summary,
            key_risks=[r.model_dump() for r in memo.key_risks],
            key_changes=[], supporting_evidence=[],
            confidence_score=0.8, limitations=["Limited context."],
            raw_json=memo.model_dump_json(),
            validation_error=None,
            retrieved_chunks=[], citations=[],
            attempts=1, selected_filing_date="2024-09-28", prior_filing_date=None,
        )
        narrative, raw = resp.format_display()
        assert "Apple faces regulatory risks" in narrative
        assert "Regulatory risk" in narrative
        assert isinstance(raw, str)

    def test_format_display_failed_memo(self):
        resp = MemoServiceResponse(
            ok=False, memo=None, executive_summary=None,
            key_risks=[], key_changes=[], supporting_evidence=[],
            confidence_score=None, limitations=[],
            raw_json=None, validation_error="JSON parse error",
            retrieved_chunks=[], citations=[],
            attempts=2, selected_filing_date=None, prior_filing_date=None,
        )
        narrative, raw = resp.format_display()
        assert "failed" in narrative.lower()
        assert "JSON parse error" in narrative

    def test_list_available_filings_empty(self, tmp_path, monkeypatch):
        from types import SimpleNamespace
        monkeypatch.setattr("app.services.memo_service.CONFIG", SimpleNamespace(chunks_dir=tmp_path / "chunks", metadata_dir=tmp_path))
        svc = MemoService(log_path=str(tmp_path / "memo.jsonl"))
        result = svc.list_available_filings(ticker="AAPL")
        assert result == []
