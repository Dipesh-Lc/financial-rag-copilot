"""
test_chains.py
Tests for FilingQAChain — smart routing heuristics and response handling.
"""

from __future__ import annotations

import pytest

from app.rag.chains import FilingQAChain


class _FakeLLM:
    """Minimal LLM stub that returns a fixed answer."""

    def __init__(self, answer: str = "Test answer [C1]"):
        self.answer = answer
        self.calls: list[dict] = []

    def invoke(self, messages, config=None):
        self.calls.append({"messages": messages})
        return type("Response", (), {"content": self.answer})()


class _FakeRetriever:
    """Stub retriever returning a fixed chunk list."""

    def __init__(self, chunks=None):
        self.chunks = chunks or [
            {
                "chunk_id": "chunk-001",
                "text": "Apple faces significant cybersecurity risks.",
                "score": 0.12,
                "metadata": {"ticker": "AAPL", "section_name": "Risk Factors", "filing_date": "2024-09-28", "form_type": "10-K"},
                "parent_document_id": "parent-001",
                "section_name": "Risk Factors",
                "ticker": "AAPL",
                "filing_date": "2024-09-28",
                "form_type": "10-K",
                "source_url": "",
            }
        ]
        self.last_call: dict = {}

    def retrieve(self, query, *, ticker=None, form_type=None, top_k=None, strategy="similarity", **_):
        self.last_call = {"query": query, "strategy": strategy, "ticker": ticker}
        return self.chunks


class TestFilingQAChainRouting:
    def _make_chain(self, enable_upgrades=True):
        llm = _FakeLLM()
        retriever = _FakeRetriever()
        chain = FilingQAChain(retriever=retriever, llm=llm, enable_retrieval_upgrades=enable_upgrades)
        return chain, llm, retriever

    def test_similarity_strategy_passed_through_when_upgrades_disabled(self):
        chain, _, retriever = self._make_chain(enable_upgrades=False)
        chain.run("What are the risks?", strategy="similarity")
        assert retriever.last_call["strategy"] == "similarity"

    def test_broad_question_upgrades_to_multi_query(self):
        chain, _, retriever = self._make_chain(enable_upgrades=True)
        chain.run("Give me a broad summary of key material risks", strategy="similarity")
        assert retriever.last_call["strategy"] == "multi_query"

    def test_comparison_question_upgrades_to_parent_doc(self):
        chain, _, retriever = self._make_chain(enable_upgrades=True)
        chain.run("How did revenue change versus the prior period?", strategy="similarity")
        assert retriever.last_call["strategy"] in {"multi_query", "parent_doc"}

    def test_specific_question_stays_similarity(self):
        chain, _, retriever = self._make_chain(enable_upgrades=True)
        chain.run("What is Apple's EDGAR CIK number?", strategy="similarity")
        assert retriever.last_call["strategy"] == "similarity"

    def test_explicit_strategy_not_overridden_by_routing(self):
        chain, _, retriever = self._make_chain(enable_upgrades=True)
        chain.run("Summary of key risks", strategy="parent_doc")
        assert retriever.last_call["strategy"] == "parent_doc"

    def test_result_has_correct_fields(self):
        chain, _, _ = self._make_chain()
        result = chain.run("What are cybersecurity risks?", ticker="AAPL")
        assert result.question == "What are cybersecurity risks?"
        assert isinstance(result.answer, str)
        assert isinstance(result.citations, list)
        assert isinstance(result.retrieved_chunks, list)

    def test_ticker_forwarded_to_retriever(self):
        chain, _, retriever = self._make_chain()
        chain.run("What are the risks?", ticker="MSFT")
        assert retriever.last_call["ticker"] == "MSFT"

    def test_llm_invoked_once_per_question(self):
        chain, llm, _ = self._make_chain()
        chain.run("What are the risks?")
        assert len(llm.calls) == 1


class TestQuestionHeuristics:
    def test_broad_terms_detected(self):
        chain = FilingQAChain.__new__(FilingQAChain)
        for term in ["summary", "overview", "key risks", "material changes"]:
            assert chain._question_is_broad(term)

    def test_long_question_is_broad(self):
        chain = FilingQAChain.__new__(FilingQAChain)
        long_q = "What are the top three emerging risk factors that management identified?"
        assert chain._question_is_broad(long_q)

    def test_short_specific_question_not_broad(self):
        chain = FilingQAChain.__new__(FilingQAChain)
        assert not chain._question_is_broad("What is the EPS?")

    def test_comparison_terms_detected(self):
        chain = FilingQAChain.__new__(FilingQAChain)
        assert chain._question_requests_comparison("How did revenue change?")
        assert chain._question_requests_comparison("Compare the prior year results")
        assert not chain._question_requests_comparison("What is the revenue?")
