"""
test_memo_generation.py
Tests generate_memo() (backwards-compat shim) and MemoGenerator directly.
Mocks the LLM at the generator level — no live API calls.
"""

import json
import pytest
from unittest.mock import MagicMock, patch

from app.rag.memo_generator import generate_memo, MemoGenerator
from app.rag.structured_outputs import FinancialMemo

SAMPLE_CHUNKS = [
    {
        "chunk_id": "c001",
        "text": "Apple faces intense competition in the smartphone market from low-cost rivals.",
        "section_name": "Risk Factors",
        "filing_date": "2024-09-28",
        "ticker": "AAPL",
        "metadata": {"ticker": "AAPL", "section_name": "Risk Factors",
                     "filing_date": "2024-09-28", "form_type": "10-K"},
        "score": 0.1,
        "parent_document_id": "doc001",
        "source_url": "",
        "form_type": "10-K",
    }
]

VALID_MEMO_JSON = json.dumps({
    "company": "Apple Inc.",
    "ticker": "AAPL",
    "form_type": "10-K",
    "filing_date": "2024-09-28",
    "summary": "Apple faces competitive risks in multiple product categories.",
    "key_risks": [
        {
            "risk_title": "Market Competition",
            "description": "Intense rivalry from Android manufacturers.",
            "severity": "high",
            "evidence_quote": "intense competition in the smartphone market",
            "section": "Risk Factors",
        }
    ],
    "key_changes": ["Increased AI competition noted"],
    "supporting_evidence": [
        {"citation_id": "C1", "chunk_id": "c001", "excerpt": "Apple faces..."}
    ],
    "confidence_score": 0.75,
    "limitations": ["Based on limited excerpts."],
})


def _make_mock_llm(response_text: str):
    mock_response = MagicMock()
    mock_response.content = response_text
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = mock_response
    return mock_llm


# Patch MemoGenerator._build_default_llm so no API key is needed
_PATCH = "app.rag.memo_generator.MemoGenerator._build_default_llm"


def test_valid_memo_via_generate_memo():
    with patch(_PATCH, return_value=_make_mock_llm(VALID_MEMO_JSON)):
        memo = generate_memo(
            ticker="AAPL", chunks=SAMPLE_CHUNKS,
            form_type="10-K", filing_date="2024-09-28",
        )
    assert isinstance(memo, FinancialMemo)
    assert memo.ticker == "AAPL"
    assert len(memo.key_risks) == 1
    assert memo.key_risks[0].severity == "high"
    assert memo.confidence_score == 0.75


def test_memo_with_json_fences():
    fenced = f"```json\n{VALID_MEMO_JSON}\n```"
    with patch(_PATCH, return_value=_make_mock_llm(fenced)):
        memo = generate_memo(ticker="AAPL", chunks=SAMPLE_CHUNKS)
    assert memo.ticker == "AAPL"
    assert len(memo.key_risks) == 1


def test_fallback_on_invalid_json():
    with patch(_PATCH, return_value=_make_mock_llm("not json {{{")):
        memo = generate_memo(ticker="AAPL", chunks=SAMPLE_CHUNKS)
    assert isinstance(memo, FinancialMemo)
    assert memo.ticker == "AAPL"
    assert memo.confidence_score == 0.0
    assert len(memo.limitations) > 0   # limitations is now list[str]


def test_fallback_on_wrong_schema():
    wrong = json.dumps({"unexpected_field": "value"})
    with patch(_PATCH, return_value=_make_mock_llm(wrong)):
        memo = generate_memo(ticker="MSFT", chunks=SAMPLE_CHUNKS)
    assert isinstance(memo, FinancialMemo)
    assert memo.ticker == "MSFT"
    assert memo.confidence_score == 0.0


def test_empty_chunks_do_not_raise():
    with patch(_PATCH, return_value=_make_mock_llm(VALID_MEMO_JSON)):
        memo = generate_memo(ticker="AAPL", chunks=[])
    assert memo is not None


def test_company_name_defaults_to_ticker():
    with patch(_PATCH, return_value=_make_mock_llm(VALID_MEMO_JSON)):
        memo = generate_memo(ticker="AAPL", chunks=SAMPLE_CHUNKS, company="")
    assert memo.company in ("Apple Inc.", "AAPL")


def test_memo_generator_class_directly():
    """Test MemoGenerator class directly with injected LLM."""
    mock_llm = _make_mock_llm(VALID_MEMO_JSON)

    class _StubRetriever:
        def retrieve(self, *a, **kw):
            return SAMPLE_CHUNKS

    generator = MemoGenerator(retriever=_StubRetriever(), llm=mock_llm)
    result = generator.generate(ticker="AAPL", form_type="10-K", filing_date="2024-09-28")
    assert result.ok
    assert result.memo is not None
    assert result.memo.ticker == "AAPL"


def test_memo_generation_result_has_citations():
    mock_llm = _make_mock_llm(VALID_MEMO_JSON)

    class _StubRetriever:
        def retrieve(self, *a, **kw):
            return SAMPLE_CHUNKS

    generator = MemoGenerator(retriever=_StubRetriever(), llm=mock_llm)
    result = generator.generate(ticker="AAPL")
    assert isinstance(result.citations, list)
    assert isinstance(result.retrieved_chunks, list)
