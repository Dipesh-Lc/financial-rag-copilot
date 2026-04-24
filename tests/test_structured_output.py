"""test_structured_output.py — Pydantic schema validation tests."""

import pytest
from pydantic import ValidationError
from app.rag.structured_outputs import (
    FinancialMemo, RiskItem, QAResponse, SupportingEvidence, StructuredOutputParser
)


# ── FinancialMemo ─────────────────────────────────────────────────────────────

def test_financial_memo_minimal():
    memo = FinancialMemo(
        company="Apple Inc.", ticker="AAPL", form_type="10-K",
        filing_date="2024-09-28", summary="Apple disclosed several risks.",
    )
    assert memo.ticker == "AAPL"
    assert memo.key_risks == []
    assert memo.confidence_score == 0.0


def test_confidence_clamped_above():
    memo = FinancialMemo(
        company="T", ticker="T", form_type="10-K",
        filing_date="2024-01-01", summary="s", confidence_score=2.5,
    )
    assert memo.confidence_score == 1.0


def test_confidence_clamped_below():
    memo = FinancialMemo(
        company="T", ticker="T", form_type="10-K",
        filing_date="2024-01-01", summary="s", confidence_score=-0.5,
    )
    assert memo.confidence_score == 0.0


def test_memo_with_risks():
    memo = FinancialMemo(
        company="Apple Inc.", ticker="AAPL", form_type="10-K",
        filing_date="2024-09-28", summary="Summary",
        key_risks=[
            RiskItem(risk_title="Competition", description="Intense rivalry.", severity="high"),
            RiskItem(risk_title="Regulation", description="EU DMA compliance.", severity="medium"),
        ],
        confidence_score=0.8,
    )
    assert len(memo.key_risks) == 2
    assert memo.key_risks[0].severity == "high"


def test_memo_serializes_to_dict():
    memo = FinancialMemo(
        company="T", ticker="T", form_type="10-K",
        filing_date="2024-01-01", summary="s",
    )
    d = memo.model_dump()
    assert "key_risks" in d
    assert "supporting_evidence" in d
    assert "limitations" in d
    assert isinstance(d["limitations"], list)   # now list[str], not str


def test_limitations_is_list():
    memo = FinancialMemo(
        company="T", ticker="T", form_type="10-K",
        filing_date="2024-01-01", summary="s",
        limitations=["Limited context.", "Prior filing unavailable."],
    )
    assert isinstance(memo.limitations, list)
    assert len(memo.limitations) == 2


# ── RiskItem ──────────────────────────────────────────────────────────────────

def test_risk_severity_normalised_uppercase():
    r = RiskItem(risk_title="Cyber risk", description="Data breach exposure.", severity="HIGH")
    assert r.severity == "high"


def test_risk_severity_unknown_defaults_medium():
    r = RiskItem(risk_title="Cyber risk", description="Data breach exposure.", severity="critical")
    assert r.severity == "medium"


def test_risk_defaults():
    r = RiskItem(risk_title="Supply Chain", description="Disruption risk.")
    assert r.severity == "medium"
    assert r.evidence_quote == ""


def test_risk_item_alias_works():
    """risk_title alias should populate the title field."""
    r = RiskItem(risk_title="Competition", description="Market rivalry.")
    assert r.title == "Competition"


def test_risk_item_title_field_direct():
    """title field name should also work directly."""
    r = RiskItem(title="Competition", description="Market rivalry.")
    assert r.title == "Competition"


def test_risk_model_dump_uses_field_name():
    """model_dump() returns 'title', not 'risk_title'."""
    r = RiskItem(risk_title="Test", description="A description.")
    d = r.model_dump()
    assert "title" in d
    # by_alias=True should return risk_title
    d_alias = r.model_dump(by_alias=True)
    assert "risk_title" in d_alias


# ── SupportingEvidence ────────────────────────────────────────────────────────

def test_supporting_evidence_minimal():
    ev = SupportingEvidence(citation_id="C1", excerpt="Apple faces competition.")
    assert ev.citation_id == "C1"
    assert ev.chunk_id is None


# ── QAResponse ────────────────────────────────────────────────────────────────

def test_qa_response_defaults():
    qa = QAResponse(question="What?", answer="Something.")
    assert qa.citations == []
    assert qa.retrieved_chunks == []
    assert qa.confidence_score == 0.0   # renamed from 'confidence'


def test_qa_response_full():
    qa = QAResponse(
        question="q", answer="a",
        citations=[{"section": "Risk Factors"}],
        retrieved_chunks=[{"text": "x"}],
        confidence_score=0.9,
    )
    assert len(qa.citations) == 1
    assert qa.confidence_score == 0.9


# ── StructuredOutputParser ────────────────────────────────────────────────────

def test_parser_valid_json():
    import json
    parser = StructuredOutputParser(FinancialMemo)
    payload = json.dumps({
        "company": "Apple", "ticker": "AAPL", "form_type": "10-K",
        "filing_date": "2024-09-28", "summary": "Risks identified.",
        "confidence_score": 0.8,
    })
    result = parser.parse(payload)
    assert result.ok
    assert result.parsed.ticker == "AAPL"


def test_parser_strips_json_fences():
    import json
    parser = StructuredOutputParser(FinancialMemo)
    raw = json.dumps({
        "company": "MSFT", "ticker": "MSFT", "form_type": "10-K",
        "filing_date": "2024-06-30", "summary": "s",
    })
    result = parser.parse(f"```json\n{raw}\n```")
    assert result.ok
    assert result.parsed.ticker == "MSFT"


def test_parser_invalid_json_returns_error():
    parser = StructuredOutputParser(FinancialMemo)
    result = parser.parse("not json at all {{{")
    assert not result.ok
    assert result.error is not None


def test_parser_wrong_schema_returns_error():
    import json
    parser = StructuredOutputParser(FinancialMemo)
    result = parser.parse(json.dumps({"unexpected_field": "value"}))
    assert not result.ok
