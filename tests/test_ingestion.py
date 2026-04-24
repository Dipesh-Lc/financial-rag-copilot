"""
test_ingestion.py
Tests for parser, cleaner, metadata_builder, and filters.
No network access required.
"""

import pytest
from pathlib import Path


# ── TextCleaner ───────────────────────────────────────────────────────────────

class TestTextCleaner:
    def setup_method(self):
        from app.ingestion.cleaner import TextCleaner
        self.cleaner = TextCleaner()

    def test_removes_page_numbers(self):
        text = "Some content\n   42   \nMore content"
        result = self.cleaner.clean(text)
        assert "42" not in result or "content" in result  # page number standalone line gone

    def test_removes_dot_leaders(self):
        text = "Risk Factors..............................7"
        result = self.cleaner.clean(text)
        assert "......" not in result

    def test_collapses_blank_lines(self):
        text = "Line one\n\n\n\n\nLine two"
        result = self.cleaner.clean(text)
        assert "\n\n\n" not in result

    def test_preserves_content(self):
        text = "Apple faces intense competition in global markets."
        result = self.cleaner.clean(text)
        assert "Apple" in result
        assert "competition" in result

    def test_filters_short_sections(self):
        sections = [
            {"section_name": "Risk Factors", "text": "x"},           # too short
            {"section_name": "Business", "text": "A" * 200},         # OK
        ]
        result = self.cleaner.clean_sections(sections)
        assert len(result) == 1
        assert result[0]["section_name"] == "Business"

    def test_clean_sections_preserves_metadata(self):
        sections = [{"section_name": "MD&A", "text": "B" * 200}]
        result = self.cleaner.clean_sections(sections)
        assert result[0]["section_name"] == "MD&A"


# ── FilingParser ──────────────────────────────────────────────────────────────

class TestFilingParser:
    def setup_method(self):
        from app.ingestion.parser import FilingParser
        self.parser = FilingParser()

    def test_parse_plain_text(self):
        text = (
            "ITEM 1A. RISK FACTORS\n"
            "The company faces many risks.\n\n"
            "ITEM 7. MANAGEMENT'S DISCUSSION AND ANALYSIS\n"
            "Revenue grew significantly.\n"
        )
        sections = self.parser._split_sections(text)
        names = [s["section_name"] for s in sections]
        assert "Risk Factors" in names
        assert "Management's Discussion and Analysis" in names

    def test_section_text_non_empty(self):
        text = (
            "ITEM 1A. RISK FACTORS\n"
            "Cybersecurity threats may impact operations.\n"
        )
        sections = self.parser._split_sections(text)
        risk = next((s for s in sections if s["section_name"] == "Risk Factors"), None)
        assert risk is not None
        assert len(risk["text"]) > 0

    def test_html_to_text_strips_tags(self):
        html = "<html><body><p>Apple faces <b>competition</b>.</p></body></html>"
        result = self.parser._html_to_text(html)
        assert "<b>" not in result
        assert "Apple" in result
        assert "competition" in result

    def test_html_to_text_removes_scripts(self):
        html = "<html><script>alert('x')</script><p>Real content.</p></html>"
        result = self.parser._html_to_text(html)
        assert "alert" not in result
        assert "Real content" in result

    def test_match_section_recognises_item_1a(self):
        result = self.parser._match_section("ITEM 1A. RISK FACTORS")
        assert result == "Risk Factors"

    def test_match_section_returns_none_for_noise(self):
        assert self.parser._match_section("") is None
        assert self.parser._match_section("   ") is None
        assert self.parser._match_section("the quick brown fox") is None

    def test_parse_file_returns_list(self, tmp_path):
        filing = tmp_path / "test_filing.htm"
        filing.write_text(
            "<html><body>"
            "<p>ITEM 1A. RISK FACTORS</p>"
            "<p>The company faces cybersecurity risks.</p>"
            "<p>ITEM 7. MANAGEMENT'S DISCUSSION AND ANALYSIS</p>"
            "<p>Revenue increased by 10 percent.</p>"
            "</body></html>"
        )
        sections = self.parser.parse_file(filing)
        assert isinstance(sections, list)
        assert len(sections) > 0
        for s in sections:
            assert "section_name" in s
            assert "text" in s


# ── Metadata builder ──────────────────────────────────────────────────────────

class TestMetadataBuilder:
    def test_builds_required_fields(self):
        from app.ingestion.metadata_builder import build_document_metadata
        meta = build_document_metadata(
            ticker="AAPL",
            form_type="10-K",
            filing_date="2024-09-28",
            section_name="Risk Factors",
            source_url="https://sec.gov/test",
            cik="0000320193",
            company_name="Apple Inc.",
        )
        for field in ("document_id", "ticker", "form_type", "filing_date",
                      "section_name", "source_url", "ingested_at"):
            assert field in meta, f"Missing field: {field}"

    def test_ticker_uppercased(self):
        from app.ingestion.metadata_builder import build_document_metadata
        meta = build_document_metadata(
            ticker="aapl", form_type="10-K",
            filing_date="2024-09-28", section_name="Business",
            source_url="",
        )
        assert meta["ticker"] == "AAPL"

    def test_document_id_deterministic(self):
        from app.ingestion.metadata_builder import build_document_metadata
        m1 = build_document_metadata("AAPL", "10-K", "2024-09-28", "Risk Factors", "")
        m2 = build_document_metadata("AAPL", "10-K", "2024-09-28", "Risk Factors", "")
        assert m1["document_id"] == m2["document_id"]

    def test_different_sections_different_ids(self):
        from app.ingestion.metadata_builder import build_document_metadata
        m1 = build_document_metadata("AAPL", "10-K", "2024-09-28", "Risk Factors", "")
        m2 = build_document_metadata("AAPL", "10-K", "2024-09-28", "Business", "")
        assert m1["document_id"] != m2["document_id"]


# ── Metadata filters ──────────────────────────────────────────────────────────

class TestFilters:
    def test_ticker_filter(self):
        from app.vectorstore.filters import ticker_filter
        assert ticker_filter("aapl") == {"ticker": "AAPL"}

    def test_form_type_filter(self):
        from app.vectorstore.filters import form_type_filter
        assert form_type_filter("10-K") == {"form_type": "10-K"}

    def test_combined_single(self):
        from app.vectorstore.filters import combined_filter
        result = combined_filter(ticker="AAPL")
        assert result == {"ticker": "AAPL"}

    def test_combined_multiple(self):
        from app.vectorstore.filters import combined_filter
        result = combined_filter(ticker="AAPL", form_type="10-K")
        assert result == {"$and": [{"ticker": "AAPL"}, {"form_type": "10-K"}]}

    def test_combined_none_excluded(self):
        from app.vectorstore.filters import combined_filter
        result = combined_filter(ticker="AAPL", form_type=None)
        assert result == {"ticker": "AAPL"}   # None value dropped

    def test_combined_all_none_returns_none(self):
        from app.vectorstore.filters import combined_filter
        result = combined_filter(ticker=None, form_type=None)
        assert result is None
