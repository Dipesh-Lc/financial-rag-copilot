"""
conftest.py
Shared pytest fixtures and environment setup.
Sets dummy env vars so modules that import config.py don't fail
when no .env file is present.
"""

import os
import pytest

# ── Prevent real API calls in tests ──────────────────────────────────────────
os.environ.setdefault("ANTHROPIC_API_KEY", "test-key-placeholder")
os.environ.setdefault("OPENAI_API_KEY", "test-key-placeholder")
os.environ.setdefault("GOOGLE_API_KEY", "test-key-placeholder")
os.environ.setdefault("EDGAR_USER_AGENT", "Test Suite test@example.com")
os.environ.setdefault("LLM_PROVIDER", "anthropic")


@pytest.fixture(scope="session")
def sample_doc():
    return {
        "document_id": "testdoc001",
        "ticker": "AAPL",
        "company_name": "Apple Inc.",
        "form_type": "10-K",
        "filing_date": "2024-09-28",
        "section_name": "Risk Factors",
        "source_url": "https://www.sec.gov/test",
        "text": (
            "Apple faces intense competition across all markets it operates in. "
            "The smartphone market is particularly competitive, with rivals offering "
            "lower-priced alternatives. Supply chain disruptions may adversely affect "
            "production and delivery timelines. Regulatory scrutiny continues to increase "
            "in both the United States and the European Union. "
        ) * 20,
    }


@pytest.fixture(scope="session")
def sample_chunks(sample_doc):
    from app.processing.chunker import DocumentChunker
    chunker = DocumentChunker(chunk_size=200, chunk_overlap=50)
    return chunker.chunk_document(sample_doc)
