"""
metadata_builder.py
Construct standardised metadata dicts for every parsed section.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Optional


def build_document_metadata(
    ticker: str,
    form_type: str,
    filing_date: str,
    section_name: str,
    source_url: str,
    cik: Optional[str] = None,
    company_name: Optional[str] = None,
) -> dict:
    """Return a metadata dict for a single parsed section."""
    doc_id = _make_id(ticker, form_type, filing_date, section_name)
    return {
        "document_id": doc_id,
        "company_name": company_name or ticker,
        "ticker": ticker.upper(),
        "cik": cik or "",
        "form_type": form_type,
        "filing_date": filing_date,
        "section_name": section_name,
        "source_url": source_url,
        "ingested_at": datetime.now(timezone.utc).isoformat(),
    }


def _make_id(ticker: str, form_type: str, filing_date: str, section: str) -> str:
    raw = f"{ticker}_{form_type}_{filing_date}_{section}"
    return hashlib.md5(raw.encode()).hexdigest()[:12]
