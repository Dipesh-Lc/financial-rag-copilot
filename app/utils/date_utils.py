"""date_utils.py — date parsing helpers."""
from __future__ import annotations
from datetime import datetime


def parse_filing_date(date_str: str) -> datetime | None:
    for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%Y%m%d"):
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    return None
