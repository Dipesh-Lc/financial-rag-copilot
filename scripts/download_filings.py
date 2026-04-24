#!/usr/bin/env python
"""
download_filings.py
Download SEC filings for configured tickers.

Usage:
    python scripts/download_filings.py
    python scripts/download_filings.py --tickers AAPL MSFT --forms 10-K
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import DEFAULT_TICKERS, DEFAULT_FORM_TYPES, DEFAULT_FILING_COUNT
from app.ingestion.edgar_client import EdgarClient
from app.logging_config import get_logger

logger = get_logger("download_filings")


def main():
    parser = argparse.ArgumentParser(description="Download SEC filings from EDGAR")
    parser.add_argument("--tickers", nargs="+", default=DEFAULT_TICKERS)
    parser.add_argument("--forms", nargs="+", default=DEFAULT_FORM_TYPES)
    parser.add_argument("--max-filings", type=int, default=DEFAULT_FILING_COUNT)
    args = parser.parse_args()

    client = EdgarClient()
    total = 0
    for ticker in args.tickers:
        for form in args.forms:
            metas = client.get_filing_urls(ticker, form, args.max_filings)
            for meta in metas:
                path = client.download_filing(meta)
                if path:
                    total += 1
                    logger.info("✓ %s", path)

    logger.info("Downloaded %d filing documents.", total)


if __name__ == "__main__":
    main()
