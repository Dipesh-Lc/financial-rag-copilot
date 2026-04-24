#!/usr/bin/env python
"""
build_corpus.py
Parse and clean all downloaded filings; save to data/processed/.

Usage:
    python scripts/build_corpus.py
    python scripts/build_corpus.py --tickers AAPL --forms 10-K
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import DEFAULT_TICKERS, DEFAULT_FORM_TYPES
from app.ingestion.loaders import FilingLoader
from app.logging_config import get_logger

logger = get_logger("build_corpus")


def main():
    parser = argparse.ArgumentParser(description="Parse and clean downloaded filings")
    parser.add_argument("--tickers", nargs="+", default=DEFAULT_TICKERS)
    parser.add_argument("--forms", nargs="+", default=DEFAULT_FORM_TYPES)
    parser.add_argument("--max-filings", type=int, default=2)
    args = parser.parse_args()

    loader = FilingLoader()
    for ticker in args.tickers:
        docs = loader.load_ticker(ticker, form_types=args.forms, max_filings=args.max_filings)
        logger.info("%s: %d sections saved", ticker, len(docs))


if __name__ == "__main__":
    main()
