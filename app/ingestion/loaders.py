"""
loaders.py
High-level loader that ties edgar_client → parser → cleaner → metadata together.
Returns a list of cleaned document dicts ready for chunking.
"""

from __future__ import annotations

import json
from pathlib import Path

from app.config import CLEANED_DOCS_DIR
from app.ingestion.edgar_client import EdgarClient
from app.ingestion.parser import FilingParser
from app.ingestion.cleaner import TextCleaner
from app.ingestion.metadata_builder import build_document_metadata
from app.logging_config import get_logger

logger = get_logger(__name__)


class FilingLoader:
    def __init__(self):
        self.client = EdgarClient()
        self.parser = FilingParser()
        self.cleaner = TextCleaner()

    def load_ticker(
        self,
        ticker: str,
        form_types: list[str] | None = None,
        max_filings: int = 2,
    ) -> list[dict]:
        """
        Download, parse, clean, and return all sections for a ticker.
        Also persists cleaned docs to disk.
        """
        form_types = form_types or ["10-K", "10-Q"]
        all_docs: list[dict] = []

        for form_type in form_types:
            filing_metas = self.client.get_filing_urls(ticker, form_type, max_filings)
            for meta in filing_metas:
                raw_path = self.client.download_filing(meta)
                if raw_path is None:
                    continue
                sections = self.parser.parse_file(raw_path)
                sections = self.cleaner.clean_sections(sections)

                for sec in sections:
                    doc_meta = build_document_metadata(
                        ticker=ticker,
                        form_type=form_type,
                        filing_date=meta["filing_date"],
                        section_name=sec["section_name"],
                        source_url=meta["primary_doc_url"],
                        cik=meta.get("cik"),
                    )
                    doc = {**doc_meta, "text": sec["text"]}
                    all_docs.append(doc)
                    self._persist(doc)

        logger.info("Loaded %d sections for %s", len(all_docs), ticker)
        return all_docs

    def _persist(self, doc: dict) -> None:
        out_dir = CLEANED_DOCS_DIR / doc["ticker"] / doc["form_type"]
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{doc['document_id']}.json"
        out_path.write_text(json.dumps(doc, indent=2), encoding="utf-8")
