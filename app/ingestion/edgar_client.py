"""
edgar_client.py
Thin wrapper around the SEC EDGAR full-text search and submissions API.
Downloads raw filing documents for a given ticker / form type.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import requests

from app.config import EDGAR_USER_AGENT, EDGAR_BASE_URL, EDGAR_RATE_LIMIT_DELAY, FILINGS_DIR
from app.logging_config import get_logger

logger = get_logger(__name__)

HEADERS = {"User-Agent": EDGAR_USER_AGENT, "Accept-Encoding": "gzip, deflate"}


class EdgarClient:
    """Fetches filings from SEC EDGAR."""

    def __init__(self, user_agent: str = EDGAR_USER_AGENT):
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"})

    # ── CIK resolution ────────────────────────────────────────────────────────

    def get_cik(self, ticker: str) -> Optional[str]:
        """Return zero-padded 10-digit CIK for a ticker symbol."""
        url = "https://www.sec.gov/files/company_tickers.json"
        resp = self._get(url)
        if resp is None:
            return None
        data = resp.json()
        ticker_upper = ticker.upper()
        for entry in data.values():
            if entry.get("ticker", "").upper() == ticker_upper:
                cik = str(entry["cik_str"]).zfill(10)
                logger.info("Resolved %s → CIK %s", ticker, cik)
                return cik
        logger.warning("CIK not found for ticker %s", ticker)
        return None

    # ── Submissions / filings list ────────────────────────────────────────────

    def get_submissions(self, cik: str) -> Optional[dict]:
        """Return the submissions JSON for a CIK."""
        url = f"{EDGAR_BASE_URL}/submissions/CIK{cik}.json"
        resp = self._get(url)
        return resp.json() if resp else None

    def get_filing_urls(
        self,
        ticker: str,
        form_type: str = "10-K",
        max_filings: int = 2,
    ) -> list[dict]:
        """
        Return a list of filing dicts with accession number and primary document URL.
        Each dict: {accession, filing_date, form_type, primary_doc_url, cik}
        """
        cik = self.get_cik(ticker)
        if not cik:
            return []

        submissions = self.get_submissions(cik)
        if not submissions:
            return []

        filings_data = submissions.get("filings", {}).get("recent", {})
        forms = filings_data.get("form", [])
        accessions = filings_data.get("accessionNumber", [])
        dates = filings_data.get("filingDate", [])
        primary_docs = filings_data.get("primaryDocument", [])

        results: list[dict] = []
        for form, acc, date, doc in zip(forms, accessions, dates, primary_docs):
            if form != form_type:
                continue
            acc_clean = acc.replace("-", "")
            doc_url = (
                f"https://www.sec.gov/Archives/edgar/data/{int(cik)}"
                f"/{acc_clean}/{doc}"
            )
            results.append(
                {
                    "ticker": ticker.upper(),
                    "cik": cik,
                    "form_type": form,
                    "filing_date": date,
                    "accession": acc,
                    "primary_doc_url": doc_url,
                }
            )
            if len(results) >= max_filings:
                break

        logger.info("Found %d %s filings for %s", len(results), form_type, ticker)
        return results

    # ── Document download ─────────────────────────────────────────────────────

    def download_filing(self, filing_meta: dict, output_dir: Path = FILINGS_DIR) -> Optional[Path]:
        """
        Download the primary document for a filing and save to disk.
        Returns the saved file path or None on failure.
        """
        url = filing_meta["primary_doc_url"]
        ticker = filing_meta["ticker"]
        form_type = filing_meta["form_type"].replace("/", "-")
        date = filing_meta["filing_date"]

        out_subdir = output_dir / ticker / form_type
        out_subdir.mkdir(parents=True, exist_ok=True)

        filename = f"{ticker}_{form_type}_{date}.htm"
        out_path = out_subdir / filename

        if out_path.exists():
            logger.info("Already downloaded: %s", out_path)
            return out_path

        resp = self._get(url)
        if resp is None:
            return None

        out_path.write_bytes(resp.content)
        logger.info("Downloaded %s → %s", url, out_path)
        return out_path

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _get(self, url: str) -> Optional[requests.Response]:
        try:
            time.sleep(EDGAR_RATE_LIMIT_DELAY)
            resp = self.session.get(url, timeout=30)
            resp.raise_for_status()
            return resp
        except requests.RequestException as exc:
            logger.error("GET %s failed: %s", url, exc)
            return None
