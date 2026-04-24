"""
memo_service.py
High-level service for generating risk memos via the UI.

Returns typed MemoServiceResponse with all memo fields, JSONL logging,
and a list_available_filings() helper for the UI dropdown.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TYPE_CHECKING

from app.config import CONFIG
from app.rag.memo_generator import MemoGenerationResult, MemoGenerator
from app.rag.structured_outputs import FinancialMemo
from app.utils.json_utils import load_json
if TYPE_CHECKING:
    from app.vectorstore.retriever import FilingRetriever
from app.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class MemoServiceResponse:
    ok: bool
    memo: dict[str, Any] | None
    executive_summary: str | None
    key_risks: list[dict[str, Any]]
    key_changes: list[str]
    supporting_evidence: list[dict[str, Any]]
    confidence_score: float | None
    limitations: list[str]
    raw_json: str | None
    validation_error: str | None
    retrieved_chunks: list[dict[str, Any]]
    citations: list[dict[str, Any]]
    attempts: int
    selected_filing_date: str | None
    prior_filing_date: str | None

    def format_display(self) -> tuple[str, str]:
        """Return (narrative_text, raw_json_str) for the UI."""
        if not self.ok or not self.memo:
            return (
                f"⚠️ Memo generation failed after {self.attempts} attempt(s).\n\n"
                f"Error: {self.validation_error or 'Unknown error'}",
                self.raw_json or "{}",
            )

        risks_text = ""
        for r in self.key_risks:
            sev = r.get("severity", "medium").upper()
            title = r.get("title") or r.get("risk_title", "")
            desc = r.get("description", "")
            risks_text += f"**[{sev}] {title}**\n{desc}\n\n"

        changes_text = "\n".join(f"• {c}" for c in self.key_changes) if self.key_changes else "N/A"
        narrative = (
            f"### Executive Summary\n{self.executive_summary or 'N/A'}\n\n"
            f"### Key Risks\n{risks_text or 'No risks identified.'}"
            f"### Key Changes\n{changes_text}\n\n"
            f"**Confidence Score:** {self.confidence_score or 0:.0%}  "
            f"| **Attempts:** {self.attempts}"
        )
        return narrative, self.raw_json or "{}"


class MemoService:
    def __init__(
        self,
        retriever: "FilingRetriever" | None = None,
        llm: Any | None = None,
        log_path: str | Path | None = None,
        max_validation_attempts: int | None = None,
    ) -> None:
        self.generator = MemoGenerator(
            retriever=retriever,
            llm=llm,
            max_validation_attempts=max_validation_attempts,
        )
        self._retriever = retriever
        self.log_path = Path(log_path or CONFIG.metadata_dir / "memo_runs.jsonl")

    # ── Public API ────────────────────────────────────────────────────────────

    def generate(
        self,
        ticker: str,
        form_type: str = "10-K",
        filing_date: str = "",
        company: str = "",
        top_k: int | None = None,
        focus: str | None = None,
        section_name: str | None = None,
        extra_filters: dict[str, Any] | None = None,
    ) -> MemoServiceResponse:
        """Primary entry point — used by the Gradio UI."""
        result = self.generator.generate(
            ticker=ticker,
            form_type=form_type or None,
            filing_date=filing_date or None,
            company=company or None,
            section_name=section_name,
            top_k=top_k,
            extra_filters=extra_filters,
            focus=focus,
        )
        response = self._to_service_response(result)
        self._append_log({
            "ok": response.ok,
            "ticker": ticker,
            "form_type": form_type,
            "filing_date": filing_date or response.selected_filing_date,
            "selected_filing_date": response.selected_filing_date,
            "prior_filing_date": response.prior_filing_date,
            "attempts": response.attempts,
            "validation_error": response.validation_error,
        })
        return response

    # Alias for P2 callers
    def generate_memo(self, *, ticker: str, **kwargs: Any) -> MemoServiceResponse:
        return self.generate(ticker=ticker, **kwargs)

    def list_available_filings(
        self,
        ticker: str | None = None,
        form_type: str | None = None,
    ) -> list[dict[str, Any]]:
        """Enumerate all indexed filings from the chunks directory."""
        filings: dict[tuple, dict] = {}
        for path in Path(CONFIG.chunks_dir).rglob("*.json"):
            try:
                payload = load_json(path)
            except Exception:
                continue
            records = payload if isinstance(payload, list) else [payload]
            for record in records:
                rt = str(record.get("ticker", "")).upper()
                rf = str(record.get("form_type", ""))
                rd = str(record.get("filing_date", ""))
                if not rt or not rd:
                    continue
                if ticker and rt != ticker.upper():
                    continue
                if form_type and rf != form_type:
                    continue
                key = (rt, rf, rd)
                if key not in filings:
                    filings[key] = {
                        "ticker": rt,
                        "company_name": str(record.get("company_name", rt)),
                        "form_type": rf,
                        "filing_date": rd,
                        "filing_period": record.get("filing_period"),
                        "source_url": record.get("source_url"),
                    }
        return sorted(filings.values(), key=lambda x: (x["ticker"], x["form_type"], x["filing_date"]))

    # ── Internals ─────────────────────────────────────────────────────────────

    def _to_service_response(self, result: MemoGenerationResult) -> MemoServiceResponse:
        memo_dict = result.memo.model_dump() if isinstance(result.memo, FinancialMemo) else None
        raw_json = (
            json.dumps(memo_dict, indent=2, ensure_ascii=False)
            if memo_dict is not None
            else result.raw_response_text
        )
        chunks = [
            c if isinstance(c, dict) else c.to_dict()
            for c in result.retrieved_chunks
        ]
        return MemoServiceResponse(
            ok=result.ok,
            memo=memo_dict,
            executive_summary=(memo_dict or {}).get("summary"),
            key_risks=(memo_dict or {}).get("key_risks", []),
            key_changes=(memo_dict or {}).get("key_changes", []),
            supporting_evidence=(memo_dict or {}).get("supporting_evidence", []),
            confidence_score=(memo_dict or {}).get("confidence_score"),
            limitations=(memo_dict or {}).get("limitations", []),
            raw_json=raw_json,
            validation_error=result.validation_error,
            retrieved_chunks=chunks,
            citations=result.citations,
            attempts=result.attempts,
            selected_filing_date=result.selected_filing_date,
            prior_filing_date=result.prior_filing_date,
        )

    def _append_log(self, record: dict[str, Any]) -> None:
        try:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as exc:
            logger.warning("Failed to write memo log: %s", exc)


__all__ = ["MemoService", "MemoServiceResponse"]
