"""
citation_formatter.py
Typed citation building from retrieved chunks.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Sequence


@dataclass
class CitationRecord:
    citation_id: str
    chunk_id: str
    ticker: str | None
    form_type: str | None
    filing_date: str | None
    section_name: str | None
    parent_document_id: str | None
    source_url: str | None
    score: float | None
    excerpt: str


class CitationFormatter:
    def __init__(self, excerpt_chars: int = 300) -> None:
        self.excerpt_chars = excerpt_chars

    def build_citations(self, chunks: Sequence[Any]) -> list[CitationRecord]:
        citations: list[CitationRecord] = []
        for index, chunk in enumerate(chunks, start=1):
            if isinstance(chunk, dict):
                metadata = chunk.get("metadata", {})
                citations.append(CitationRecord(
                    citation_id=f"C{index}",
                    chunk_id=chunk.get("chunk_id", ""),
                    ticker=chunk.get("ticker") or metadata.get("ticker"),
                    form_type=chunk.get("form_type") or metadata.get("form_type"),
                    filing_date=chunk.get("filing_date") or metadata.get("filing_date"),
                    section_name=chunk.get("section_name") or metadata.get("section_name"),
                    parent_document_id=chunk.get("parent_document_id"),
                    source_url=chunk.get("source_url") or metadata.get("source_url"),
                    score=chunk.get("score"),
                    excerpt=self._truncate(chunk.get("text", "")),
                ))
            else:
                # RetrievedChunk dataclass
                metadata = getattr(chunk, "metadata", {}) or {}
                citations.append(CitationRecord(
                    citation_id=f"C{index}",
                    chunk_id=getattr(chunk, "chunk_id", ""),
                    ticker=metadata.get("ticker"),
                    form_type=metadata.get("form_type"),
                    filing_date=metadata.get("filing_date"),
                    section_name=metadata.get("section_name"),
                    parent_document_id=getattr(chunk, "parent_document_id", None),
                    source_url=metadata.get("source_url"),
                    score=getattr(chunk, "score", None),
                    excerpt=self._truncate(getattr(chunk, "text", "")),
                ))
        return citations

    def serialize_citations(self, chunks: Sequence[Any]) -> list[dict[str, Any]]:
        return [asdict(c) for c in self.build_citations(chunks)]

    def format_context_block(self, chunks: Sequence[Any]) -> str:
        """Build the {context} string injected into prompts."""
        if not chunks:
            return "No filing context was retrieved."
        parts: list[str] = []
        for citation in self.build_citations(chunks):
            descriptor = " | ".join(
                p for p in [
                    f"citation={citation.citation_id}",
                    f"ticker={citation.ticker or 'UNKNOWN'}",
                    f"form_type={citation.form_type or 'UNKNOWN'}",
                    f"date={citation.filing_date or 'UNKNOWN'}",
                    f"section={citation.section_name or 'UNKNOWN'}",
                ]
                if p
            )
            text = next(
                (c.get("text", "") if isinstance(c, dict) else getattr(c, "text", ""))
                for i, c in enumerate(chunks, 1)
                if i == int(citation.citation_id[1:])
            )
            parts.append(f"[{descriptor}]\n{text.strip()}")
        return "\n\n---\n\n".join(parts)

    def ensure_answer_has_citations(self, answer: str, chunks: Sequence[Any]) -> str:
        cleaned = answer.strip()
        if not cleaned or "[C" in cleaned or not chunks:
            return cleaned
        return f"{cleaned} [C1]"

    def _truncate(self, text: str) -> str:
        content = " ".join(str(text).split())
        if len(content) <= self.excerpt_chars:
            return content
        return content[: self.excerpt_chars - 3].rstrip() + "..."


# Backwards-compatible module-level helpers (used by P1's gradio_app)
_default_formatter = CitationFormatter()


def format_context_block(chunks: list[dict]) -> str:
    return _default_formatter.format_context_block(chunks)


def format_citations(chunks: list[dict]) -> list[dict[str, Any]]:
    return _default_formatter.serialize_citations(chunks)


__all__ = ["CitationFormatter", "CitationRecord", "format_context_block", "format_citations"]
