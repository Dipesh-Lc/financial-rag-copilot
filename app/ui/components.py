"""
components.py
Reusable Gradio component builders and output formatters.
"""

from __future__ import annotations

import json


def format_chunks_for_display(chunks: list[dict]) -> str:
    if not chunks:
        return "No chunks retrieved."
    lines = []
    for i, c in enumerate(chunks, 1):
        meta = c.get("metadata", {})
        section = c.get("section_name") or meta.get("section_name", "?")
        date = c.get("filing_date") or meta.get("filing_date", "?")
        ticker = c.get("ticker") or meta.get("ticker", "?")
        score = c.get("score", "?")
        text = c.get("text", "")[:400]
        lines.append(
            f"### Chunk {i} | {ticker} | {section} | {date} | score={score}\n{text}..."
        )
    return "\n\n---\n\n".join(lines)


def format_citations_for_display(citations: list[dict]) -> str:
    if not citations:
        return "No citations."
    lines = []
    for i, c in enumerate(citations, 1):
        # Support both old schema (section/filing_date) and CitationRecord (citation_id/section_name)
        section = c.get("section") or c.get("section_name", "")
        date = c.get("filing_date", "")
        ticker = c.get("ticker", "")
        cid = c.get("citation_id", str(i))
        lines.append(
            f"**[{cid}]** {ticker} | {section} | {date}"
            f"  score={c.get('score', '?')}\n   {c.get('excerpt', '')}"
        )
    return "\n\n".join(lines)


def format_memo_for_display(memo: dict) -> tuple[str, str]:
    """Return (human-readable summary, raw JSON string)."""
    if not memo:
        return "No memo generated.", "{}"

    risks_md = ""
    for r in memo.get("key_risks", []):
        severity = r.get("severity", "?").upper()
        # Support both old schema (risk_title) and new schema (title)
        title = r.get("title") or r.get("risk_title", "")
        risks_md += f"\n- **[{severity}] {title}**: {r.get('description', '')}"

    changes = "\n".join(f"- {c}" for c in memo.get("key_changes", []))

    # limitations is now list[str] — render as bullet list
    limitations_raw = memo.get("limitations", "")
    if isinstance(limitations_raw, list):
        limitations_text = "\n".join(f"- {l}" for l in limitations_raw) or "_(none)_"
    else:
        limitations_text = str(limitations_raw) or "_(none)_"

    summary_md = f"""## {memo.get('company', '')} ({memo.get('ticker', '')}) — {memo.get('form_type', '')} {memo.get('filing_date', '')}

**Executive Summary**
{memo.get('summary', '')}

**Key Risks**{risks_md}

**Key Changes**
{changes or '_(none identified)_'}

**Confidence Score**: {memo.get('confidence_score', 0):.2f}

**Limitations**
{limitations_text}
"""
    return summary_md, json.dumps(memo, indent=2)
