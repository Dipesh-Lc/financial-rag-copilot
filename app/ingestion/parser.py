from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

from bs4 import BeautifulSoup

from app.logging_config import get_logger

logger = get_logger(__name__)

# Regex patterns that commonly head 10-K/10-Q sections
SECTION_HEADERS: list[tuple[str, re.Pattern]] = [
    ("Business", re.compile(r"item\s*1[.\s]+business", re.I)),
    ("Risk Factors", re.compile(r"item\s*1a[.\s]+risk\s*factors", re.I)),
    ("Management's Discussion and Analysis", re.compile(
        r"item\s*7[.\s]+management.s\s*discussion", re.I)),
    ("Quantitative and Qualitative Disclosures About Market Risk", re.compile(
        r"item\s*7a", re.I)),
    ("Financial Statements", re.compile(r"item\s*8[.\s]+financial\s*statements", re.I)),
    ("Notes to Financial Statements", re.compile(r"notes?\s+to\s+(consolidated\s+)?financial", re.I)),
]


class FilingParser:
    """Converts a raw filing file into a list of {section_name, text} dicts."""

    def parse_file(self, file_path: Path) -> list[dict]:
        """Parse a downloaded .htm/.txt filing into sections."""
        raw = file_path.read_text(encoding="utf-8", errors="ignore")

        if file_path.suffix.lower() in {".htm", ".html"}:
            text = self._html_to_text(raw)
        else:
            text = raw

        sections = self._split_sections(text)
        logger.info("Parsed %d sections from %s", len(sections), file_path.name)
        return sections

    # ── Internal ──────────────────────────────────────────────────────────────

    def _html_to_text(self, html: str) -> str:
        parser = "xml" if html.lstrip().startswith("<?xml") else "lxml"
        soup = BeautifulSoup(html, parser)
        for tag in soup(["script", "style", "ix:nonnumeric", "ix:nonfraction"]):
            tag.decompose()
        return soup.get_text(separator="\n")

    def _split_sections(self, text: str) -> list[dict]:
        """
        Walk through lines and assign each block to the nearest section header.
        Returns a list of {section_name: str, text: str}.
        """
        lines = text.splitlines()
        sections: list[dict] = []
        current_section = "Preamble"
        buffer: list[str] = []

        for line in lines:
            matched = self._match_section(line)
            if matched:
                # Flush buffer to previous section
                content = "\n".join(buffer).strip()
                if content:
                    sections.append({"section_name": current_section, "text": content})
                current_section = matched
                buffer = []
            else:
                buffer.append(line)

        # Flush final section
        content = "\n".join(buffer).strip()
        if content:
            sections.append({"section_name": current_section, "text": content})

        return sections

    def _match_section(self, line: str) -> Optional[str]:
        stripped = line.strip()
        if len(stripped) < 4 or len(stripped) > 200:
            return None
        for name, pattern in SECTION_HEADERS:
            if pattern.search(stripped):
                return name
        return None
