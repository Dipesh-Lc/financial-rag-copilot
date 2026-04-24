"""
cleaner.py
Remove boilerplate, excessive whitespace, and noise from filing text.
"""

from __future__ import annotations

import re


class TextCleaner:
    # Patterns to strip
    _BOILERPLATE = re.compile(
        r"(table of contents|forward[- ]looking statements?|"
        r"this page intentionally left blank|"
        r"incorporated by reference|see accompanying notes)",
        re.I,
    )
    _MULTI_NEWLINE = re.compile(r"\n{3,}")
    _MULTI_SPACE = re.compile(r"[ \t]{2,}")
    _PAGE_NUM = re.compile(r"^\s*-?\d+-?\s*$", re.M)
    _DOTS = re.compile(r"\.{4,}")   # table-of-contents dot leaders

    def clean(self, text: str) -> str:
        text = self._PAGE_NUM.sub("", text)
        text = self._DOTS.sub(" ", text)
        text = self._BOILERPLATE.sub("", text)
        text = self._MULTI_SPACE.sub(" ", text)
        text = self._MULTI_NEWLINE.sub("\n\n", text)
        return text.strip()

    def clean_sections(self, sections: list[dict]) -> list[dict]:
        cleaned = []
        for sec in sections:
            cleaned_text = self.clean(sec["text"])
            if len(cleaned_text) > 100:   # drop near-empty sections
                cleaned.append({**sec, "text": cleaned_text})
        return cleaned
