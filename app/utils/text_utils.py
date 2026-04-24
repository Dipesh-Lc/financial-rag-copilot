"""text_utils.py — misc text helpers."""
from __future__ import annotations
import re


def truncate(text: str, max_chars: int = 300) -> str:
    return text[:max_chars] + "..." if len(text) > max_chars else text


def count_tokens_approx(text: str) -> int:
    """Rough token count: ~4 chars per token."""
    return len(text) // 4


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()
