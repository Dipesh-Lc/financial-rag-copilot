"""json_utils.py — safe JSON parsing helpers."""
from __future__ import annotations

import json
from pathlib import Path


def safe_parse(text: str) -> dict | list | None:
    """Strip markdown fences and parse JSON; returns None on failure."""
    text = text.strip()
    for fence in ["```json", "```"]:
        if text.startswith(fence):
            text = text[len(fence):]
    if text.endswith("```"):
        text = text[:-3]
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        return None


def load_json(path: str | Path) -> dict | list:
    """Load and parse a JSON file. Raises on failure."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def dump_json(data: dict | list, path: str | Path, indent: int = 2) -> None:
    """Write data as JSON to a file, creating parent directories as needed."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=indent, ensure_ascii=False), encoding="utf-8")
