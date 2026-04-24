"""state.py — shared application state for the Gradio session."""
from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class AppState:
    last_chunks: list[dict] = field(default_factory=list)
    last_answer: str = ""
    last_memo: dict | None = None
    selected_ticker: str = ""
    selected_form_type: str = "10-K"
