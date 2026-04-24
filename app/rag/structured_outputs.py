"""
structured_outputs.py
Pydantic v2 schemas for all structured outputs in the system.

This version is hardened for real provider behavior:
- strips fenced JSON
- extracts the largest JSON object/array from noisy text
- unwraps dict-like wrapper strings that contain a nested "text" JSON payload
- falls back to ast.literal_eval for Python-style dict strings
"""

from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass
from typing import Any, TypeVar

from pydantic import BaseModel, Field, ValidationError, field_validator


# ── Domain models ────────────────────────────────────────────────────────────


class SupportingEvidence(BaseModel):
    citation_id: str = Field(..., description="Citation label like C1 or C2.")
    chunk_id: str | None = Field(default=None)
    section_name: str | None = Field(default=None)
    filing_date: str | None = Field(default=None)
    excerpt: str = Field(..., min_length=1)
    rationale: str | None = Field(default=None)


class RiskItem(BaseModel):
    title: str = Field(..., min_length=3, alias="risk_title")
    severity: str = Field(default="medium")
    description: str = Field(..., min_length=5)
    implications: list[str] = Field(default_factory=list)
    citations: list[str] = Field(default_factory=list)
    evidence_quote: str = Field(default="")
    section: str = Field(default="")

    model_config = {"populate_by_name": True}

    @field_validator("severity", mode="before")
    @classmethod
    def normalise_severity(cls, value: str) -> str:
        v = str(value).lower().strip()
        return v if v in {"high", "medium", "low"} else "medium"


class FinancialMemo(BaseModel):
    company: str
    ticker: str
    form_type: str
    filing_date: str
    summary: str
    key_risks: list[RiskItem] = Field(default_factory=list)
    key_changes: list[str] = Field(default_factory=list)
    supporting_evidence: list[SupportingEvidence] = Field(default_factory=list)
    confidence_score: float = Field(default=0.0, ge=0.0, le=1.0)
    limitations: list[str] = Field(default_factory=list)

    @field_validator("confidence_score", mode="before")
    @classmethod
    def clamp_confidence(cls, value: Any) -> float:
        try:
            return max(0.0, min(1.0, float(value)))
        except (TypeError, ValueError):
            return 0.0


class QAResponse(BaseModel):
    question: str
    answer: str
    citations: list[dict[str, Any]] = Field(default_factory=list)
    retrieved_chunks: list[dict[str, Any]] = Field(default_factory=list)
    confidence_score: float = Field(default=0.0, ge=0.0, le=1.0)
    limitations: list[str] = Field(default_factory=list)


# ── Structured output parser ────────────────────────────────────────────────

StructuredModelT = TypeVar("StructuredModelT", bound=BaseModel)


@dataclass
class StructuredParseResult:
    ok: bool
    parsed: BaseModel | None
    raw_text: str
    error: str | None = None
    payload: dict[str, Any] | None = None


class StructuredOutputParser:
    """
    Parse raw model output into a validated Pydantic model.

    Handles:
    - clean JSON strings
    - fenced JSON
    - prose wrapped around JSON
    - Python-style dict strings
    - wrapper strings like:
      "{'type': 'text', 'text': '{ ...json... }', 'extras': {...}}"
    """

    def __init__(self, model_cls: type[StructuredModelT]) -> None:
        self.model_cls = model_cls

    def parse(self, text: str | dict[str, Any] | BaseModel) -> StructuredParseResult:
        if isinstance(text, self.model_cls):
            return StructuredParseResult(
                ok=True,
                parsed=text,
                raw_text=text.model_dump_json(),
                payload=text.model_dump(),
            )

        raw_text = text if isinstance(text, str) else json.dumps(text, ensure_ascii=False)

        try:
            payload = self._coerce_to_dict(text)
            parsed = self.model_cls.model_validate(payload)
            return StructuredParseResult(
                ok=True,
                parsed=parsed,
                raw_text=raw_text,
                payload=payload,
            )
        except (json.JSONDecodeError, ValidationError, TypeError, ValueError, SyntaxError) as exc:
            return StructuredParseResult(
                ok=False,
                parsed=None,
                raw_text=raw_text,
                error=str(exc),
            )

    def parse_or_raise(self, text: str | dict[str, Any] | BaseModel) -> StructuredModelT:
        result = self.parse(text)
        if not result.ok or result.parsed is None:
            raise ValueError(result.error or "Failed to parse structured output.")
        return result.parsed  # type: ignore[return-value]

    @staticmethod
    def extract_json_candidate(text: str) -> str:
        """
        Strip fences and isolate the outermost JSON object/array from noisy text.
        """
        stripped = text.strip()

        if stripped.startswith("{") or stripped.startswith("["):
            return stripped

        fenced = re.search(
            r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```",
            text,
            flags=re.DOTALL | re.IGNORECASE,
        )
        if fenced:
            return fenced.group(1).strip()

        start = min((i for i in [text.find("{"), text.find("[")] if i != -1), default=-1)
        if start == -1:
            raise ValueError("No JSON object or array found in model output.")

        open_char = text[start]
        close_char = "}" if open_char == "{" else "]"
        depth = 0
        in_string = False
        escape = False

        for idx in range(start, len(text)):
            ch = text[idx]
            if escape:
                escape = False
                continue
            if ch == "\\":
                escape = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == open_char:
                depth += 1
            elif ch == close_char:
                depth -= 1
                if depth == 0:
                    return text[start : idx + 1]

        raise ValueError("Could not isolate a valid JSON block from model output.")

    @staticmethod
    def _unwrap_wrapper_literal(payload: str) -> str:
        """
        Handle model outputs that were stringified as Python dict-like wrappers, e.g.
        "{'type': 'text', 'text': '{...json...}', 'extras': {...}}"
        """
        stripped = payload.strip()

        if not stripped.startswith("{") or "'text'" not in stripped:
            return payload

        try:
            obj = ast.literal_eval(stripped)
        except Exception:
            return payload

        if isinstance(obj, dict):
            text_value = obj.get("text")
            if isinstance(text_value, str):
                return text_value

            content_value = obj.get("content")
            if isinstance(content_value, str):
                return content_value

        return payload

    def _coerce_to_dict(self, payload: str | dict[str, Any] | BaseModel) -> dict[str, Any]:
        if isinstance(payload, BaseModel):
            return payload.model_dump()

        if isinstance(payload, dict):
            return payload

        if not isinstance(payload, str):
            raise TypeError(f"Unsupported payload type: {type(payload)!r}")

        candidate_source = self._unwrap_wrapper_literal(payload)
        candidate = self.extract_json_candidate(candidate_source)

        try:
            loaded = json.loads(candidate)
        except json.JSONDecodeError:
            loaded = ast.literal_eval(candidate)

        if not isinstance(loaded, dict):
            raise TypeError("Expected a JSON object for structured output parsing.")

        return loaded


def render_schema_instructions(model_cls: type[BaseModel]) -> str:
    return json.dumps(model_cls.model_json_schema(), indent=2, ensure_ascii=False)


__all__ = [
    "FinancialMemo",
    "QAResponse",
    "RiskItem",
    "SupportingEvidence",
    "StructuredOutputParser",
    "StructuredParseResult",
    "render_schema_instructions",
]