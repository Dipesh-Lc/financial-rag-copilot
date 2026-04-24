from __future__ import annotations

import inspect
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol

from app.config import CONFIG
from app.llm.factory import LLMConfigError, build_llm
from app.llm.response_utils import coerce_llm_text
from app.logging_config import get_logger
from app.rag.citation_formatter import CitationFormatter
from app.rag.structured_outputs import (
    FinancialMemo,
    StructuredOutputParser,
    render_schema_instructions,
)
from app.utils.json_utils import load_json

logger = get_logger(__name__)


class SupportsInvoke(Protocol):
    def invoke(self, input: Any, config: Any | None = None) -> Any: ...


@dataclass(frozen=True)
class MemoPromptPayload:
    system_prompt: str
    user_prompt: str
    current_context: str
    prior_context: str


@dataclass
class MemoGenerationResult:
    ok: bool
    memo: FinancialMemo | None
    raw_response_text: str
    retrieved_chunks: list[Any]
    citations: list[dict[str, Any]]
    prompt: MemoPromptPayload
    validation_error: str | None = None
    attempts: int = 1
    selected_filing_date: str | None = None
    prior_filing_date: str | None = None


_SYSTEM_PROMPT = """\
You are an AI financial intelligence analyst.

Use only the provided SEC filing evidence.

Return ONLY a valid JSON object that matches the required schema.
Do not include markdown fences.
Do not include commentary before or after the JSON.
Use double quotes for all keys and string values.
The response must be parseable by json.loads().

Rules:
- Do not invent numbers, dates, comparisons, or business claims.
- If evidence is weak, missing, or there is no prior filing context, say so in limitations.
- Keep all risk statements grounded in the supplied excerpts.
- Cite evidence using citation ids like C1, C2.
- Key changes must compare the current filing to the prior filing only when prior evidence is provided.
- Supporting evidence excerpts should be short, concrete, and traceable to the cited chunk.
- Keep evidence excerpts concise.
"""

_USER_TEMPLATE = """\
Generate a structured financial risk memo for the selected filing.

Selected Filing:
- Company: {company}
- Ticker: {ticker}
- Form Type: {form_type}
- Filing Date: {filing_date}
- Prior Comparable Filing Date: {prior_filing_date}

Output constraints:
- Return ONLY a JSON object.
- Use double quotes everywhere.
- No markdown fences.
- No commentary.
- summary must be <= 90 words.
- Return at most 2 key_risks.
- Return at most 3 key_changes.
- Return at most 3 supporting_evidence items.
- Each risk item may include at most 2 implications.
- Each evidence_quote must be <= 160 characters.
- Each supporting_evidence.excerpt must be <= 160 characters.
- If prior context is weak, keep key_changes short and say so in limitations.

Current Filing Context:
{current_context}

Prior Comparable Filing Context:
{prior_context}

Return a JSON object that matches this schema exactly:
{schema}
"""


class MemoGenerator:
    def __init__(
        self,
        retriever: Any | None = None,
        llm: SupportsInvoke | Callable[..., Any] | None = None,
        citation_formatter: CitationFormatter | None = None,
        max_validation_attempts: int | None = None,
    ) -> None:
        self.retriever = retriever
        self.llm = llm
        self.citation_formatter = citation_formatter or CitationFormatter()
        self.parser = StructuredOutputParser(FinancialMemo)
        self.max_validation_attempts = max(
            1, max_validation_attempts or CONFIG.memo_validation_retries
        )

    def generate(
        self,
        *,
        ticker: str,
        form_type: str | None = None,
        filing_date: str | None = None,
        company: str | None = None,
        section_name: str | None = None,
        top_k: int | None = None,
        extra_filters: dict[str, Any] | None = None,
        focus: str | None = None,
    ) -> MemoGenerationResult:
        selected_date = filing_date or self._resolve_latest_filing_date(
            ticker=ticker, form_type=form_type
        )
        prior_date = self._find_prior_filing_date(
            ticker=ticker,
            form_type=form_type,
            selected_filing_date=selected_date,
        )

        retrieval_query = self._build_retrieval_query(
            ticker=ticker,
            form_type=form_type,
            filing_date=selected_date,
            focus=focus,
        )

        current_top_k = min(top_k or CONFIG.default_top_k, 5)
        prior_top_k = 2 if prior_date else 0

        current_chunks = self._retrieve(
            retrieval_query,
            ticker=ticker,
            form_type=form_type,
            filing_date=selected_date,
            section_name=section_name,
            top_k=current_top_k,
            extra_filters=extra_filters,
        )
        prior_chunks = (
            self._retrieve(
                retrieval_query + " Compare with prior filing disclosures and changes.",
                ticker=ticker,
                form_type=form_type,
                filing_date=prior_date,
                section_name=section_name,
                top_k=prior_top_k,
                extra_filters=extra_filters,
            )
            if prior_date
            else []
        )
        combined = current_chunks + prior_chunks

        resolved_company = (
            company
            or self._infer_field(current_chunks, "company_name")
            or self._infer_field(prior_chunks, "company_name")
            or ticker
        )
        prompt = self._build_prompt(
            company=resolved_company,
            ticker=ticker,
            form_type=form_type
            or self._infer_field(current_chunks, "form_type", default="UNKNOWN"),
            filing_date=selected_date
            or self._infer_field(current_chunks, "filing_date", default="UNKNOWN"),
            prior_filing_date=prior_date,
            current_chunks=current_chunks,
            prior_chunks=prior_chunks,
        )

        llm = self.llm or self._build_default_llm()
        last_error: str | None = None
        raw_text = ""

        for attempt in range(1, self.max_validation_attempts + 1):
            attempt_prompt = (
                prompt if attempt == 1 else self._build_retry_prompt(prompt, raw_text, last_error)
            )
            response = self._generate_response(llm, attempt_prompt)
            raw_text = self._coerce_response_text(response)

            logger.warning("RAW MEMO TEXT:\n%s", raw_text)

            parsed = self.parser.parse(raw_text)
            if parsed.ok and parsed.parsed is not None:
                logger.info("Memo generated for %s (attempt %d)", ticker, attempt)
                return MemoGenerationResult(
                    ok=True,
                    memo=parsed.parsed,
                    raw_response_text=raw_text,
                    retrieved_chunks=combined,
                    citations=self.citation_formatter.serialize_citations(combined),
                    prompt=attempt_prompt,
                    attempts=attempt,
                    selected_filing_date=selected_date,
                    prior_filing_date=prior_date,
                )

            last_error = parsed.error or "Unknown validation error"
            logger.warning("Memo attempt %d validation error: %s", attempt, last_error)

        logger.error(
            "Memo generation failed after %d attempts: %s",
            self.max_validation_attempts,
            last_error,
        )
        return MemoGenerationResult(
            ok=False,
            memo=None,
            raw_response_text=raw_text,
            retrieved_chunks=combined,
            citations=self.citation_formatter.serialize_citations(combined),
            prompt=prompt,
            validation_error=last_error,
            attempts=self.max_validation_attempts,
            selected_filing_date=selected_date,
            prior_filing_date=prior_date,
        )

    # ── Prompt building ──────────────────────────────────────────────────────

    def _build_prompt(
        self,
        *,
        company: str,
        ticker: str,
        form_type: str,
        filing_date: str,
        prior_filing_date: str | None,
        current_chunks: Sequence[Any],
        prior_chunks: Sequence[Any],
    ) -> MemoPromptPayload:
        current_context = self._format_context(current_chunks, label="current")
        prior_context = self._format_context(prior_chunks, label="prior")
        user_prompt = _USER_TEMPLATE.format(
            company=company,
            ticker=ticker,
            form_type=form_type,
            filing_date=filing_date,
            prior_filing_date=prior_filing_date or "NONE AVAILABLE",
            current_context=current_context,
            prior_context=prior_context,
            schema=render_schema_instructions(FinancialMemo),
        )
        return MemoPromptPayload(
            system_prompt=_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            current_context=current_context,
            prior_context=prior_context,
        )

    def _build_retry_prompt(
        self,
        base: MemoPromptPayload,
        raw_text: str,
        error: str | None,
    ) -> MemoPromptPayload:
        retry_user = (
            f"{base.user_prompt}\n\n"
            "The previous response did not validate.\n"
            f"Validation error: {error or 'Unknown error'}\n\n"
            "Return ONLY a corrected JSON object.\n"
            "No markdown fences.\n"
            "No commentary.\n"
            "Use double quotes for all keys and strings.\n\n"
            f"Previous response:\n{raw_text}\n"
        )
        return MemoPromptPayload(
            system_prompt=base.system_prompt,
            user_prompt=retry_user,
            current_context=base.current_context,
            prior_context=base.prior_context,
        )

    # ── LLM invocation ───────────────────────────────────────────────────────

    @staticmethod
    def _build_default_llm() -> Any:
        try:
            return build_llm(temperature=0.1, max_tokens=max(CONFIG.llm_max_tokens, 8192))
        except LLMConfigError as exc:
            raise RuntimeError(f"Could not build LLM for memo generation: {exc}") from exc

    def _generate_response(self, llm: Any, prompt: MemoPromptPayload) -> Any:
        if hasattr(llm, "invoke"):
            return llm.invoke(self._build_langchain_payload(prompt))
        if callable(llm):
            sig = inspect.signature(llm)
            if len(sig.parameters) == 1:
                return llm(prompt)
            return llm(prompt.system_prompt, prompt.user_prompt)
        raise TypeError("Unsupported LLM type provided to MemoGenerator.")

    @staticmethod
    def _build_langchain_payload(prompt: MemoPromptPayload) -> Any:
        try:
            from langchain_core.messages import HumanMessage, SystemMessage
        except ImportError:
            return f"System:\n{prompt.system_prompt}\n\nUser:\n{prompt.user_prompt}"
        return [
            SystemMessage(content=prompt.system_prompt),
            HumanMessage(content=prompt.user_prompt),
        ]

    @staticmethod
    def _coerce_response_text(response: Any) -> str:
        """
        Normalize provider-specific response objects into the text payload that
        should actually be parsed.

        Handles:
        - plain strings
        - AIMessage(content="...")
        - AIMessage(content=[{"type":"text","text":"..."}])
        - dict-like wrappers
        - fallback object reprs
        """
        if response is None:
            return ""

        if isinstance(response, str):
            return response

        # LangChain / provider message objects
        content = getattr(response, "content", None)

        if isinstance(content, str):
            return content

        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                    continue
                if isinstance(item, dict):
                    if isinstance(item.get("text"), str):
                        parts.append(item["text"])
                        continue
                    if item.get("type") == "text" and isinstance(item.get("text"), str):
                        parts.append(item["text"])
                        continue
                maybe_text = getattr(item, "text", None)
                if isinstance(maybe_text, str):
                    parts.append(maybe_text)
            if parts:
                return "\n".join(p for p in parts if p)

        # Dict-like direct objects
        if isinstance(response, dict):
            if isinstance(response.get("text"), str):
                return response["text"]
            if isinstance(response.get("content"), str):
                return response["content"]

        # Some Gemini/LangChain stacks stringify to a dict-like wrapper repr.
        # Try to peel out the 'text' field from that repr at parse time instead.
        return str(response)

    # ── Context formatting ───────────────────────────────────────────────────

    def _format_context(self, chunks: Sequence[Any], *, label: str) -> str:
        if not chunks:
            return f"No {label} filing context was found."

        blocks: list[str] = []
        for idx, chunk in enumerate(chunks, start=1):
            if isinstance(chunk, dict):
                meta = chunk.get("metadata", {}) or {}
                text = chunk.get("text", "")
                chunk_id = chunk.get("chunk_id", "")
            else:
                meta = getattr(chunk, "metadata", {}) or {}
                text = getattr(chunk, "text", "")
                chunk_id = getattr(chunk, "chunk_id", "")

            compact_text = str(text).strip()
            if len(compact_text) > 1800:
                compact_text = compact_text[:1800].rstrip() + " ..."

            header = ", ".join(
                [
                    f"citation=C{idx}",
                    f"context={label}",
                    f"ticker={meta.get('ticker', 'UNKNOWN')}",
                    f"form_type={meta.get('form_type', 'UNKNOWN')}",
                    f"filing_date={meta.get('filing_date', 'UNKNOWN')}",
                    f"section={meta.get('section_name', 'UNKNOWN')}",
                    f"chunk_id={chunk_id}",
                ]
            )
            blocks.append(f"[{header}]\n{compact_text}")

        return "\n\n".join(blocks)

    # ── Retrieval helpers ────────────────────────────────────────────────────

    def _retrieve(
        self,
        query: str,
        *,
        ticker: str,
        form_type: str | None,
        filing_date: str | None,
        section_name: str | None,
        top_k: int | None,
        extra_filters: dict | None,
    ) -> list[Any]:
        if self.retriever is None:
            return []

        # Best effort: pass richer filters if the retriever supports them.
        try:
            return self.retriever.retrieve(
                query,
                ticker=ticker,
                form_type=form_type,
                filing_date=filing_date,
                section_name=section_name,
                top_k=top_k,
                strategy="similarity",
                extra_filters=extra_filters,
            )
        except TypeError:
            return self.retriever.retrieve(
                query,
                ticker=ticker,
                form_type=form_type,
                top_k=top_k,
                strategy="similarity",
            )

    def _infer_field(
        self,
        chunks: Sequence[Any],
        field: str,
        *,
        default: str | None = None,
    ) -> str | None:
        for chunk in chunks:
            meta = (
                chunk.get("metadata", {})
                if isinstance(chunk, dict)
                else (getattr(chunk, "metadata", {}) or {})
            )
            value = meta.get(field)
            if value:
                return str(value)
        return default

    @staticmethod
    def _build_retrieval_query(
        *,
        ticker: str,
        form_type: str | None,
        filing_date: str | None,
        focus: str | None,
    ) -> str:
        parts = [f"Generate a financial risk memo for {ticker}"]
        if form_type:
            parts.append(f"using the {form_type} filing")
        if filing_date:
            parts.append(f"dated {filing_date}")
        parts.append(
            "Focus on risk factors, management discussion, business risks, notes, "
            "operational implications, material changes, and emerging risks."
        )
        if focus:
            parts.append(f"Extra focus: {focus}")
        return " ".join(parts)

    def _resolve_latest_filing_date(self, *, ticker: str, form_type: str | None) -> str | None:
        filings = self._list_available_filings(ticker=ticker, form_type=form_type)
        return filings[-1]["filing_date"] if filings else None

    def _find_prior_filing_date(
        self,
        *,
        ticker: str,
        form_type: str | None,
        selected_filing_date: str | None,
    ) -> str | None:
        if not selected_filing_date:
            return None
        filings = self._list_available_filings(ticker=ticker, form_type=form_type)
        prior = [f["filing_date"] for f in filings if f["filing_date"] < selected_filing_date]
        return prior[-1] if prior else None

    def _list_available_filings(
        self,
        *,
        ticker: str,
        form_type: str | None = None,
    ) -> list[dict[str, str]]:
        by_key: dict[tuple, dict] = {}
        for path in Path(CONFIG.chunks_dir).rglob("*.json"):
            try:
                payload = load_json(path)
            except Exception:
                continue

            records = payload if isinstance(payload, list) else [payload]
            for record in records:
                rt = str(record.get("ticker", "")).upper()
                if rt != ticker.upper():
                    continue
                rf = str(record.get("form_type", ""))
                if form_type and rf != form_type:
                    continue
                rd = str(record.get("filing_date", ""))
                if not rd:
                    continue

                key = (rt, rf, rd)
                by_key[key] = {
                    "ticker": rt,
                    "form_type": rf,
                    "filing_date": rd,
                    "company_name": str(record.get("company_name", rt)),
                }

        return sorted(by_key.values(), key=lambda x: x["filing_date"])


__all__ = ["MemoGenerator", "MemoGenerationResult", "MemoPromptPayload"]


def generate_memo(
    ticker: str,
    chunks: list,
    form_type: str = "10-K",
    filing_date: str = "",
    company: str = "",
) -> FinancialMemo:
    """
    Backwards-compatible convenience wrapper around MemoGenerator.
    Injects chunks directly as memo context instead of querying Chroma.
    """

    class _ChunkInjectionMemoGenerator(MemoGenerator):
        def __init__(self, chunks_override: list, **kw):
            super().__init__(**kw)
            self._chunks_override = chunks_override

        def _retrieve(
            self,
            query,
            *,
            ticker,
            form_type,
            filing_date,
            section_name,
            top_k,
            extra_filters,
        ):
            return self._chunks_override

    generator = _ChunkInjectionMemoGenerator(chunks_override=chunks)
    result = generator.generate(
        ticker=ticker,
        form_type=form_type or None,
        filing_date=filing_date or None,
        company=company or None,
    )

    if result.memo is not None:
        return result.memo

    return FinancialMemo(
        company=company or ticker,
        ticker=ticker,
        form_type=form_type,
        filing_date=filing_date,
        summary="Memo generation failed. See limitations field.",
        confidence_score=0.0,
        limitations=[result.validation_error or "Unknown error"],
    )