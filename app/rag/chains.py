from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Callable, Protocol

from app.config import CONFIG
from app.llm.factory import build_llm, LLMConfigError
from app.llm.response_utils import coerce_llm_text
from app.rag.citation_formatter import CitationFormatter
from app.rag.prompts import PromptPayload, QA_PROMPT, MEMO_PROMPT, build_qa_prompt
from app.logging_config import get_logger

logger = get_logger(__name__)


class SupportsInvoke(Protocol):
    def invoke(self, input: Any, config: Any | None = None) -> Any: ...


@dataclass
class QAChainResult:
    question: str
    answer: str
    citations: list[dict[str, Any]]
    retrieved_chunks: list[Any]
    prompt: PromptPayload
    raw_response: Any


# ── FilingQAChain ─────────────────────────────────────────────────────────────

class FilingQAChain:
    """
    Orchestrates retrieval + generation for Q&A.

    Smart routing: automatically selects multi-query or parent-document
    retrieval for broad/comparison questions unless explicitly overridden.
    LLM is resolved from the factory (Anthropic / OpenAI / Gemini).
    """

    def __init__(
        self,
        retriever: Any | None = None,
        llm: SupportsInvoke | Callable[..., Any] | None = None,
        citation_formatter: CitationFormatter | None = None,
        enable_retrieval_upgrades: bool | None = None,
    ) -> None:
        self.retriever = retriever
        self.llm = llm
        self.citation_formatter = citation_formatter or CitationFormatter()
        self.enable_retrieval_upgrades = (
            CONFIG.enable_retrieval_upgrades
            if enable_retrieval_upgrades is None
            else enable_retrieval_upgrades
        )

    def run(
        self,
        question: str,
        *,
        ticker: str | None = None,
        form_type: str | None = None,
        filing_date: str | None = None,
        section_name: str | None = None,
        top_k: int | None = None,
        strategy: str = "similarity",
        extra_filters: dict[str, Any] | None = None,
        use_multi_query: bool | None = None,
        use_parent_context: bool | None = None,
    ) -> QAChainResult:
        if self.retriever is None:
            raise RuntimeError("FilingQAChain requires a retriever to be provided.")

        # Smart strategy routing
        if self.enable_retrieval_upgrades and strategy == "similarity":
            if self._resolve_use_multi_query(question, use_multi_query):
                strategy = "multi_query"
            elif self._resolve_use_parent_context(question, use_parent_context):
                strategy = "parent_doc"

        chunks = self.retriever.retrieve(
            question,
            ticker=ticker,
            form_type=form_type,
            top_k=top_k,
            strategy=strategy,
        )

        prompt = build_qa_prompt(question, chunks)
        response = self._generate_answer(prompt)
        answer = self.citation_formatter.ensure_answer_has_citations(
            self._coerce_response_text(response), chunks
        )
        citations = self.citation_formatter.serialize_citations(chunks)
        return QAChainResult(
            question=question,
            answer=answer,
            citations=citations,
            retrieved_chunks=chunks,
            prompt=prompt,
            raw_response=response,
        )

    # ── Routing heuristics ────────────────────────────────────────────────────

    def _resolve_use_multi_query(self, question: str, requested: bool | None) -> bool:
        if requested is not None:
            return requested
        return CONFIG.enable_multi_query and self._question_is_broad(question)

    def _resolve_use_parent_context(self, question: str, requested: bool | None) -> bool:
        if requested is not None:
            return requested
        return CONFIG.enable_parent_document_retrieval and (
            self._question_is_broad(question) or self._question_requests_comparison(question)
        )

    @staticmethod
    def _question_is_broad(question: str) -> bool:
        lowered = question.lower()
        broad_terms = ["summary", "overview", "broad", "overall", "main", "top", "key", "material"]
        return any(t in lowered for t in broad_terms) or len(question.split()) >= 12

    @staticmethod
    def _question_requests_comparison(question: str) -> bool:
        lowered = question.lower()
        return any(t in lowered for t in ["compare", "change", "changed", "difference", "versus", "prior"])

    # ── LLM invocation ────────────────────────────────────────────────────────

    def _generate_answer(self, prompt: PromptPayload) -> Any:
        llm = self.llm or self._build_default_llm()
        if hasattr(llm, "invoke"):
            return llm.invoke(self._build_langchain_payload(prompt))
        if callable(llm):
            return self._call_callable_llm(llm, prompt)
        raise RuntimeError("No supported LLM interface was provided to FilingQAChain.")

    def _call_callable_llm(self, llm: Callable[..., Any], prompt: PromptPayload) -> Any:
        sig = inspect.signature(llm)
        if len(sig.parameters) == 1:
            return llm(prompt)
        return llm(prompt.system_prompt, prompt.user_prompt)

    @staticmethod
    def _build_default_llm() -> SupportsInvoke:
        try:
            return build_llm()
        except LLMConfigError as exc:
            raise RuntimeError(f"Could not build default LLM: {exc}") from exc

    @staticmethod
    def _build_langchain_payload(prompt: PromptPayload) -> Any:
        try:
            from langchain_core.messages import HumanMessage, SystemMessage
        except ImportError:
            return f"System:\n{prompt.system_prompt}\n\nUser:\n{prompt.user_prompt}"
        return [SystemMessage(content=prompt.system_prompt), HumanMessage(content=prompt.user_prompt)]

    @staticmethod
    def _coerce_response_text(response: Any) -> str:
        return coerce_llm_text(response)


# ── Backwards-compatible factory functions ────────────────────────────────────

def build_qa_chain(retriever: Any | None = None) -> FilingQAChain:
    """Return a FilingQAChain ready to use via .run()."""
    return FilingQAChain(retriever=retriever)


def build_memo_chain():
    """
    Return a LangChain runnable for memo generation.
    Input: {company, ticker, form_type, filing_date, prior_filing_date, current_context, prior_context, schema}
    Output: AIMessage (JSON)
    """
    from langchain_core.runnables import RunnableLambda
    llm = build_llm()

    def assemble(inputs: dict) -> dict:
        return {k: inputs.get(k, "") for k in [
            "company", "ticker", "form_type", "filing_date",
            "prior_filing_date", "current_context", "prior_context", "schema",
        ]}

    return RunnableLambda(assemble) | MEMO_PROMPT | llm


__all__ = [
    "FilingQAChain",
    "QAChainResult",
    "SupportsInvoke",
    "build_qa_chain",
    "build_memo_chain",
]
