"""
prompts.py
LangChain prompt templates + raw prompt payload dataclass.
"""

from __future__ import annotations

from dataclasses import dataclass

from langchain_core.prompts import ChatPromptTemplate


# ── QA prompt ─────────────────────────────────────────────────────────────────

QA_SYSTEM = (
    "You are an AI financial analyst specialising in SEC filings. "
    "Answer questions using ONLY the provided filing context. "
    "Cite every claim with [Section, Date] or [CN] citation markers. "
    "If the context does not contain enough information, say so explicitly."
)

QA_HUMAN = (
    "Context from SEC filings:\n{context}\n\n"
    "Question: {question}\n\n"
    "Answer with inline citations:"
)

QA_PROMPT = ChatPromptTemplate.from_messages([
    ("system", QA_SYSTEM),
    ("human", QA_HUMAN),
])


# ── Memo prompt ───────────────────────────────────────────────────────────────

MEMO_SYSTEM = (
    "You are an AI financial intelligence analyst. "
    "Use only the provided SEC filing evidence. "
    "Return valid JSON that matches the required schema exactly. "
    "Rules:\n"
    "- Do not invent numbers, dates, comparisons, or business claims.\n"
    "- If evidence is weak or missing, say so in limitations.\n"
    "- Keep all risk statements grounded in the supplied excerpts.\n"
    "- Cite evidence using citation ids like C1, C2.\n"
    "- Key changes must compare current to prior filing only when prior evidence is provided.\n"
    "- Output JSON only — no prose, no markdown fences."
)

MEMO_HUMAN = (
    "Generate a structured financial risk memo for the selected filing.\n\n"
    "Selected Filing:\n"
    "- Company: {company}\n"
    "- Ticker: {ticker}\n"
    "- Form Type: {form_type}\n"
    "- Filing Date: {filing_date}\n"
    "- Prior Comparable Filing Date: {prior_filing_date}\n\n"
    "Your memo should cover:\n"
    "1. An executive-style summary of the most material risks.\n"
    "2. Top emerging risks supported by retrieved evidence.\n"
    "3. Key changes from the prior comparable filing if prior context is available.\n"
    "4. Financial or operational implications for the business.\n"
    "5. Supporting citations and uncertainty notes.\n\n"
    "Current Filing Context:\n{current_context}\n\n"
    "Prior Comparable Filing Context:\n{prior_context}\n\n"
    "Return a JSON object matching this schema exactly:\n{schema}"
)

MEMO_PROMPT = ChatPromptTemplate.from_messages([
    ("system", MEMO_SYSTEM),
    ("human", MEMO_HUMAN),
])


# ── Raw payload dataclass (for non-LangChain LLM calls) ──────────────────────

@dataclass(frozen=True)
class PromptPayload:
    system_prompt: str
    user_prompt: str


def build_qa_prompt(question: str, chunks: list) -> PromptPayload:
    from app.rag.citation_formatter import format_context_block
    context = format_context_block(chunks)
    return PromptPayload(
        system_prompt=QA_SYSTEM,
        user_prompt=f"Context from SEC filings:\n{context}\n\nQuestion: {question}\n\nAnswer with inline citations:",
    )
