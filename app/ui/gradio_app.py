from __future__ import annotations

import json

import gradio as gr

from app.config import DEFAULT_FORM_TYPES, CONFIG
from app.ui.components import (
    format_chunks_for_display,
    format_citations_for_display,
    format_memo_for_display,
)
from app.logging_config import get_logger

logger = get_logger(__name__)

_svc: dict = {}


def _get_services() -> dict:
    if not _svc:
        from app.vectorstore.chroma_store import get_vector_store
        from app.vectorstore.retriever import FilingRetriever
        from app.services.query_service import QueryService
        from app.services.memo_service import MemoService
        from app.services.ingestion_service import IngestionService

        store = get_vector_store()
        retriever = FilingRetriever(store)
        _svc["store"] = store
        _svc["query"] = QueryService(retriever)
        _svc["memo"] = MemoService(retriever)
        _svc["retriever"] = retriever
        _svc["ingestion"] = IngestionService(store=store)
    return _svc


def _llm_badge() -> str:
    from app.llm.factory import detect_available_providers
    available = detect_available_providers()
    provider = CONFIG.llm_provider.upper()
    model = CONFIG.llm_model_name
    avail_str = " · ".join(p.upper() for p in available) if available else "none configured"
    return (
        f"**LLM:** `{provider} / {model}`  "
        f"| **Available providers:** {avail_str}  "
        f"| Smart routing: {'✅' if CONFIG.enable_retrieval_upgrades else '❌'}"
    )


def ask_question(question, ticker, form_type, strategy, top_k):
    if not question.strip():
        return "Please enter a question.", "", ""
    svc = _get_services()
    try:
        result = svc["query"].answer(
            question,
            ticker=ticker.strip().upper() if ticker.strip() else None,
            form_type=form_type or None,
            top_k=int(top_k),
            strategy=strategy,
        )
        return (
            result.answer,
            format_citations_for_display(result.citations),
            format_chunks_for_display(result.retrieved_chunks),
        )
    except Exception as exc:
        logger.exception("ask_question error")
        return f"⚠️ Error: {exc}", "", ""


def generate_risk_memo(ticker, form_type, filing_date, top_k, focus):
    if not ticker.strip():
        return "Please enter a ticker symbol.", "{}"
    svc = _get_services()
    try:
        response = svc["memo"].generate(
            ticker=ticker.strip().upper(),
            form_type=form_type or "10-K",
            filing_date=filing_date.strip() or "",
            top_k=int(top_k),
            focus=focus.strip() or None,
        )
        return response.format_display()
    except Exception as exc:
        logger.exception("generate_risk_memo error")
        return f"⚠️ Error: {exc}", "{}"


def retrieve_evidence(query, ticker, form_type, strategy, top_k):
    if not query.strip():
        return "Please enter a query."
    svc = _get_services()
    try:
        chunks = svc["retriever"].retrieve(
            query,
            ticker=ticker.strip().upper() if ticker.strip() else None,
            form_type=form_type or None,
            top_k=int(top_k),
            strategy=strategy,
        )
        header = f"**{len(chunks)} chunk(s) retrieved** · strategy=`{strategy}`\n\n---\n\n"
        return header + format_chunks_for_display(chunks)
    except Exception as exc:
        logger.exception("retrieve_evidence error")
        return f"⚠️ Error: {exc}"


def ingest_ticker(ticker, form_types_str, max_filings):
    if not ticker.strip():
        return "Please enter a ticker symbol."
    svc = _get_services()
    forms = [f.strip() for f in form_types_str.split(",") if f.strip()]
    try:
        result = svc["ingestion"].ingest_ticker(
            ticker.strip().upper(),
            form_types=forms or None,
            max_filings=int(max_filings),
        )
        return (
            f"✅ **{result['ticker']}** ingested successfully\n"
            f"Sections found: {result['docs']}  \n"
            f"New sections: {result['new_docs']}  \n"
            f"Chunks indexed: {result['chunks']}"
        )
    except Exception as exc:
        logger.exception("ingest_ticker error")
        return f"⚠️ Error: {exc}"


def load_eval_results():
    metadata_candidates = sorted(
        CONFIG.metadata_dir.glob("eval_results_*.json"),
        key=lambda path: path.stat().st_mtime,
    )
    eval_candidates = sorted(
        CONFIG.eval_dir.glob("eval_results_*.json"),
        key=lambda path: path.stat().st_mtime,
    )

    if metadata_candidates:
        latest = metadata_candidates[-1]
    elif eval_candidates:
        latest = eval_candidates[-1]
    else:
        return {"status": "No eval results found.", "hint": "Run: python scripts/run_eval.py"}

    data = json.loads(latest.read_text(encoding="utf-8"))
    return {
        "file": latest.name,
        "strategy": data.get("strategy"),
        "num_questions": data.get("num_questions"),
        "summary": data.get("summary", {}),
    }


def list_available_filings_ui(ticker, form_type):
    svc = _get_services()
    filings = svc["memo"].list_available_filings(
        ticker=ticker.strip().upper() if ticker.strip() else None,
        form_type=form_type or None,
    )
    if not filings:
        return "No indexed filings found. Ingest a ticker first."
    lines = [
        f"**{f['ticker']}** · {f['form_type']} · {f['filing_date']} · {f.get('company_name', '')}"
        for f in filings
    ]
    return "\n".join(lines)


STRATEGIES = ["similarity", "multi_query", "parent_doc"]
STRATEGY_HELP = (
    "**similarity** — cosine nearest-neighbours (fastest)  \n"
    "**multi_query** — query expansion + union (broader recall)  \n"
    "**parent_doc** — child chunks → full parent section (more context)  \n"
    "\n*Smart routing auto-upgrades similarity to multi_query/parent_doc for broad questions.*"
)


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="AI Financial Intelligence Copilot", theme=gr.themes.Soft()) as demo:

        gr.Markdown(
            "# 🏦 AI Financial Intelligence Copilot\n"
            "RAG over SEC filings — grounded Q&A · structured risk memos · "
            "source citations · evaluation suite"
        )
        gr.Markdown(_llm_badge())

        with gr.Tabs():

            with gr.TabItem("📋 Ask Questions"):
                gr.Markdown(
                    "Ask anything about a company's SEC filings. "
                    "Answers are grounded in retrieved evidence with section citations."
                )
                with gr.Row():
                    qa_ticker   = gr.Textbox(label="Ticker (optional)", placeholder="AAPL", scale=1)
                    qa_form     = gr.Dropdown(choices=[""] + DEFAULT_FORM_TYPES, value="", label="Form Type", scale=1)
                    qa_strategy = gr.Dropdown(choices=STRATEGIES, value="similarity", label="Retrieval Strategy", scale=1)
                    qa_top_k    = gr.Slider(2, 10, value=5, step=1, label="Top-K", scale=1)
                qa_question = gr.Textbox(label="Question", lines=3,
                    placeholder="What cybersecurity risks does Apple disclose in its 10-K?")
                gr.Markdown(STRATEGY_HELP)
                qa_btn = gr.Button("Ask", variant="primary")
                qa_answer = gr.Markdown(label="Answer")
                with gr.Accordion("Citations", open=False):
                    qa_citations = gr.Markdown()
                with gr.Accordion("Retrieved Chunks (raw)", open=False):
                    qa_chunks = gr.Markdown()
                qa_btn.click(fn=ask_question,
                    inputs=[qa_question, qa_ticker, qa_form, qa_strategy, qa_top_k],
                    outputs=[qa_answer, qa_citations, qa_chunks])

            with gr.TabItem("📄 Risk Memo"):
                gr.Markdown(
                    "Generate a Pydantic-validated `FinancialMemo` JSON from filing excerpts. "
                    "Includes prior-filing comparison when a previous filing is indexed."
                )
                with gr.Row():
                    m_ticker = gr.Textbox(label="Ticker *", placeholder="MSFT", scale=2)
                    m_form   = gr.Dropdown(choices=DEFAULT_FORM_TYPES, value="10-K", label="Form Type", scale=1)
                    m_date   = gr.Textbox(label="Filing Date (optional)", placeholder="2024-09-28", scale=2)
                    m_top_k  = gr.Slider(5, 15, value=10, step=1, label="Evidence chunks", scale=1)
                m_focus = gr.Textbox(label="Focus (optional)",
                    placeholder="cybersecurity risks, AI investments, regulatory exposure")
                with gr.Row():
                    memo_btn         = gr.Button("Generate Risk Memo", variant="primary")
                    list_filings_btn = gr.Button("List Indexed Filings")
                memo_summary = gr.Markdown(label="Memo Summary")
                memo_json    = gr.Code(language="json", label="Validated JSON Output")
                filings_list = gr.Markdown(label="Available Filings")
                memo_btn.click(fn=generate_risk_memo,
                    inputs=[m_ticker, m_form, m_date, m_top_k, m_focus],
                    outputs=[memo_summary, memo_json])
                list_filings_btn.click(fn=list_available_filings_ui,
                    inputs=[m_ticker, m_form], outputs=filings_list)

            with gr.TabItem("🔍 Inspect Evidence"):
                gr.Markdown("Run a retrieval query and inspect chunks: similarity scores, section metadata, source URLs.")
                with gr.Row():
                    ev_ticker   = gr.Textbox(label="Ticker (optional)", placeholder="GOOGL", scale=1)
                    ev_form     = gr.Dropdown(choices=[""] + DEFAULT_FORM_TYPES, value="", label="Form Type", scale=1)
                    ev_strategy = gr.Dropdown(choices=STRATEGIES, value="similarity", label="Strategy", scale=1)
                    ev_top_k    = gr.Slider(2, 15, value=5, step=1, label="Top-K", scale=1)
                ev_query  = gr.Textbox(label="Query", lines=2,
                    placeholder="capital allocation and share repurchase program")
                ev_btn    = gr.Button("Retrieve")
                ev_output = gr.Markdown()
                ev_btn.click(fn=retrieve_evidence,
                    inputs=[ev_query, ev_ticker, ev_form, ev_strategy, ev_top_k],
                    outputs=ev_output)

            with gr.TabItem("⬇️ Ingest Filings"):
                gr.Markdown(
                    "Download, parse, chunk, embed, and index a new ticker. "
                    "Uses the same shared Chroma store as Q&A and memos.  \n"
                    "For bulk ingestion, the CLI is faster: `python scripts/download_filings.py`"
                )
                with gr.Row():
                    ing_ticker = gr.Textbox(label="Ticker *", placeholder="NVDA", scale=2)
                    ing_forms  = gr.Textbox(label="Form Types (comma-separated)", value="10-K,10-Q", scale=2)
                    ing_max    = gr.Slider(1, 5, value=2, step=1, label="Max filings per type", scale=1)
                ing_btn    = gr.Button("Ingest", variant="primary")
                ing_status = gr.Markdown()
                ing_btn.click(fn=ingest_ticker,
                    inputs=[ing_ticker, ing_forms, ing_max], outputs=ing_status)

            with gr.TabItem("📊 Evaluation"):
                gr.Markdown(
                    "View the latest evaluation run summary.  \n"
                    "Generate a new run: `python scripts/run_eval.py --strategy multi_query`"
                )
                eval_btn = gr.Button("Load Latest Results")
                eval_out = gr.JSON(label="Evaluation Summary")
                eval_btn.click(fn=load_eval_results, inputs=[], outputs=eval_out)

    return demo
