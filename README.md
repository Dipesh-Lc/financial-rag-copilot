---
title: Financial RAG Copilot
emoji: 📈
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# Financial RAG Copilot

[![CI](https://github.com/Dipesh-Lc/financial-rag-copilot/actions/workflows/ci.yml/badge.svg)](https://github.com/Dipesh-Lc/financial-rag-copilot/actions/workflows/ci.yml)

**Live demo:** _coming soon — deploy this Space on Hugging Face to add the link here_

Financial RAG Copilot is a RAG project for SEC filings. It downloads real EDGAR filings, parses and chunks them, indexes them in Chroma, answers filing-grounded questions with citations, and generates structured financial risk memos through a simple Gradio UI.

![Q&A Screenshot](assets/screenshots/qa-page.png)

## What this project demonstrates

- retrieval-augmented generation over real SEC 10-K and 10-Q filings
- embeddings and vector search with Hugging Face + Chroma
- grounded Q&A with source citations
- structured memo generation with Pydantic validation
- a lightweight evaluation workflow for retrieval and answer quality
- a usable local web UI built with Gradio

## Core features

### Filing Q&A
Ask questions about indexed filings and get answers grounded in retrieved evidence.

- ticker and form-type filtering
- citation-backed answers
- retrieval strategy selection
- raw evidence inspection

### Risk memo generation
Generate a structured memo for a filing using retrieved evidence only.

The memo includes:
- summary
- key risks
- key changes when prior filing context exists
- supporting evidence
- confidence score
- limitations

### Evidence inspection
Inspect the retrieved chunks behind an answer.

### Evaluation
Run a small labeled evaluation workflow and load the latest results in the UI.

## Screenshots

### Risk Memo
![Risk Memo Screenshot](assets/screenshots/memo-page.png)

### Evidence Inspection
![Evidence Screenshot](assets/screenshots/evidence-page.png)

### Evaluation
![Q&A Screenshot](assets/screenshots/eval-page.png)

## Stack

- Python
- LangChain
- Hugging Face sentence-transformers
- Chroma
- Gradio
- Pydantic
- Pytest

## Repository layout

```text
financial-rag-copilot/
├── app/
│   ├── ingestion/
│   ├── processing/
│   ├── embeddings/
│   ├── vectorstore/
│   ├── rag/
│   ├── evaluation/
│   ├── services/
│   └── ui/
├── assets/screenshots/
├── data/
│   ├── eval/
│   ├── seed/          ← committed AAPL 10-K seed corpus (auto-indexed on first run)
│   └── processed/
├── scripts/
├── tests/
├── .env.example
├── Dockerfile
├── pyproject.toml
├── README.md
└── requirements.txt
```

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/Dipesh-Lc/financial-rag-copilot
cd financial-rag-copilot
```

### 2. Create an environment

Using conda:

```bash
conda create -n fincopilot python=3.11 -y
conda activate fincopilot
```

Or using venv:

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

### 4. Configure environment variables

```bash
cp .env.example .env
```

Set at least:
- `EDGAR_USER_AGENT`
- one LLM provider key

Example:

```env
LLM_PROVIDER=gemini
LLM_MODEL_NAME=gemini-2.5-flash
GOOGLE_API_KEY=your_key_here
EDGAR_USER_AGENT=Your Name your_email@example.com
```

## Quickstart

### 1. Run tests

```bash
pytest
```

### 2. Launch the app

```bash
python -m app.main
```

The app starts on `http://127.0.0.1:7860`. On first launch with an empty index, it automatically builds a demo index from the committed AAPL 10-K seed corpus — no manual setup required.

Then open `http://127.0.0.1:7860`.

### 3. (Optional) Download additional filings

```bash
python scripts/download_filings.py --tickers AAPL --forms 10-K --max-filings 1
python scripts/download_filings.py --tickers AAPL --forms 10-Q --max-filings 1
```

### 4. (Optional) Build the cleaned corpus

```bash
python scripts/build_corpus.py --ticker AAPL
```

### 5. (Optional) Rebuild the vector index

```bash
python scripts/build_index.py --reset
```

### 6. (Optional) Seed evaluation data

```bash
python scripts/seed_eval_set.py
```

### 7. (Optional) Run evaluation

```bash
python scripts/run_eval.py
python scripts/run_eval.py --strategy multi_query
```

## Deploying to Hugging Face Spaces

This repo deploys as a **Docker Space** (the `README.md` front-matter sets `sdk: docker`).

Required Space Secrets (set in the Space settings):
- `GOOGLE_API_KEY` (or `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` for other providers)
- `EDGAR_USER_AGENT` — only needed if you want the app to download new filings; the seed index builds without it.
- `LLM_PROVIDER` / `LLM_MODEL_NAME` — optional overrides

The container binds `0.0.0.0:7860` and builds the demo index from the committed seed corpus on cold start.

## Example prompts

Q&A:
- What are the main risk factors highlighted in Apple's latest 10-K?
- What supply-chain risks does Microsoft mention in recent filings?

Memo:
- Generate a financial risk memo for AAPL using the latest 10-K.

## Evaluation notes

The evaluation set is intentionally small. Current metrics include retrieval hit rate, section hit rate, answer relevance, key-point coverage, citation presence, supported sentence ratio, and schema validity.

## Known limitations

- memo quality still depends on provider behavior and token budget
- the evaluation set is small and focused on a narrow set of filings
- advanced retrieval modes do not always outperform plain similarity
- some answer-quality checks are heuristic

## Summary

Built a RAG application over SEC filings using Python, LangChain, Chroma, Hugging Face embeddings, and Gradio, with citation-backed Q&A, structured risk memos, and an evaluation workflow.

## License

MIT
