# Financial RAG Copilot

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
![Q&A Screenshot](assets/screenshots/qa-page.png)

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
ai-financial-intelligence-copilot/
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
│   └── processed/
├── scripts/
├── tests/
├── .env.example
├── pyproject.toml
├── README.md
└── requirements.txt
```

## Setup

### 1. Clone the repo

```bash
git clone <your-repo-url>
cd ai-financial-intelligence-copilot
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
LLM_MODEL_NAME=gemini-3.1-pro-preview
GOOGLE_API_KEY=your_key_here
EDGAR_USER_AGENT=Your Name your_email@example.com
```

## Quickstart

### 1. Run tests

```bash
pytest
```

### 2. Download filings

```bash
python scripts/download_filings.py --tickers AAPL --forms 10-K --max-filings 1
python scripts/download_filings.py --tickers AAPL --forms 10-Q --max-filings 1
```

### 3. Build the cleaned corpus

```bash
python scripts/build_corpus.py --ticker AAPL
```

### 4. Build the vector index

```bash
python scripts/build_index.py --reset
```

### 5. Seed evaluation data

```bash
python scripts/seed_eval_set.py
```

### 6. Run evaluation

```bash
python scripts/run_eval.py
python scripts/run_eval.py --strategy multi_query
```

### 7. Launch the app

```bash
python -m app.main
```

Then open `http://127.0.0.1:7860`.

## Example prompts

Q&A:
- What are the main risk factors highlighted in Apple’s latest 10-K?
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
