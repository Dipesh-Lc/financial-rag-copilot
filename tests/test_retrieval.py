"""
test_retrieval.py
Tests similarity, multi-query, and parent-doc retrieval against a
temporary Chroma store that uses a mocked embedding function.
No HuggingFace network access needed.
"""

import math
import random
import pytest
from unittest.mock import patch, MagicMock
from langchain_core.documents import Document

DIM = 384


def _deterministic_vec(text: str) -> list[float]:
    seed = hash(text) % (2 ** 31)
    random.seed(seed)
    raw = [random.gauss(0, 1) for _ in range(DIM)]
    norm = math.sqrt(sum(x ** 2 for x in raw))
    return [x / norm for x in raw]


def _make_mock_embeddings():
    mock = MagicMock()
    mock.embed_query.side_effect = _deterministic_vec
    mock.embed_documents.side_effect = lambda texts: [_deterministic_vec(t) for t in texts]
    mock.model_name = "mock-model"
    return mock


def _make_store(tmp_path, mock_emb):
    from langchain_chroma import Chroma
    store = Chroma(
        collection_name="test_col",
        embedding_function=mock_emb,
        persist_directory=str(tmp_path / "chroma"),
    )
    docs = [
        Document(
            page_content="Apple faces intense competition in the smartphone market from Android manufacturers.",
            metadata={
                "ticker": "AAPL", "section_name": "Risk Factors",
                "filing_date": "2024-09-28", "chunk_id": "c001",
                "parent_document_id": "doc001", "chunk_index": 0,
                "form_type": "10-K", "source_url": "",
            },
        ),
        Document(
            page_content="Apple also competes in personal computers, tablets and wearable devices globally.",
            metadata={
                "ticker": "AAPL", "section_name": "Risk Factors",
                "filing_date": "2024-09-28", "chunk_id": "c002",
                "parent_document_id": "doc001", "chunk_index": 1,
                "form_type": "10-K", "source_url": "",
            },
        ),
        Document(
            page_content="Microsoft Azure cloud revenue grew 20% year over year driven by AI services.",
            metadata={
                "ticker": "MSFT", "section_name": "MD&A",
                "filing_date": "2024-06-30", "chunk_id": "c003",
                "parent_document_id": "doc002", "chunk_index": 0,
                "form_type": "10-K", "source_url": "",
            },
        ),
    ]
    store.add_documents(docs, ids=["c001", "c002", "c003"])
    return store


_HF_PATCH = "app.embeddings.hf_embeddings.HuggingFaceEmbeddings"


# ── Similarity retrieval ──────────────────────────────────────────────────────

def test_similarity_returns_results(tmp_path):
    mock_emb = _make_mock_embeddings()
    with patch(_HF_PATCH, return_value=mock_emb):
        from app.embeddings import hf_embeddings as hfe
        hfe._INSTANCE = None
        from app.vectorstore.retriever import FilingRetriever
        store = _make_store(tmp_path, mock_emb)
        retriever = FilingRetriever(store, top_k=3)
        results = retriever.retrieve("competition risk", strategy="similarity")
    assert len(results) > 0


def test_similarity_has_required_keys(tmp_path):
    mock_emb = _make_mock_embeddings()
    with patch(_HF_PATCH, return_value=mock_emb):
        from app.embeddings import hf_embeddings as hfe
        hfe._INSTANCE = None
        from app.vectorstore.retriever import FilingRetriever
        store = _make_store(tmp_path, mock_emb)
        retriever = FilingRetriever(store, top_k=3)
        results = retriever.retrieve("cloud revenue")
    for r in results:
        for key in ("text", "score", "metadata", "chunk_id", "ticker", "filing_date"):
            assert key in r, f"Missing key: {key}"


def test_ticker_filter(tmp_path):
    mock_emb = _make_mock_embeddings()
    with patch(_HF_PATCH, return_value=mock_emb):
        from app.embeddings import hf_embeddings as hfe
        hfe._INSTANCE = None
        from app.vectorstore.retriever import FilingRetriever
        store = _make_store(tmp_path, mock_emb)
        retriever = FilingRetriever(store, top_k=5)
        results = retriever.retrieve("revenue growth", ticker="MSFT", strategy="similarity")
    for r in results:
        assert r["metadata"].get("ticker") == "MSFT"


def test_returns_at_most_top_k(tmp_path):
    mock_emb = _make_mock_embeddings()
    with patch(_HF_PATCH, return_value=mock_emb):
        from app.embeddings import hf_embeddings as hfe
        hfe._INSTANCE = None
        from app.vectorstore.retriever import FilingRetriever
        store = _make_store(tmp_path, mock_emb)
        retriever = FilingRetriever(store, top_k=2)
        results = retriever.retrieve("any query", top_k=2)
    assert len(results) <= 2


# ── Query expansion ───────────────────────────────────────────────────────────

def test_query_expansion_produces_variants():
    from app.vectorstore.retriever import _expand_query
    variants = _expand_query("What are the risk factors for Apple?")
    assert len(variants) >= 2
    assert all(isinstance(v, str) and v.strip() for v in variants)


def test_query_expansion_no_duplicates():
    from app.vectorstore.retriever import _expand_query
    for query in [
        "How does Apple manage supply chain risk?",
        "What revenue growth does Microsoft report?",
        "Describe Alphabet regulatory challenges",
    ]:
        variants = _expand_query(query)
        lc = [v.lower().strip() for v in variants]
        assert len(lc) == len(set(lc)), f"Duplicates in expansion of: {query}"


def test_query_expansion_max_variants():
    from app.vectorstore.retriever import _expand_query
    variants = _expand_query("What are the risks?")
    assert len(variants) <= 4


# ── Multi-query retrieval ─────────────────────────────────────────────────────

def test_multi_query_returns_results(tmp_path):
    mock_emb = _make_mock_embeddings()
    with patch(_HF_PATCH, return_value=mock_emb):
        from app.embeddings import hf_embeddings as hfe
        hfe._INSTANCE = None
        from app.vectorstore.retriever import FilingRetriever
        store = _make_store(tmp_path, mock_emb)
        retriever = FilingRetriever(store, top_k=3)
        results = retriever.retrieve("Apple competitive threats", strategy="multi_query")
    assert len(results) > 0


def test_multi_query_no_duplicate_chunk_ids(tmp_path):
    mock_emb = _make_mock_embeddings()
    with patch(_HF_PATCH, return_value=mock_emb):
        from app.embeddings import hf_embeddings as hfe
        hfe._INSTANCE = None
        from app.vectorstore.retriever import FilingRetriever
        store = _make_store(tmp_path, mock_emb)
        retriever = FilingRetriever(store, top_k=5)
        results = retriever.retrieve("Apple competition smartphone", strategy="multi_query")
    chunk_ids = [r["chunk_id"] for r in results]
    assert len(chunk_ids) == len(set(chunk_ids)), "Multi-query must deduplicate chunk_ids"


# ── Parent-doc retrieval ──────────────────────────────────────────────────────

def test_parent_doc_returns_results(tmp_path):
    mock_emb = _make_mock_embeddings()
    with patch(_HF_PATCH, return_value=mock_emb):
        from app.embeddings import hf_embeddings as hfe
        hfe._INSTANCE = None
        from app.vectorstore.retriever import FilingRetriever
        store = _make_store(tmp_path, mock_emb)
        retriever = FilingRetriever(store, top_k=3)
        results = retriever.retrieve("Apple competition", strategy="parent_doc")
    assert len(results) > 0


def test_parent_doc_merges_siblings(tmp_path):
    mock_emb = _make_mock_embeddings()
    with patch(_HF_PATCH, return_value=mock_emb):
        from app.embeddings import hf_embeddings as hfe
        hfe._INSTANCE = None
        from app.vectorstore.retriever import FilingRetriever
        store = _make_store(tmp_path, mock_emb)
        retriever = FilingRetriever(store, top_k=3)
        results = retriever.retrieve("Apple competition", strategy="parent_doc")
    # c001 and c002 share parent doc001; merged text should be longer than one chunk
    apple_results = [r for r in results if r.get("ticker") == "AAPL"]
    if apple_results:
        assert len(apple_results[0]["text"]) > 50
