"""test_chunking.py — DocumentChunker and SectionAwareChunker tests."""

import pytest
from app.processing.chunker import DocumentChunker


# ── Basic chunking ────────────────────────────────────────────────────────────

def test_chunks_produced(sample_doc):
    chunker = DocumentChunker(chunk_size=200, chunk_overlap=50)
    chunks = chunker.chunk_document(sample_doc)
    assert len(chunks) > 1, "Long document should produce more than one chunk"


def test_metadata_preserved(sample_doc):
    chunker = DocumentChunker(chunk_size=200, chunk_overlap=50)
    chunks = chunker.chunk_document(sample_doc)
    for chunk in chunks:
        assert chunk["ticker"] == "AAPL"
        assert chunk["section_name"] == "Risk Factors"
        assert chunk["form_type"] == "10-K"
        assert chunk["parent_document_id"] == "testdoc001"
        assert chunk["filing_date"] == "2024-09-28"


def test_chunk_ids_unique(sample_doc):
    chunker = DocumentChunker(chunk_size=200, chunk_overlap=50)
    chunks = chunker.chunk_document(sample_doc)
    ids = [c["chunk_id"] for c in chunks]
    assert len(ids) == len(set(ids)), "All chunk_ids must be unique"


def test_no_empty_chunks(sample_doc):
    chunker = DocumentChunker(chunk_size=200, chunk_overlap=50)
    chunks = chunker.chunk_document(sample_doc)
    for chunk in chunks:
        assert len(chunk["text"].strip()) > 0, "No chunk should be empty"


def test_chunk_index_sequential(sample_doc):
    chunker = DocumentChunker(chunk_size=200, chunk_overlap=50)
    chunks = chunker.chunk_document(sample_doc)
    indices = [c["chunk_index"] for c in chunks]
    assert indices == list(range(len(chunks))), "chunk_index must be sequential from 0"


def test_chunk_documents_batch(sample_doc):
    chunker = DocumentChunker(chunk_size=300, chunk_overlap=60)
    doc2 = {**sample_doc, "document_id": "doc002", "ticker": "MSFT"}
    all_chunks = chunker.chunk_documents([sample_doc, doc2])
    tickers = {c["ticker"] for c in all_chunks}
    assert "AAPL" in tickers
    assert "MSFT" in tickers


# ── Section-aware chunker ─────────────────────────────────────────────────────

def test_section_aware_preserves_sections(sample_doc):
    from app.processing.section_splitter import SectionAwareChunker
    doc_mda = {**sample_doc, "document_id": "doc_mda", "section_name": "MD&A"}
    chunker = SectionAwareChunker(chunk_size=200, chunk_overlap=50)
    chunks = chunker.chunk_documents([sample_doc, doc_mda])
    sections = {c["section_name"] for c in chunks}
    assert "Risk Factors" in sections
    assert "MD&A" in sections
