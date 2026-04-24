"""
test_embeddings.py
Tests the embedding wrapper logic and interface contract.
Uses a mock embedding model so no HuggingFace Hub download is needed.
Integration tests that actually call the model are marked @pytest.mark.integration
and are skipped in CI unless HF_INTEGRATION=1 is set.
"""

import os
import math
import pytest
from unittest.mock import MagicMock, patch


# ── Mock embedding fixture ────────────────────────────────────────────────────

DIM = 384   # dimension of all-MiniLM-L6-v2

def _make_mock_emb():
    """Return a mock that satisfies the HuggingFaceEmbeddings interface."""
    import random
    mock = MagicMock()

    def _fake_query(text: str):
        # Deterministic vector based on text hash, unit-normalised
        seed = hash(text) % (2 ** 31)
        random.seed(seed)
        raw = [random.gauss(0, 1) for _ in range(DIM)]
        norm = math.sqrt(sum(x ** 2 for x in raw))
        return [x / norm for x in raw]

    def _fake_docs(texts):
        return [_fake_query(t) for t in texts]

    mock.embed_query.side_effect = _fake_query
    mock.embed_documents.side_effect = _fake_docs
    mock.model_name = "sentence-transformers/all-MiniLM-L6-v2"
    return mock


# ── Unit tests (always run, no network) ──────────────────────────────────────

_PATCH = "app.embeddings.hf_embeddings.HuggingFaceEmbeddings"


def test_embed_query_returns_floats():
    with patch(_PATCH, return_value=_make_mock_emb()):
        from app.embeddings import hf_embeddings as hfe
        hfe._INSTANCE = None                  # reset singleton
        emb = hfe.get_embeddings()
        vec = emb.embed_query("What are Apple's risk factors?")
    assert isinstance(vec, list)
    assert len(vec) == DIM
    assert all(isinstance(v, float) for v in vec)


def test_embed_query_consistent_dimension():
    with patch(_PATCH, return_value=_make_mock_emb()):
        from app.embeddings import hf_embeddings as hfe
        hfe._INSTANCE = None
        emb = hfe.get_embeddings()
        v1 = emb.embed_query("risk factors supply chain")
        v2 = emb.embed_query("revenue growth cloud services")
    assert len(v1) == len(v2) == DIM


def test_embed_documents_batch():
    with patch(_PATCH, return_value=_make_mock_emb()):
        from app.embeddings import hf_embeddings as hfe
        hfe._INSTANCE = None
        emb = hfe.get_embeddings()
        docs = [
            "Apple faces competition in smartphones.",
            "Microsoft Azure cloud revenue grew.",
            "Alphabet advertising revenue trends.",
        ]
        vecs = emb.embed_documents(docs)
    assert len(vecs) == 3
    assert all(len(v) == DIM for v in vecs)
    # All same dimension
    assert len({len(v) for v in vecs}) == 1


def test_singleton_reuse():
    mock_instance = _make_mock_emb()
    with patch(_PATCH, return_value=mock_instance):
        from app.embeddings import hf_embeddings as hfe
        hfe._INSTANCE = None
        emb1 = hfe.get_embeddings()
        emb2 = hfe.get_embeddings()
    assert emb1 is emb2, "get_embeddings() must return the same singleton"


def test_normalized_embeddings():
    with patch(_PATCH, return_value=_make_mock_emb()):
        from app.embeddings import hf_embeddings as hfe
        hfe._INSTANCE = None
        emb = hfe.get_embeddings()
        vec = emb.embed_query("test normalization")
    norm = math.sqrt(sum(v ** 2 for v in vec))
    assert abs(norm - 1.0) < 0.05, f"Expected unit norm, got {norm:.4f}"


def test_different_texts_different_vectors():
    with patch(_PATCH, return_value=_make_mock_emb()):
        from app.embeddings import hf_embeddings as hfe
        hfe._INSTANCE = None
        emb = hfe.get_embeddings()
        v1 = emb.embed_query("apple risk factors")
        v2 = emb.embed_query("microsoft cloud revenue")
    assert v1 != v2, "Distinct texts should produce distinct vectors"


# ── Integration marker (skipped unless HF_INTEGRATION=1) ─────────────────────

@pytest.mark.skipif(
    os.getenv("HF_INTEGRATION") != "1",
    reason="Requires HuggingFace network access. Set HF_INTEGRATION=1 to run.",
)
def test_real_embedding_model():
    """Integration test: actually loads all-MiniLM-L6-v2 from HuggingFace."""
    from app.embeddings import hf_embeddings as hfe
    hfe._INSTANCE = None
    emb = hfe.get_embeddings()
    vec = emb.embed_query("SEC risk factors disclosure")
    assert isinstance(vec, list)
    assert len(vec) == DIM
