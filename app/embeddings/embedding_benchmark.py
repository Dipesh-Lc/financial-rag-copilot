"""
embedding_benchmark.py
Compare multiple embedding models on a small retrieval benchmark.
Used in notebooks/03_embedding_benchmark.ipynb.
"""

from __future__ import annotations

import time
from typing import Callable

import numpy as np


CANDIDATE_MODELS = [
    "sentence-transformers/all-MiniLM-L6-v2",       # fast, small
    "sentence-transformers/all-mpnet-base-v2",       # higher quality
    "BAAI/bge-small-en-v1.5",                        # strong for retrieval
]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    va, vb = np.array(a), np.array(b)
    return float(np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb) + 1e-9))


def benchmark_model(
    model_name: str,
    queries: list[str],
    corpus: list[str],
    relevant_indices: list[list[int]],
) -> dict:
    """
    Compute hit-rate@5 and average latency for a given embedding model.
    Returns a dict with model_name, hit_rate, avg_latency_ms.
    """
    from app.embeddings.hf_embeddings import get_embeddings

    emb = get_embeddings(model_name)

    t0 = time.perf_counter()
    corpus_vecs = emb.embed_documents(corpus)
    index_time = time.perf_counter() - t0

    hits = 0
    query_times = []
    for q, relevant in zip(queries, relevant_indices):
        t1 = time.perf_counter()
        q_vec = emb.embed_query(q)
        query_times.append(time.perf_counter() - t1)

        scores = [cosine_similarity(q_vec, c) for c in corpus_vecs]
        top5 = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:5]
        if any(r in top5 for r in relevant):
            hits += 1

    return {
        "model_name": model_name,
        "hit_rate_at_5": hits / len(queries) if queries else 0.0,
        "avg_query_latency_ms": round(np.mean(query_times) * 1000, 2),
        "index_time_s": round(index_time, 3),
    }
