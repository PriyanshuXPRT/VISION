"""Tests for the brute-force vector index."""
from __future__ import annotations

import numpy as np

from vision.ai.embedding_search import BruteForceIndex


def _normed(rng, n, d):
    x = rng.standard_normal((n, d)).astype(np.float32)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def test_empty_index_returns_nothing():
    idx = BruteForceIndex(8)
    assert idx.ntotal() == 0
    assert idx.search(np.ones(8, dtype=np.float32)) == []


def test_search_returns_correct_match():
    rng = np.random.default_rng(0)
    embs = _normed(rng, 50, 16)
    idx = BruteForceIndex(16)
    idx.add(list(range(50)), embs)
    assert idx.ntotal() == 50
    q = embs[7]
    hits = idx.search(q, top_k=3)
    assert hits[0].id == 7
    assert hits[0].score == 1.0


def test_remove_then_search():
    rng = np.random.default_rng(1)
    embs = _normed(rng, 20, 16)
    idx = BruteForceIndex(16)
    idx.add(list(range(20)), embs)
    idx.remove([3, 4])
    assert idx.ntotal() == 18
    hits = idx.search(embs[3], top_k=1)
    assert hits[0].id != 3 and hits[0].id != 4
