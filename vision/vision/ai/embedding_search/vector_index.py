"""
Vector search backend for 1:N identification.

Two interchangeable implementations:
- `FaissIndex`     — fast, in-memory, IVF/Flat/HNSW
- `BruteForceIndex`— pure-numpy fallback (no external dep, used in tests/edge)
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from vision.core.logging import logger


@dataclass(slots=True)
class SearchHit:
    id: int
    score: float


class VectorIndex(ABC):
    @abstractmethod
    def add(self, ids: Sequence[int], embeddings: NDArray[np.float32]) -> None: ...

    @abstractmethod
    def remove(self, ids: Sequence[int]) -> None: ...

    @abstractmethod
    def search(self, query: NDArray[np.float32], top_k: int = 5) -> list[SearchHit]: ...

    @abstractmethod
    def ntotal(self) -> int: ...


# -----------------------------------------------------------------------------
# Pure-numpy fallback
# -----------------------------------------------------------------------------
class BruteForceIndex(VectorIndex):
    def __init__(self, dim: int) -> None:
        self.dim = dim
        self._ids: list[int] = []
        self._matrix: NDArray[np.float32] = np.zeros((0, dim), dtype=np.float32)

    def add(self, ids: Sequence[int], embeddings: NDArray[np.float32]) -> None:
        embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
        if embeddings.ndim != 2 or embeddings.shape[1] != self.dim:
            raise ValueError(f"Expected (N, {self.dim})")
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12
        embeddings = embeddings / norms
        if not self._ids:
            self._matrix = embeddings
        else:
            self._matrix = np.concatenate([self._matrix, embeddings], axis=0)
        self._ids.extend(ids)

    def remove(self, ids: Sequence[int]) -> None:
        id_set = set(ids)
        keep = [i for i, _id in enumerate(self._ids) if _id not in id_set]
        self._ids = [self._ids[i] for i in keep]
        if keep:
            self._matrix = self._matrix[keep]
        else:
            self._matrix = np.zeros((0, self.dim), dtype=np.float32)

    def search(self, query: NDArray[np.float32], top_k: int = 5) -> list[SearchHit]:
        if self.ntotal() == 0:
            return []
        q = query.astype(np.float32) / (np.linalg.norm(query) + 1e-12)
        sims = self._matrix @ q
        order = np.argsort(-sims)[:top_k]
        return [SearchHit(id=self._ids[i], score=float(sims[i])) for i in order]

    def ntotal(self) -> int:
        return len(self._ids)


# -----------------------------------------------------------------------------
# FAISS
# -----------------------------------------------------------------------------
class FaissIndex(VectorIndex):
    """Wraps faiss. Optional dep — imports lazily."""

    def __init__(self, dim: int, *, index_type: str = "Flat", nlist: int = 64) -> None:
        try:
            import faiss  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("faiss-cpu is not installed") from exc
        self.dim = dim
        self._faiss = faiss
        self._ids: list[int] = []
        if index_type == "Flat":
            self._index = faiss.IndexFlatIP(dim)
        elif index_type == "IVFFlat":
            quantiser = faiss.IndexFlatIP(dim)
            self._index = faiss.IndexIVFFlat(quantiser, dim, nlist, faiss.METRIC_INNER_PRODUCT)
        elif index_type == "HNSW":
            self._index = faiss.IndexHNSWFlat(dim, 32, faiss.METRIC_INNER_PRODUCT)
        else:
            raise ValueError(f"Unknown index_type: {index_type}")
        # Inner-product requires normalised vectors
        self._index.metric_type = faiss.METRIC_INNER_PRODUCT

    def add(self, ids: Sequence[int], embeddings: NDArray[np.float32]) -> None:
        embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
        if embeddings.ndim != 2 or embeddings.shape[1] != self.dim:
            raise ValueError(f"Expected (N, {self.dim})")
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12
        embeddings = embeddings / norms
        if isinstance(self._index, self._faiss.IndexIVFFlat) and not self._index.is_trained:
            self._index.train(embeddings)
        self._index.add(embeddings)
        self._ids.extend(ids)

    def remove(self, ids: Sequence[int]) -> None:
        if not ids:
            return
        keep = [i for i, _id in enumerate(self._ids) if _id not in set(ids)]
        dropped = [i for i, _id in enumerate(self._ids) if _id in set(ids)]
        if not dropped:
            return
        # Re-build (faiss has no remove for Flat/IVFFlat)
        logger.warning("FaissIndex.remove is O(N) — rebuilding index without {} ids", len(dropped))
        kept_matrix = (
            np.zeros((len(keep), self.dim), dtype=np.float32) if keep else np.zeros((0, self.dim), dtype=np.float32)
        )
        new_ids: list[int] = []
        # We have no original matrix; need to re-add from outside
        raise NotImplementedError(
            "FaissIndex.remove requires re-adding kept vectors. Use the repository wrapper."
        )

    def search(self, query: NDArray[np.float32], top_k: int = 5) -> list[SearchHit]:
        if self.ntotal() == 0:
            return []
        q = query.astype(np.float32) / (np.linalg.norm(query) + 1e-12)
        scores, idxs = self._index.search(q[None, :], top_k)
        hits: list[SearchHit] = []
        for s, i in zip(scores[0], idxs[0]):
            if i < 0:
                continue
            hits.append(SearchHit(id=self._ids[i], score=float(s)))
        return hits

    def ntotal(self) -> int:
        return self._index.ntotal


# -----------------------------------------------------------------------------
# Factory
# -----------------------------------------------------------------------------
def build_index(dim: int, *, prefer: str = "faiss", **kwargs) -> VectorIndex:
    if prefer == "faiss":
        try:
            return FaissIndex(dim, **kwargs)
        except Exception as exc:  # noqa: BLE001
            logger.warning("FAISS unavailable ({}), falling back to brute force", exc)
    return BruteForceIndex(dim)
