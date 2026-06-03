"""
1:N identification service.

Wraps a `VectorIndex` (FAISS or brute force) backed by face templates stored
in SQLite. Rebuilds the index in-memory at startup and on demand.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from vision.ai import VectorIndex, build_index
from vision.ai.recognition import FaceRecognizer
from vision.config import settings
from vision.core.exceptions import VisionError
from vision.core.logging import logger
from vision.core.types import IdentificationResult, UserMatch
from vision.database import Database, FaceTemplateRepository, UserRepository


@dataclass(slots=True)
class IndexEntry:
    user_id: int
    template_id: int
    name: str


class IdentificationService:
    def __init__(
        self,
        db: Database,
        recognizer: FaceRecognizer,
        *,
        prefer: str = "faiss",
    ) -> None:
        self.db = db
        self.recognizer = recognizer
        self.users = UserRepository(db)
        self.templates = FaceTemplateRepository(db)
        self.index: VectorIndex = build_index(settings.embedding_dim, prefer=prefer)
        self._entries: list[IndexEntry] = []
        self._lock = threading.RLock()
        self.rebuild()

    # ---------------------------------------------------------------- index
    def rebuild(self) -> None:
        with self._lock:
            templates = self.templates.list_all()
            self.index = build_index(settings.embedding_dim, prefer=type(self.index).__name__.lower().replace("index", ""))
            self._entries.clear()
            if not templates:
                logger.info("Identification index built · 0 templates")
                return
            ids: list[int] = []
            embs: list[NDArray[np.float32]] = []
            for t in templates:
                user = self._safe_user(t.user_id)
                if user is None or not user.is_active:
                    continue
                self._entries.append(IndexEntry(user_id=t.user_id, template_id=t.template_id, name=user.name))
                ids.append(t.user_id)
                embs.append(t.embedding)
            ids_arr = list(range(len(self._entries)))
            if embs:
                self.index.add(ids_arr, np.stack(embs, axis=0))
                # Note: we store sequential ids in the index and map back via _entries
                self._sequential_id_to_entry = {i: e for i, e in enumerate(self._entries)}
            logger.info("Identification index built · {} active templates", len(self._entries))

    def add_template(self, template_id: int) -> None:
        with self._lock:
            tpl = self.templates.get(template_id)
            user = self._safe_user(tpl.user_id)
            if user is None or not user.is_active:
                return
            self._entries.append(IndexEntry(user_id=tpl.user_id, template_id=tpl.template_id, name=user.name))
            sequential_id = len(self._entries) - 1
            self.index.add([sequential_id], tpl.embedding[None, :])

    def remove_user(self, user_id: int) -> None:
        with self._lock:
            self.templates.delete_for_user(user_id)
            self.rebuild()

    # ---------------------------------------------------------------- identify
    def identify(
        self,
        embedding: NDArray[np.float32],
        *,
        top_k: int | None = None,
        threshold: float | None = None,
    ) -> IdentificationResult:
        top_k = top_k or settings.recognition_top_k
        threshold = threshold or settings.recognition_threshold
        with self._lock:
            if self.index.ntotal() == 0:
                return IdentificationResult()
            hits = self.index.search(embedding, top_k=top_k)
        candidates: list[UserMatch] = []
        for h in hits:
            entry = self._entries[h.id]
            candidates.append(
                UserMatch(
                    user_id=entry.user_id,
                    name=entry.name,
                    similarity=float(h.score),
                    template_id=entry.template_id,
                )
            )
        best = candidates[0] if candidates else None
        is_identified = bool(best and best.similarity >= threshold)
        return IdentificationResult(
            best_match=best if is_identified else None,
            candidates=candidates,
            is_identified=is_identified,
        )

    # ---------------------------------------------------------------- helpers
    def _safe_user(self, user_id: int):
        try:
            return self.users.get(user_id)
        except VisionError:
            return None
