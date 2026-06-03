"""ArcFace 512-D face recognition (InsightFace R100, ONNX)."""
from __future__ import annotations

import hashlib
import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from numpy.typing import NDArray

from vision.config import settings
from vision.core.exceptions import InferenceError, ModelNotFoundError
from vision.core.logging import logger
from vision.core.math_utils import l2_normalize
from vision.core.onnx_runner import OnnxRunner

# InsightFace buffalo_l ArcFace (R100, MS1M, 512-D)
_ARCFACE_URL = (
    "https://github.com/deepinsight/insightface/releases/download/v0.7/arcface_r100.onnx"
)
_ARCFACE_SHA256 = ""  # left empty; weight file is signed by upstream


def _download(url: str, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading {} -> {}", url, dst)
    with urllib.request.urlopen(url, timeout=60) as resp, open(dst, "wb") as fh:
        while True:
            chunk = resp.read(1 << 16)
            if not chunk:
                break
            fh.write(chunk)
    if _ARCFACE_SHA256:
        h = hashlib.sha256(dst.read_bytes()).hexdigest()
        if h != _ARCFACE_SHA256:
            dst.unlink(missing_ok=True)
            raise ModelNotFoundError(f"SHA256 mismatch for {dst}")


@dataclass(slots=True)
class FaceTemplate:
    user_id: int
    template_id: int
    embedding: NDArray[np.float32]   # L2-normalised
    quality_score: float


class FaceRecognizer:
    """ArcFace embedder + similarity / 1:N identification."""

    def __init__(
        self,
        model_path: str | Path | None = None,
        *,
        input_size: int | None = None,
        download: bool | None = None,
    ) -> None:
        self.input_size = input_size or settings.arcface_input_size
        path = Path(model_path or settings.arcface_path)
        if not path.is_file() and (download if download is not None else settings.download_pretrained):
            try:
                _download(_ARCFACE_URL, path)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Could not download ArcFace weights: {}", exc)
        if not path.is_file():
            raise ModelNotFoundError(f"ArcFace model not found: {path}")
        self.runner = OnnxRunner(path)
        self.runner.warmup()
        logger.info("FaceRecognizer ready · {} · dim={}", path.name, settings.embedding_dim)

    # ---------------------------------------------------------------- embed
    def generate_embedding(
        self,
        aligned_face: NDArray[np.uint8],
    ) -> NDArray[np.float32]:
        """Return a 512-D L2-normalised embedding for an aligned (112, 112, 3) RGB face."""
        if aligned_face.shape[:2] != (self.input_size, self.input_size):
            aligned_face = cv2.resize(
                aligned_face, (self.input_size, self.input_size), interpolation=cv2.INTER_LINEAR
            )
        x = aligned_face.astype(np.float32)
        x = (x - 127.5) / 127.5
        x = np.transpose(x, (2, 0, 1))[None, ...]
        try:
            outs = self.runner.run({self.runner.input_names[0]: x})
        except Exception as exc:  # noqa: BLE001
            raise InferenceError(f"ArcFace inference failed: {exc}") from exc
        emb = np.array(outs[0]).squeeze()
        if emb.ndim > 1:
            emb = emb[0]
        return l2_normalize(emb.astype(np.float32))

    # ---------------------------------------------------------------- compare
    def compare_embeddings(
        self,
        a: NDArray[np.float32],
        b: NDArray[np.float32],
    ) -> float:
        """Cosine similarity, 0..1. Assumes L2-normalised embeddings."""
        return float(np.dot(a.astype(np.float32), b.astype(np.float32)))

    def identify_user(
        self,
        embedding: NDArray[np.float32],
        templates: list[FaceTemplate],
        *,
        threshold: float | None = None,
        top_k: int | None = None,
    ) -> tuple[Optional[FaceTemplate], list[tuple[FaceTemplate, float]]]:
        """1:N identification by cosine similarity.

        Returns (best_match_or_None, ranked_top_k).
        """
        thresh = threshold or settings.recognition_threshold
        k = top_k or settings.recognition_top_k
        if not templates:
            return None, []
        embeds = np.stack([t.embedding for t in templates], axis=0)
        embeds = embeds / (np.linalg.norm(embeds, axis=1, keepdims=True) + 1e-12)
        sims = embeds @ embedding.astype(np.float32)
        order = np.argsort(-sims)[:k]
        ranked = [(templates[i], float(sims[i])) for i in order]
        best_t, best_s = ranked[0]
        return (best_t if best_s >= thresh else None), ranked

    # ---------------------------------------------------------------- DB-facing
    def register_embedding(
        self,
        embeddings: list[NDArray[np.float32]],
        *,
        user_id: int,
        template_id: int,
        quality_scores: list[float] | None = None,
    ) -> FaceTemplate:
        """Aggregate multiple embeddings of one user into a single template.

        Strategy: mean-of-L2-normalised, re-normalise. Drop outliers (>1.5*IQR).
        """
        if not embeddings:
            raise ValueError("register_embedding requires at least 1 embedding")
        arr = np.stack([l2_normalize(e) for e in embeddings], axis=0)
        # Outlier rejection
        med = np.median(arr, axis=0, keepdims=True)
        dists = 1.0 - (arr @ med.T).squeeze()
        if dists.size >= 4:
            q1, q3 = np.percentile(dists, [25, 75])
            iqr = q3 - q1
            keep = dists <= (q3 + 1.5 * iqr)
            if keep.sum() >= 1:
                arr = arr[keep]
        mean = arr.mean(axis=0)
        template = l2_normalize(mean)
        quality = float(np.mean(quality_scores)) if quality_scores else 0.0
        return FaceTemplate(
            user_id=user_id,
            template_id=template_id,
            embedding=template,
            quality_score=quality,
        )
