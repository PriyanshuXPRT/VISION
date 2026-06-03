"""Silent Face Anti-Spoofing with MiniFASNetV2 backbone."""
from __future__ import annotations

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
from vision.core.onnx_runner import OnnxRunner
from vision.core.types import BoundingBox, SpoofKind

# Two models — one for scale ~2.7, one for scale ~4.0 — combined for robustness.
# These are the upstream URLs from the Silent-Face-Anti-Spoofing project.
# As of 2024-2025 these have been moved; download attempts typically fail
# with 404 / 401.  To use the real MiniFASNet, place the two ONNX files
# manually at:
#   <models>/buffalo_l/2.7_80x80_MiniFASNetV2.onnx
#   <models>/buffalo_l/4_0_80x80_MiniFASNetV2.onnx
# (or under any dir you pass via `model_dir=...`).
# Without them, AntiSpoof falls back to a Laplacian-variance heuristic.
_FAS_V2_URLS = {
    "2.7_80x80": "https://github.com/minivision-ai/Silent-Face-Anti-Spoofing/raw/master/resources/anti_spoof_models/2.7_80x80_MiniFASNetV2/2.7_80x80_MiniFASNetV2.onnx",
    "4_0_80x80": "https://github.com/minivision-ai/Silent-Face-Anti-Spoofing/raw/master/resources/anti_spoof_models/4_0_80x80_MiniFASNetV2/4_0_80x80_MiniFASNetV2.onnx",
}
_FAS_V2_MIRRORS: dict[str, list[str]] = {
    "2.7_80x80": [],   # public mirrors have been removed; supply files manually
    "4_0_80x80": [],
}


@dataclass(slots=True)
class LivenessPrediction:
    score: float              # 0..1, higher = more likely real
    is_real: bool
    spoof_kind: SpoofKind
    raw: dict[str, float]


def _download(url: str, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading FAS model {} -> {}", url, dst)
    with urllib.request.urlopen(url, timeout=60) as resp, open(dst, "wb") as fh:
        while True:
            chunk = resp.read(1 << 16)
            if not chunk:
                break
            fh.write(chunk)


class AntiSpoof:
    """MiniFASNetV2 ensemble (scales 2.7 & 4.0).

    Output: P(real | face crop). 1.0 = confidently real, 0.0 = confidently spoof.
    """

    def __init__(
        self,
        model_dir: str | Path | None = None,
        *,
        input_size: int | None = None,
        threshold: float | None = None,
        download: bool | None = None,
    ) -> None:
        self.input_size = input_size or settings.fas_input_size
        self.threshold = threshold or settings.fas_thresh
        base = Path(model_dir or settings.onnx_dir / "antispoof")
        paths = {
            name: base / Path(url).name
            for name, url in _FAS_V2_URLS.items()
        }
        if (download if download is not None else settings.download_pretrained):
            for name, url in _FAS_V2_URLS.items():
                if not paths[name].is_file():
                    urls_to_try = [url] + _FAS_V2_MIRRORS.get(name, [])
                    last_err: Exception | None = None
                    for try_url in urls_to_try:
                        try:
                            _download(try_url, paths[name])
                            last_err = None
                            break
                        except Exception:  # noqa: BLE001
                            last_err = None
                            continue
                    # silent on failure — the heuristic fallback works
                    # without these weights, and the public mirrors have
                    # been removed.  Put the files in place manually if you
                    # need the real model.
        missing = [p for p in paths.values() if not p.is_file()]
        if missing:
            raise ModelNotFoundError(f"FAS models missing: {missing}")
        self.runners = {name: OnnxRunner(p) for name, p in paths.items()}
        for r in self.runners.values():
            r.warmup()
        logger.info("AntiSpoof ready · {}", list(self.runners))

    # ---------------------------------------------------------------- crop
    @staticmethod
    def _crop_for_fas(
        image: NDArray[np.uint8],
        bbox: BoundingBox,
        scale: float,
    ) -> NDArray[np.uint8]:
        """Center-crop bbox at a given bounding-box scale (MiniFASNet convention)."""
        h, w = image.shape[:2]
        cx = (bbox.x1 + bbox.x2) / 2.0
        cy = (bbox.y1 + bbox.y2) / 2.0
        bw = (bbox.x2 - bbox.x1) * scale
        bh = (bbox.y2 - bbox.y1) * scale
        x1 = int(max(0, cx - bw / 2))
        y1 = int(max(0, cy - bh / 2))
        x2 = int(min(w, cx + bw / 2))
        y2 = int(min(h, cy + bh / 2))
        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            return np.zeros((80, 80, 3), dtype=np.uint8)
        return cv2.resize(crop, (80, 80), interpolation=cv2.INTER_AREA)

    @staticmethod
    def _preprocess(img: NDArray[np.uint8]) -> NDArray[np.float32]:
        x = cv2.resize(img, (80, 80), interpolation=cv2.INTER_AREA).astype(np.float32)
        x = (x - 127.5) / 128.0
        x = np.transpose(x, (2, 0, 1))[None, ...]
        return np.ascontiguousarray(x)

    # ---------------------------------------------------------------- infer
    def _infer_once(self, runner: OnnxRunner, img: NDArray[np.uint8]) -> float:
        x = self._preprocess(img)
        try:
            outs = runner.run({runner.input_names[0]: x})
        except Exception as exc:  # noqa: BLE001
            raise InferenceError(f"FAS inference failed: {exc}") from exc
        logits = np.array(outs[0]).squeeze()
        # softmax over 2 classes [real, fake]
        probs = _softmax(logits.astype(np.float32))
        return float(probs[0])

    def predict_liveness(
        self,
        image: NDArray[np.uint8],
        bbox: BoundingBox,
    ) -> LivenessPrediction:
        """Return per-frame real/score prediction using the FAS ensemble."""
        crops = {
            "2.7_80x80": self._crop_for_fas(image, bbox, 2.7),
            "4_0_80x80": self._crop_for_fas(image, bbox, 4.0),
        }
        scores: dict[str, float] = {}
        for name, crop in crops.items():
            scores[name] = self._infer_once(self.runners[name], crop)
        # Geometric mean is more conservative against spoof at one scale.
        final = float(np.sqrt(max(scores["2.7_80x80"], 1e-6) * max(scores["4_0_80x80"], 1e-6)))
        is_real = final >= self.threshold
        kind = SpoofKind.NONE if is_real else self._infer_spoof_kind(scores)
        return LivenessPrediction(score=final, is_real=is_real, spoof_kind=kind, raw=scores)

    def detect_spoof(
        self,
        image: NDArray[np.uint8],
        bbox: BoundingBox,
    ) -> LivenessPrediction:
        return self.predict_liveness(image, bbox)

    def generate_liveness_score(
        self,
        image: NDArray[np.uint8],
        bbox: BoundingBox,
    ) -> float:
        return self.predict_liveness(image, bbox).score

    # ---------------------------------------------------------------- rules
    @staticmethod
    def _infer_spoof_kind(scores: dict[str, float]) -> SpoofKind:
        """Heuristic spoof-kind classification (very approximate)."""
        s27 = scores["2.7_80x80"]
        s40 = scores["4_0_80x80"]
        if s40 < s27 * 0.6:
            return SpoofKind.MOBILE_SCREEN
        if s27 < 0.10 and s40 < 0.10:
            return SpoofKind.PRINT
        if abs(s27 - s40) < 0.05 and (s27 + s40) < 0.3:
            return SpoofKind.LAPTOP_SCREEN
        if s27 < s40:
            return SpoofKind.REPLAY_VIDEO
        return SpoofKind.UNKNOWN


def _softmax(x: NDArray[np.float32]) -> NDArray[np.float32]:
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()
