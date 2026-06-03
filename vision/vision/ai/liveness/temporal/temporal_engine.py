"""
Temporal Liveness Engine.

Fuses blink, eye-motion, head-pose, and per-frame passive-FAS scores over a
sliding window of frames, and emits a single temporal liveness score.

Two backends:
  - HeuristicEngine: pure-Python fallback, used in tests and the Android
    preview (no ONNX needed).
  - OnnxTemporalEngine: loads a trained LSTM/Transformer exported to ONNX
    and runs it on the device.

Feature layout (D=8):
  [0] passive_liveness
  [1] ear
  [2] eye_motion
  [3] |yaw| / 30
  [4] |pitch| / 20
  [5] |roll| / 20
  [6] head_motion_rms (1-second window)
  [7] blink_rate_per_sec
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from vision.config import settings
from vision.core.exceptions import InferenceError, ModelNotFoundError
from vision.core.logging import logger
from vision.core.onnx_runner import OnnxRunner


# -----------------------------------------------------------------------------
# Feature extraction
# -----------------------------------------------------------------------------
FEATURE_DIM = 8


def build_feature_vector(
    *,
    passive_liveness: float,
    ear: float,
    eye_motion: float,
    yaw: float,
    pitch: float,
    roll: float,
    head_motion_rms: float,
    blink_rate_per_sec: float,
) -> NDArray[np.float32]:
    return np.array(
        [
            float(np.clip(passive_liveness, 0.0, 1.0)),
            float(np.clip(ear, 0.0, 0.5)),
            float(np.clip(eye_motion, 0.0, 1.0)),
            float(np.clip(abs(yaw) / 30.0, 0.0, 1.0)),
            float(np.clip(abs(pitch) / 20.0, 0.0, 1.0)),
            float(np.clip(abs(roll) / 20.0, 0.0, 1.0)),
            float(np.clip(head_motion_rms, 0.0, 1.0)),
            float(np.clip(blink_rate_per_sec / 2.0, 0.0, 1.0)),
        ],
        dtype=np.float32,
    )


# -----------------------------------------------------------------------------
# Heuristic backend
# -----------------------------------------------------------------------------
@dataclass(slots=True)
class TemporalResult:
    final_score: float         # 0..1 — probability the sequence is a real person
    is_live: bool
    rationale: list[str]


class HeuristicEngine:
    """Score function over the window; no learned weights required."""

    def __init__(
        self,
        *,
        seq_len: int | None = None,
        stride: int | None = None,
        liveness_threshold: float | None = None,
    ) -> None:
        self.seq_len = seq_len or settings.temporal_seq_len
        self.stride = stride or settings.temporal_stride
        self.thresh = liveness_threshold or settings.liveness_threshold
        self._buf: deque[NDArray[np.float32]] = deque(maxlen=512)

    def reset(self) -> None:
        self._buf.clear()

    def push(self, feature: NDArray[np.float32]) -> None:
        if feature.shape != (FEATURE_DIM,):
            raise ValueError(f"Feature must be ({FEATURE_DIM},)")
        self._buf.append(feature.astype(np.float32))

    def is_ready(self) -> bool:
        return len(self._buf) >= self.seq_len

    def score(self) -> TemporalResult:
        if not self.is_ready():
            return TemporalResult(0.0, False, ["buffer not full"])
        seq = np.stack(list(self._buf)[-self.seq_len :: max(1, self.stride)], axis=0)
        if seq.shape[0] < 4:
            return TemporalResult(0.0, False, ["not enough sampled frames"])
        rationale: list[str] = []
        passive = float(seq[:, 0].mean())
        ear = float(seq[:, 1].mean())
        eye = float(seq[:, 2].mean())
        head_var = float(seq[:, 6].mean())
        blink = float(seq[:, 7].mean())

        # Sub-scores
        s_passive = passive
        s_blink = 1.0 - math.exp(-2.5 * blink)            # reward some blinking
        s_eye = eye
        s_head = 1.0 - math.exp(-3.0 * head_var)          # reward head motion
        # Penalise too-still sequences (spoof often has no micro-motion)
        s_motion = 0.5 * (s_head + s_eye)

        final = (
            0.45 * s_passive
            + 0.15 * s_blink
            + 0.15 * s_eye
            + 0.15 * s_head
            + 0.10 * s_motion
        )
        final = float(np.clip(final, 0.0, 1.0))
        rationale.extend(
            [
                f"passive={passive:.2f}",
                f"blink_rate={blink:.2f}",
                f"eye_motion={eye:.2f}",
                f"head_var={head_var:.2f}",
            ]
        )
        return TemporalResult(
            final_score=final,
            is_live=final >= self.thresh,
            rationale=rationale,
        )


# -----------------------------------------------------------------------------
# ONNX temporal model
# -----------------------------------------------------------------------------
class OnnxTemporalEngine:
    """Run a trained LSTM/Transformer exported to ONNX.

    Expected input shape: (1, seq_len, FEATURE_DIM) float32
    Expected output: (1, 1) float32 (sigmoid prob)
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        *,
        seq_len: int | None = None,
        stride: int | None = None,
    ) -> None:
        self.seq_len = seq_len or settings.temporal_seq_len
        self.stride = stride or settings.temporal_stride
        path = Path(model_path or settings.temporal_path)
        if not path.is_file():
            raise ModelNotFoundError(f"Temporal model not found: {path}")
        self.runner = OnnxRunner(path)
        self._buf: deque[NDArray[np.float32]] = deque(maxlen=512)
        logger.info("OnnxTemporalEngine ready · {}", path.name)

    def reset(self) -> None:
        self._buf.clear()

    def push(self, feature: NDArray[np.float32]) -> None:
        if feature.shape != (FEATURE_DIM,):
            raise ValueError(f"Feature must be ({FEATURE_DIM},)")
        self._buf.append(feature.astype(np.float32))

    def is_ready(self) -> bool:
        return len(self._buf) >= self.seq_len

    def score(self) -> TemporalResult:
        if not self.is_ready():
            return TemporalResult(0.0, False, ["buffer not full"])
        seq = np.stack(list(self._buf)[-self.seq_len :: max(1, self.stride)], axis=0)
        if seq.shape[0] != self.seq_len:
            # pad
            pad = np.zeros((self.seq_len - seq.shape[0], FEATURE_DIM), dtype=np.float32)
            seq = np.concatenate([seq, pad], axis=0)
        x = seq[None, ...].astype(np.float32)
        try:
            outs = self.runner.run({self.runner.input_names[0]: x})
        except Exception as exc:  # noqa: BLE001
            raise InferenceError(f"Temporal inference failed: {exc}") from exc
        y = float(np.array(outs[0]).squeeze())
        y = float(np.clip(y, 0.0, 1.0))
        return TemporalResult(final_score=y, is_live=y >= 0.5, rationale=["onnx_temporal"])


# -----------------------------------------------------------------------------
# Aggregate
# -----------------------------------------------------------------------------
class TemporalLivenessEngine:
    """Combines Heuristic + (optional) ONNX for a robust score."""

    def __init__(self, *, prefer_onnx: bool = True) -> None:
        self.heuristic = HeuristicEngine()
        self.onnx_engine: Optional[OnnxTemporalEngine] = None
        if prefer_onnx:
            try:
                self.onnx_engine = OnnxTemporalEngine()
            except Exception as exc:  # noqa: BLE001
                logger.warning("ONNX temporal model unavailable: {}; using heuristic", exc)

    def reset(self) -> None:
        self.heuristic.reset()
        if self.onnx_engine is not None:
            self.onnx_engine.reset()

    def push(self, feature: NDArray[np.float32]) -> None:
        self.heuristic.push(feature)
        if self.onnx_engine is not None:
            self.onnx_engine.push(feature)

    def is_ready(self) -> bool:
        return self.heuristic.is_ready()

    def score(self) -> TemporalResult:
        h = self.heuristic.score()
        if self.onnx_engine is None:
            return h
        o = self.onnx_engine.score()
        # Weighted average — heuristic still grounds when ONNX is uncertain.
        final = 0.5 * h.final_score + 0.5 * o.final_score
        return TemporalResult(
            final_score=float(final),
            is_live=final >= 0.5,
            rationale=[*h.rationale, *o.rationale],
        )
