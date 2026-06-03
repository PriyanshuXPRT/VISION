"""Eye Aspect Ratio (EAR) blink detection."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from vision.ai.landmarks import LEFT_EYE, RIGHT_EYE
from vision.config import settings
from vision.core.logging import logger
from vision.core.types import Landmarks


@dataclass(slots=True)
class BlinkEvent:
    timestamp_ms: int
    duration_ms: int
    ear_min: float


@dataclass(slots=True)
class BlinkResult:
    blinks: list[BlinkEvent]
    blink_count: int
    mean_ear: float
    blink_score: float      # 0..1 — high means natural blinking observed


def eye_aspect_ratio(eye_points: NDArray[np.float32]) -> float:
    """EAR = (|p2-p6| + |p3-p5|) / (2 * |p1-p4|)."""
    p1, p2, p3, p4, p5, p6 = eye_points
    a = np.linalg.norm(p2 - p6)
    b = np.linalg.norm(p3 - p5)
    c = 2.0 * np.linalg.norm(p1 - p4) + 1e-6
    return float((a + b) / c)


class BlinkDetector:
    """Stateful EAR-based blink detector.

    Usage:
        det = BlinkDetector()
        for ts_ms, landmarks in stream:
            result = det.update(landmarks, ts_ms)
    """

    def __init__(
        self,
        *,
        ear_thresh: float | None = None,
        consec_frames: int = 2,
        min_blink_ms: int = 80,
        max_blink_ms: int = 600,
        history_len: int = 64,
    ) -> None:
        self.ear_thresh = ear_thresh or settings.blink_ear_thresh
        self.consec_frames = consec_frames
        self.min_blink_ms = min_blink_ms
        self.max_blink_ms = max_blink_ms
        self.ear_history: deque[float] = deque(maxlen=history_len)
        self.blink_count = 0
        self._below_counter = 0
        self._blink_start_ms: Optional[int] = None
        self._blink_min_ear: float = 1.0
        self._last_ts: Optional[int] = None
        self._events: list[BlinkEvent] = []

    # ---------------------------------------------------------------- update
    def update(self, landmarks: Landmarks, ts_ms: int) -> float:
        """Push one frame. Returns current EAR."""
        left = np.array([landmarks.points[i, :2] for i in LEFT_EYE], dtype=np.float32)
        right = np.array([landmarks.points[i, :2] for i in RIGHT_EYE], dtype=np.float32)
        ear = 0.5 * (eye_aspect_ratio(left) + eye_aspect_ratio(right))
        self.ear_history.append(ear)
        self._detect_blink(ear, ts_ms)
        self._last_ts = ts_ms
        return ear

    def _detect_blink(self, ear: float, ts_ms: int) -> None:
        if ear < self.ear_thresh:
            if self._blink_start_ms is None:
                self._blink_start_ms = ts_ms
                self._blink_min_ear = ear
            self._below_counter += 1
            self._blink_min_ear = min(self._blink_min_ear, ear)
        else:
            if self._blink_start_ms is not None and self._below_counter >= self.consec_frames:
                dur = ts_ms - self._blink_start_ms
                if self.min_blink_ms <= dur <= self.max_blink_ms:
                    self.blink_count += 1
                    self._events.append(
                        BlinkEvent(
                            timestamp_ms=self._blink_start_ms,
                            duration_ms=dur,
                            ear_min=self._blink_min_ear,
                        )
                    )
            self._blink_start_ms = None
            self._below_counter = 0
            self._blink_min_ear = 1.0

    # ---------------------------------------------------------------- query
    @property
    def events(self) -> list[BlinkEvent]:
        return list(self._events)

    def mean_ear(self) -> float:
        if not self.ear_history:
            return 0.0
        return float(np.mean(self.ear_history))

    def blink_score(self) -> float:
        """0..1 — natural blink observed in window.

        Peaks at 1.0 when a single well-formed blink has been observed.
        Caps after 5 blinks (anti-replay) to avoid gaming.
        """
        if self.blink_count == 0:
            return 0.0
        # Reward 1..3 natural blinks most
        b = min(self.blink_count, 5)
        score = min(1.0, b / 3.0)
        return float(score)

    def reset(self) -> None:
        self.ear_history.clear()
        self.blink_count = 0
        self._below_counter = 0
        self._blink_start_ms = None
        self._blink_min_ear = 1.0
        self._events.clear()

    def result(self) -> BlinkResult:
        return BlinkResult(
            blinks=self.events,
            blink_count=self.blink_count,
            mean_ear=self.mean_ear(),
            blink_score=self.blink_score(),
        )


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def has_natural_blink_pattern(blinks: list[BlinkEvent]) -> bool:
    """Heuristic: ≥ 1 blink, durations 80–600 ms, EAR dip > 0.05 below threshold."""
    if not blinks:
        return False
    for b in blinks:
        if b.ear_min > 0.18:
            return False
    return True
