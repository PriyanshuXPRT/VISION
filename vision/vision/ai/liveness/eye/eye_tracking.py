"""Eye tracking: pupil motion, drift, saccades, gaze shifts."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from vision.ai.landmarks import LEFT_IRIS, RIGHT_EYE, RIGHT_IRIS
from vision.config import settings
from vision.core.types import Landmarks


@dataclass(slots=True)
class EyeTrackingResult:
    motion_score: float     # 0..1
    drift_px: float
    saccade_count: int
    gaze_shift_count: int
    fixations: int


def _iris_center(landmarks: Landmarks, indices: list[int]) -> Optional[NDArray[np.float32]]:
    if not indices or any(i >= landmarks.points.shape[0] for i in indices):
        return None
    pts = landmarks.points[indices, :2]
    return pts.mean(axis=0)


def _eye_box(landmarks: Landmarks, indices: list[int]) -> Optional[NDArray[np.float32]]:
    if not indices or any(i >= landmarks.points.shape[0] for i in indices):
        return None
    pts = landmarks.points[indices, :2]
    mn = pts.min(axis=0)
    mx = pts.max(axis=0)
    return np.array([mn[0], mn[1], mx[0], mx[1]], dtype=np.float32)


class EyeTracker:
    """Tracks iris motion within the eye bounding box.

    Produces:
      - motion_score: 0..1 — natural eye motion observed
      - drift_px: cumulative iris drift across the window
      - saccade_count: fast eye jumps
      - gaze_shift_count: large eye-box relative motion
    """

    def __init__(
        self,
        *,
        motion_thresh: float | None = None,
        saccade_velocity_px: float = 8.0,
        gaze_shift_norm: float = 0.15,
        history_len: int = 64,
    ) -> None:
        self.motion_thresh = motion_thresh or settings.eye_motion_thresh
        self.saccade_vel = saccade_velocity_px
        self.gaze_shift_norm = gaze_shift_norm
        self._iris_history: deque[NDArray[np.float32]] = deque(maxlen=history_len)
        self._eye_box_history: deque[NDArray[np.float32]] = deque(maxlen=history_len)
        self._timestamps: deque[int] = deque(maxlen=history_len)
        self.saccade_count = 0
        self.gaze_shift_count = 0
        self.fixations = 0

    # ---------------------------------------------------------------- update
    def update(
        self,
        landmarks: Landmarks,
        ts_ms: int,
        *,
        has_refine: bool = True,
    ) -> float:
        if has_refine:
            left_iris = _iris_center(landmarks, LEFT_IRIS)
            right_iris = _iris_center(landmarks, RIGHT_IRIS)
            iris = (
                0.5 * (left_iris + right_iris)
                if left_iris is not None and right_iris is not None
                else (left_iris or right_iris)
            )
        else:
            # Fallback: use eye-centroid as a proxy for iris position
            left_eye = landmarks.points[LEFT_EYE, :2].mean(axis=0)
            right_eye = landmarks.points[RIGHT_EYE, :2].mean(axis=0)
            iris = 0.5 * (left_eye + right_eye)
        eye_box = _eye_box(landmarks, RIGHT_EYE)  # use one box to normalise

        if iris is None or eye_box is None:
            return 0.0

        self._iris_history.append(iris.astype(np.float32))
        self._eye_box_history.append(eye_box.astype(np.float32))
        self._timestamps.append(ts_ms)
        self._detect_events()
        return self.motion_score()

    # ---------------------------------------------------------------- logic
    def _detect_events(self) -> None:
        if len(self._iris_history) < 2:
            return
        prev = self._iris_history[-2]
        cur = self._iris_history[-1]
        prev_t = self._timestamps[-2]
        cur_t = self._timestamps[-1]
        dt = max(1, cur_t - prev_t) / 1000.0
        vel = float(np.linalg.norm(cur - prev) / dt)
        if vel >= self.saccade_vel:
            self.saccade_count += 1

        eb = self._eye_box_history[-1]
        bw, bh = max(1.0, eb[2] - eb[0]), max(1.0, eb[3] - eb[1])
        # gaze shift = iris position relative to eye-box centroid
        center = np.array([(eb[0] + eb[2]) / 2.0, (eb[1] + eb[3]) / 2.0], dtype=np.float32)
        rel = (cur - center) / np.array([bw, bh], dtype=np.float32)
        if float(np.linalg.norm(rel)) >= self.gaze_shift_norm:
            self.gaze_shift_count += 1

    # ---------------------------------------------------------------- query
    def motion_score(self) -> float:
        if len(self._iris_history) < 2:
            return 0.0
        # Variance of iris displacement over eye-box size.
        eb = self._eye_box_history[-1]
        bw, bh = max(1.0, eb[2] - eb[0]), max(1.0, eb[3] - eb[1])
        diffs = np.diff(np.stack(self._iris_history, axis=0), axis=0)
        norms = np.linalg.norm(diffs, axis=1) / np.sqrt(bw * bh)
        var = float(np.clip(np.var(norms) * 50.0, 0.0, 1.0))
        # Add a small bonus for saccades and gaze shifts
        bonus = min(0.4, 0.05 * (self.saccade_count + self.gaze_shift_count))
        return float(min(1.0, var + bonus))

    def drift_px(self) -> float:
        if len(self._iris_history) < 2:
            return 0.0
        first = self._iris_history[0]
        last = self._iris_history[-1]
        return float(np.linalg.norm(last - first))

    def result(self) -> EyeTrackingResult:
        return EyeTrackingResult(
            motion_score=self.motion_score(),
            drift_px=self.drift_px(),
            saccade_count=self.saccade_count,
            gaze_shift_count=self.gaze_shift_count,
            fixations=self.fixations,
        )

    def reset(self) -> None:
        self._iris_history.clear()
        self._eye_box_history.clear()
        self._timestamps.clear()
        self.saccade_count = 0
        self.gaze_shift_count = 0
        self.fixations = 0
