"""Head pose estimation via MediaPipe + OpenCV solvePnP."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
from numpy.typing import NDArray

from vision.ai.landmarks import CHIN, FOREHEAD, LEFT_EAR, LEFT_EYE, NOSE_TIP, RIGHT_EAR, RIGHT_EYE
from vision.config import settings
from vision.core.math_utils import angle_diff_deg
from vision.core.types import Landmarks

# Approximate 3D model coordinates (in mm) of the same MediaPipe landmark indices.
_MODEL_3D = np.array(
    [
        [0.0, 0.0, 0.0],          # NOSE_TIP
        [0.0, -63.6, -12.5],      # CHIN
        [0.0, 63.6, -12.5],       # FOREHEAD (approx.)
        [-43.3, 32.7, -26.0],     # RIGHT_EAR
        [43.3, 32.7, -26.0],      # LEFT_EAR
        [-21.0, 0.0, -30.0],      # inner right eye (approx)
        [21.0, 0.0, -30.0],       # inner left eye (approx)
    ],
    dtype=np.float64,
)

_LM_INDICES = [NOSE_TIP, CHIN, FOREHEAD, RIGHT_EAR, LEFT_EAR, 133, 362]


@dataclass(slots=True)
class HeadPoseSample:
    yaw: float
    pitch: float
    roll: float
    timestamp_ms: int


@dataclass(slots=True)
class HeadPoseResult:
    yaw_range: float
    pitch_range: float
    roll_range: float
    mean_yaw: float
    mean_pitch: float
    head_pose_score: float   # 0..1
    samples: list[HeadPoseSample]


class HeadPoseEstimator:
    """solvePnP-based head pose with 6-DoF smoothing."""

    def __init__(
        self,
        *,
        yaw_range: float | None = None,
        pitch_range: float | None = None,
        smooth: int = 3,
    ) -> None:
        self.yaw_range = yaw_range or settings.head_yaw_range
        self.pitch_range = pitch_range or settings.head_pitch_range
        self.smooth = smooth
        self._samples: deque[HeadPoseSample] = deque(maxlen=128)
        self._yaw_buf: deque[float] = deque(maxlen=smooth)
        self._pitch_buf: deque[float] = deque(maxlen=smooth)
        self._roll_buf: deque[float] = deque(maxlen=smooth)

    # ---------------------------------------------------------------- estimate
    def estimate(
        self,
        landmarks: Landmarks,
        image_shape_hw: tuple[int, int],
        ts_ms: int,
    ) -> HeadPoseSample:
        h, w = image_shape_hw
        focal_length = float(w)
        center = (w / 2.0, h / 2.0)
        camera_matrix = np.array(
            [
                [focal_length, 0, center[0]],
                [0, focal_length, center[1]],
                [0, 0, 1],
            ],
            dtype=np.float64,
        )
        dist = np.zeros((4, 1), dtype=np.float64)

        pts2d = np.array(
            [landmarks.points[i, :2] for i in _LM_INDICES], dtype=np.float64
        )
        ok, rvec, tvec = cv2.solvePnP(
            _MODEL_3D, pts2d, camera_matrix, dist, flags=cv2.SOLVEPNP_ITERATIVE
        )
        if not ok:
            sample = HeadPoseSample(yaw=0.0, pitch=0.0, roll=0.0, timestamp_ms=ts_ms)
            self._samples.append(sample)
            return sample
        rmat, _ = cv2.Rodrigues(rvec)
        # Camera-to-world rotation: pose is R^T (model -> camera -> world)
        proj = rmat.T
        sy = np.sqrt(proj[0, 0] * proj[0, 0] + proj[1, 0] * proj[1, 0])
        singular = sy < 1e-6
        if not singular:
            pitch = float(np.degrees(np.arctan2(proj[2, 1], proj[2, 2])))
            yaw = float(np.degrees(np.arctan2(-proj[2, 0], sy)))
            roll = float(np.degrees(np.arctan2(proj[1, 0], proj[0, 0])))
        else:
            pitch = float(np.degrees(np.arctan2(-proj[1, 2], proj[1, 1])))
            yaw = float(np.degrees(np.arctan2(-proj[2, 0], sy)))
            roll = 0.0
        sample = HeadPoseSample(yaw=yaw, pitch=pitch, roll=roll, timestamp_ms=ts_ms)
        self._samples.append(sample)
        self._yaw_buf.append(yaw)
        self._pitch_buf.append(pitch)
        self._roll_buf.append(roll)
        return sample

    # ---------------------------------------------------------------- query
    def smoothed(self) -> tuple[float, float, float]:
        if not self._yaw_buf:
            return 0.0, 0.0, 0.0
        return (
            float(np.mean(self._yaw_buf)),
            float(np.mean(self._pitch_buf)),
            float(np.mean(self._roll_buf)),
        )

    def head_pose_score(self) -> float:
        """0..1 — natural head motion across the sample window.

        Rewards yaw/pitch excursions within tolerance; penalises zero motion.
        """
        if not self._samples:
            return 0.0
        yaws = np.array([s.yaw for s in self._samples])
        pitches = np.array([s.pitch for s in self._samples])
        rolls = np.array([s.roll for s in self._samples])
        yaw_span = float(yaws.max() - yaws.min())
        pitch_span = float(pitches.max() - pitches.min())
        roll_span = float(rolls.max() - rolls.min())
        # Reward at least 8° of yaw, 5° of pitch, 3° of roll
        y_score = min(1.0, yaw_span / 8.0)
        p_score = min(1.0, pitch_span / 5.0)
        r_score = min(1.0, roll_span / 3.0)
        return float(0.5 * y_score + 0.3 * p_score + 0.2 * r_score)

    def is_frontal(self, *, tol_yaw: float = 15.0, tol_pitch: float = 10.0) -> bool:
        y, p, _ = self.smoothed()
        return abs(y) <= tol_yaw and abs(p) <= tol_pitch

    def result(self) -> HeadPoseResult:
        if not self._samples:
            return HeadPoseResult(0, 0, 0, 0, 0, 0, [])
        yaws = [s.yaw for s in self._samples]
        pitches = [s.pitch for s in self._samples]
        rolls = [s.roll for s in self._samples]
        return HeadPoseResult(
            yaw_range=float(max(yaws) - min(yaws)),
            pitch_range=float(max(pitches) - min(pitches)),
            roll_range=float(max(rolls) - min(rolls)),
            mean_yaw=float(np.mean(yaws)),
            mean_pitch=float(np.mean(pitches)),
            head_pose_score=self.head_pose_score(),
            samples=list(self._samples),
        )

    def reset(self) -> None:
        self._samples.clear()
        self._yaw_buf.clear()
        self._pitch_buf.clear()
        self._roll_buf.clear()
