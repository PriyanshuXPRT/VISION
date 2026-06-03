"""Tests for the blink and eye-state heuristics (no real face required)."""
from __future__ import annotations

import numpy as np

from vision.ai.liveness.blink import BlinkDetector, eye_aspect_ratio


def _open_eye(width: float = 10.0, height: float = 5.0) -> np.ndarray:
    """6-point eye landmark layout: corners horizontal, top/bottom vertical."""
    return np.array(
        [
            [0.0, 0.0],          # p1: left corner
            [width * 0.3, -height],  # p2: top inner
            [width * 0.7, -height],  # p3: top outer
            [width, 0.0],        # p4: right corner
            [width * 0.7, height],   # p5: bottom outer
            [width * 0.3, height],   # p6: bottom inner
        ],
        dtype=np.float32,
    )


def test_ear_open_vs_closed():
    open_eye = _open_eye(width=10.0, height=5.0)
    closed_eye = _open_eye(width=10.0, height=0.1)
    ear_open = eye_aspect_ratio(open_eye)
    ear_closed = eye_aspect_ratio(closed_eye)
    assert ear_open > 0.3, f"open EAR too low: {ear_open}"
    assert ear_closed < 0.1, f"closed EAR too high: {ear_closed}"
    assert ear_open > ear_closed


def test_blink_detector_counts_blink():
    det = BlinkDetector(ear_thresh=0.20, consec_frames=1, min_blink_ms=50, max_blink_ms=800)
    # synthetic landmark container: not real, just shape-stable
    class LM:
        def __init__(self, height: float) -> None:
            self.points = np.zeros((468, 3), dtype=np.float32)
            eye = _open_eye(width=10.0, height=height)
            for i, idx in enumerate([33, 160, 158, 133, 153, 144]):
                self.points[idx] = [eye[i, 0], eye[i, 1], 0.0]
            # Mirror to the right eye so both sides contribute.
            for i, idx in enumerate([362, 385, 387, 263, 373, 380]):
                self.points[idx] = [eye[i, 0] + 30, eye[i, 1], 0.0]

    ts = 0
    for _ in range(30):          # 30 open frames
        det.update(LM(5.0), ts); ts += 20
    for _ in range(3):           # 3 closed frames (60 ms)
        det.update(LM(0.1), ts); ts += 20
    for _ in range(30):          # 30 open frames
        det.update(LM(5.0), ts); ts += 20
    r = det.result()
    assert r.blink_count >= 1, f"expected ≥ 1 blink, got {r.blink_count}"
    assert r.blink_score > 0.0
