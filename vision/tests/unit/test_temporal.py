"""Tests for the temporal engine."""
from __future__ import annotations

import numpy as np

from vision.ai.liveness.temporal import HeuristicEngine, build_feature_vector


def test_build_feature_shape():
    f = build_feature_vector(
        passive_liveness=0.9, ear=0.3, eye_motion=0.4, yaw=10, pitch=5, roll=2,
        head_motion_rms=0.2, blink_rate_per_sec=0.5,
    )
    assert f.shape == (8,)
    assert np.isfinite(f).all()


def test_heuristic_engine_buffering():
    eng = HeuristicEngine(seq_len=8, stride=1, liveness_threshold=0.5)
    assert not eng.is_ready()
    rng = np.random.default_rng(0)
    for i in range(8):
        eng.push(rng.uniform(0.3, 1.0, size=8).astype(np.float32))
    assert eng.is_ready()
    r = eng.score()
    assert 0.0 <= r.final_score <= 1.0
