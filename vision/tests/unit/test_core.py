"""Smoke tests for the core types and math utilities."""
from __future__ import annotations

import numpy as np
import pytest

from vision.core.math_utils import (
    angle_diff_deg,
    cosine_similarity,
    cosine_similarity_matrix,
    l2_normalize,
    rotation_matrix_to_euler_xyz,
)
from vision.core.types import (
    AuthenticationResult,
    BoundingBox,
    Decision,
    IdentificationResult,
    LivenessReport,
    SpoofKind,
)


def test_l2_normalize_zero_vector():
    z = np.zeros(8, dtype=np.float32)
    out = l2_normalize(z)
    assert np.allclose(out, z)


def test_cosine_similarity_orthogonal_zero():
    a = np.array([1.0, 0.0], dtype=np.float32)
    b = np.array([0.0, 1.0], dtype=np.float32)
    assert cosine_similarity(a, b) == pytest.approx(0.0, abs=1e-6)


def test_cosine_similarity_identical_one():
    a = np.random.default_rng(0).standard_normal(32).astype(np.float32)
    a = l2_normalize(a)
    assert cosine_similarity(a, a) == pytest.approx(1.0, abs=1e-5)


def test_cosine_similarity_matrix_shape():
    q = l2_normalize(np.array([1.0, 0.0, 0.0], dtype=np.float32))
    bank = np.array([[1, 0, 0], [0, 1, 0], [-1, 0, 0]], dtype=np.float32)
    bank = bank / np.linalg.norm(bank, axis=1, keepdims=True)
    sims = cosine_similarity_matrix(q, bank)
    assert sims.shape == (3,)
    assert sims[0] == pytest.approx(1.0, abs=1e-5)
    assert sims[2] == pytest.approx(-1.0, abs=1e-5)


def test_angle_diff_deg_wrap():
    assert angle_diff_deg(170, -170) == pytest.approx(-20.0, abs=1e-6)


def test_rotation_matrix_identity_zero_angles():
    e = rotation_matrix_to_euler_xyz(np.eye(3))
    assert all(abs(x) < 1e-6 for x in e)


def test_dataclass_defaults():
    r = AuthenticationResult(decision=Decision.ACCEPT, user_id=1, user_name="x")
    assert r.similarity == 0.0
    assert r.liveness.final_score == 0.0
    assert r.reason == ""
    assert isinstance(r.liveness, LivenessReport)


def test_bbox_area_zero_for_invalid():
    b = BoundingBox(10, 10, 10, 10)
    assert b.area() == 0.0
    assert b.width() == 0.0


def test_identification_result_empty():
    res = IdentificationResult()
    assert res.best_match is None
    assert res.candidates == []
    assert not res.is_identified


def test_spoof_kind_values():
    assert SpoofKind.NONE.value == "none"
    assert SpoofKind.PRINT.value == "print"
