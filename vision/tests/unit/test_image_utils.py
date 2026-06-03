"""Unit tests for image utilities."""
from __future__ import annotations

import numpy as np
import pytest

from vision.core.image_utils import (
    bgr_to_rgb,
    brightness_mean,
    clip_bbox,
    estimate_face_quality,
    expand_bbox,
    hwc_to_chw,
    laplacian_sharpness,
    normalize,
    resize_keep_aspect,
)


def test_bgr_rgb_inverse():
    img = np.random.default_rng(0).integers(0, 255, (10, 10, 3), dtype=np.uint8)
    assert np.array_equal(bgr_to_rgb(bgr_to_rgb(img)), img)


def test_clip_bbox_no_overflow():
    x1, y1, x2, y2 = clip_bbox((-5, -5, 9999, 9999), 640, 480)
    assert 0 <= x1 < x2 <= 640
    assert 0 <= y1 < y2 <= 480


def test_expand_bbox_margin():
    x1, y1, x2, y2 = expand_bbox((100, 100, 200, 200), 640, 480, margin=0.1)
    assert x1 < 100 and y1 < 100 and x2 > 200 and y2 > 200


def test_laplacian_sharpness_blur_vs_sharp():
    rng = np.random.default_rng(0)
    sharp = rng.integers(0, 255, (64, 64), dtype=np.uint8)
    blur = np.stack([np.full((64, 64), 128, dtype=np.uint8)] * 1, axis=0).squeeze()
    assert laplacian_sharpness(blur) < laplacian_sharpness(sharp)


def test_brightness_mean():
    img = np.full((8, 8, 3), 200, dtype=np.uint8)
    assert brightness_mean(img) == pytest.approx(200.0, abs=1e-3)


def test_quality_blurry_image_low():
    img = np.full((96, 96, 3), 128, dtype=np.uint8)
    q = estimate_face_quality(img)
    assert q < 0.5


def test_normalize_shape_and_range():
    img = np.full((8, 8, 3), 127, dtype=np.uint8)
    out = normalize(img, mean=(127.5, 127.5, 127.5), std=(128.0, 128.0, 128.0))
    assert out.shape == (8, 8, 3)
    assert out.dtype == np.float32


def test_hwc_to_chw():
    img = np.zeros((4, 4, 3), dtype=np.float32)
    out = hwc_to_chw(img)
    assert out.shape == (3, 4, 4)


def test_resize_keep_aspect():
    img = np.zeros((100, 200, 3), dtype=np.uint8)
    out = resize_keep_aspect(img, (50, 50))
    assert max(out.shape[:2]) == 50
