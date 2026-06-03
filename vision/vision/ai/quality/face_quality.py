"""Per-frame face quality scoring."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from vision.core.image_utils import brightness_mean, laplacian_sharpness


@dataclass(slots=True)
class QualityBreakdown:
    sharpness: float
    brightness: float
    size_score: float
    pose_score: float
    overall: float


def score_face(
    image: NDArray[np.uint8],
    bbox_area_px: int,
    image_area_px: int,
    *,
    yaw_deg: float = 0.0,
    pitch_deg: float = 0.0,
    roll_deg: float = 0.0,
    min_sharpness: float = 60.0,
    min_brightness: float = 40.0,
    max_brightness: float = 220.0,
    pose_tolerance: float = 25.0,
) -> QualityBreakdown:
    """0..1 quality score across sharpness, brightness, relative size, and pose."""
    sharp = laplacian_sharpness(image)
    bri = brightness_mean(image)
    sharp_s = min(1.0, sharp / max(min_sharpness, 1.0))
    if bri < min_brightness:
        bri_s = bri / min_brightness
    elif bri > max_brightness:
        bri_s = max(0.0, 1.0 - (bri - max_brightness) / (255.0 - max_brightness))
    else:
        bri_s = 1.0
    rel = (bbox_area_px / max(image_area_px, 1)) if image_area_px > 0 else 0.0
    size_s = float(np.clip(rel * 6.0, 0.0, 1.0))   # ~16% rel area ~= 1
    yaw_pen = max(0.0, 1.0 - abs(yaw_deg) / pose_tolerance)
    pitch_pen = max(0.0, 1.0 - abs(pitch_deg) / pose_tolerance)
    roll_pen = max(0.0, 1.0 - abs(roll_deg) / pose_tolerance)
    pose_s = float((yaw_pen + pitch_pen + roll_pen) / 3.0)
    overall = float(0.35 * sharp_s + 0.20 * bri_s + 0.20 * size_s + 0.25 * pose_s)
    return QualityBreakdown(
        sharpness=sharp_s,
        brightness=bri_s,
        size_score=size_s,
        pose_score=pose_s,
        overall=overall,
    )
