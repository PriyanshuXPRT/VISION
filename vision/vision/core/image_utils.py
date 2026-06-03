"""Image / array utilities used across the AI pipeline."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

from vision.core.exceptions import AssetError


# -----------------------------------------------------------------------------
# I/O
# -----------------------------------------------------------------------------
def load_image(path: str | Path, *, color_format: str = "rgb") -> NDArray[np.uint8]:
    """Read an image as RGB uint8 ndarray."""
    path = Path(path)
    if not path.is_file():
        raise AssetError(f"Image not found: {path}")
    raw = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if raw is None:
        raise AssetError(f"cv2 could not decode image: {path}")
    if color_format.lower() == "rgb":
        return cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
    if color_format.lower() == "bgr":
        return raw
    raise ValueError(f"Unknown color format '{color_format}'")


def bgr_to_rgb(img: NDArray[np.uint8]) -> NDArray[np.uint8]:
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def rgb_to_bgr(img: NDArray[np.uint8]) -> NDArray[np.uint8]:
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


# -----------------------------------------------------------------------------
# Geometry helpers
# -----------------------------------------------------------------------------
def clip_bbox(
    bbox_xyxy: tuple[float, float, float, float],
    w: int,
    h: int,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox_xyxy
    x1 = int(max(0, min(w - 1, round(x1))))
    y1 = int(max(0, min(h - 1, round(y1))))
    x2 = int(max(0, min(w, round(x2))))
    y2 = int(max(0, min(h, round(y2))))
    if x2 <= x1:
        x2 = x1 + 1
    if y2 <= y1:
        y2 = y1 + 1
    return x1, y1, x2, y2


def expand_bbox(
    bbox_xyxy: tuple[float, float, float, float],
    w: int,
    h: int,
    margin: float = 0.15,
) -> tuple[float, float, float, float]:
    """Symmetrically enlarge a bbox by `margin * side`."""
    x1, y1, x2, y2 = bbox_xyxy
    bw, bh = x2 - x1, y2 - y1
    x1 -= bw * margin
    y1 -= bh * margin
    x2 += bw * margin
    y2 += bh * margin
    return (
        float(max(0, x1)),
        float(max(0, y1)),
        float(min(w - 1, x2)),
        float(min(h - 1, y2)),
    )


# -----------------------------------------------------------------------------
# Preprocessing
# -----------------------------------------------------------------------------
def resize_keep_aspect(image: NDArray[np.uint8], size: tuple[int, int]) -> NDArray[np.uint8]:
    h, w = image.shape[:2]
    tw, th = size
    scale = min(tw / w, th / h)
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    return cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)


def normalize(
    image: NDArray[np.uint8],
    mean: tuple[float, float, float] = (127.5, 127.5, 127.5),
    std: tuple[float, float, float] = (128.0, 128.0, 128.0),
) -> NDArray[np.float32]:
    img = image.astype(np.float32)
    img = (img - np.array(mean, dtype=np.float32)) / np.array(std, dtype=np.float32)
    return img


def hwc_to_chw(image: NDArray[np.float32]) -> NDArray[np.float32]:
    return np.transpose(image, (2, 0, 1)).copy()


def add_batch_dim(x: NDArray[np.float32]) -> NDArray[np.float32]:
    return x[None, ...]


# -----------------------------------------------------------------------------
# Quality
# -----------------------------------------------------------------------------
def laplacian_sharpness(image: NDArray[np.uint8]) -> float:
    """Variance of Laplacian — higher is sharper. Useful quality signal."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def brightness_mean(image: NDArray[np.uint8]) -> float:
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
    return float(gray.mean())


def estimate_face_quality(
    image: NDArray[np.uint8],
    *,
    min_sharpness: float = 50.0,
    min_brightness: float = 40.0,
    max_brightness: float = 220.0,
) -> float:
    """Return a 0..1 quality score combining sharpness, brightness, size heuristics."""
    sharp = laplacian_sharpness(image)
    bri = brightness_mean(image)
    s_score = min(1.0, sharp / 200.0)
    b_score = 1.0
    if bri < min_brightness:
        b_score = bri / min_brightness
    elif bri > max_brightness:
        b_score = max(0.0, 1.0 - (bri - max_brightness) / (255.0 - max_brightness))
    return float(0.7 * s_score + 0.3 * b_score)


# -----------------------------------------------------------------------------
# Decoding
# -----------------------------------------------------------------------------
def decode_jpeg_bytes(data: bytes) -> NDArray[np.uint8]:
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise AssetError("Could not decode JPEG bytes")
    return bgr_to_rgb(img)
