"""Math utilities: similarity, angle normalization, linalg helpers."""
from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray


# -----------------------------------------------------------------------------
# Embedding math
# -----------------------------------------------------------------------------
def l2_normalize(x: NDArray[np.float32], eps: float = 1e-12) -> NDArray[np.float32]:
    norm = float(np.linalg.norm(x))
    if norm < eps:
        return x
    return (x / norm).astype(np.float32)


def cosine_similarity(a: NDArray[np.float32], b: NDArray[np.float32]) -> float:
    """Cosine similarity for two 1-D vectors; both assumed non-zero."""
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def cosine_similarity_matrix(
    query: NDArray[np.float32],
    bank: NDArray[np.float32],
) -> NDArray[np.float32]:
    """Compute cosine similarity between one query and N bank embeddings.

    `bank` shape: (N, D). Returns shape (N,).
    """
    q = query / (np.linalg.norm(query) + 1e-12)
    b = bank / (np.linalg.norm(bank, axis=1, keepdims=True) + 1e-12)
    return (b @ q).astype(np.float32)


# -----------------------------------------------------------------------------
# Geometry
# -----------------------------------------------------------------------------
def rotation_matrix_to_euler_xyz(r: NDArray[np.float64]) -> tuple[float, float, float]:
    """Convert 3x3 rotation matrix to (pitch, yaw, roll) in degrees."""
    sy = math.sqrt(r[0, 0] * r[0, 0] + r[1, 0] * r[1, 0])
    singular = sy < 1e-6
    if not singular:
        pitch = math.degrees(math.atan2(r[2, 1], r[2, 2]))
        yaw = math.degrees(math.atan2(-r[2, 0], sy))
        roll = math.degrees(math.atan2(r[1, 0], r[0, 0]))
    else:
        pitch = math.degrees(math.atan2(-r[1, 2], r[1, 1]))
        yaw = math.degrees(math.atan2(-r[2, 0], sy))
        roll = 0.0
    return pitch, yaw, roll


def angle_diff_deg(a: float, b: float) -> float:
    """Smallest signed angular difference in degrees in [-180, 180]."""
    d = (a - b + 180.0) % 360.0 - 180.0
    return d
