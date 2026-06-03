"""Face alignment: 5-landmark similarity transform (ArcFace template)."""
from __future__ import annotations

import math

import cv2
import numpy as np
from numpy.typing import NDArray

_ARCFACE_REF_112 = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)


def estimate_similarity_transform(
    src: NDArray[np.float32],
    dst: NDArray[np.float32],
) -> NDArray[np.float32]:
    """Estimate 2x3 similarity matrix (rotation + scale + translation) via least-squares."""
    if src.shape != dst.shape or src.shape[1] != 2:
        raise ValueError(f"src/dst must be (N, 2), got {src.shape}/{dst.shape}")
    src_c = src - src.mean(axis=0, keepdims=True)
    dst_c = dst - dst.mean(axis=0, keepdims=True)
    src_scale = math.sqrt((src_c ** 2).sum() / src.shape[0])
    dst_scale = math.sqrt((dst_c ** 2).sum() / dst.shape[0])
    src_n = src_c / (src_scale + 1e-9)
    dst_n = dst_c / (dst_scale + 1e-9)
    H = src_n.T @ dst_n
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1
        R = Vt.T @ U.T
    s = dst_scale / (src_scale + 1e-9)
    t = dst.mean(axis=0) - s * R @ src.mean(axis=0)
    M = np.zeros((2, 3), dtype=np.float32)
    M[0, 0] = R[0, 0] * s
    M[0, 1] = R[0, 1] * s
    M[1, 0] = R[1, 0] * s
    M[1, 1] = R[1, 1] * s
    M[0, 2] = t[0]
    M[1, 2] = t[1]
    return M


def warp_face(
    image: NDArray[np.uint8],
    landmarks_5: NDArray[np.float32],
    output_size: int = 112,
) -> NDArray[np.uint8]:
    """Align a face to the ArcFace reference template of size (output_size, output_size)."""
    ref = _ARCFACE_REF_112 * (output_size / 112.0)
    M = estimate_similarity_transform(landmarks_5.astype(np.float32), ref)
    return cv2.warpAffine(
        image,
        M,
        (output_size, output_size),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
