"""Face detection, alignment, and quality assessment (SCRFD-10GF)."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
from numpy.typing import NDArray

from vision.ai.alignment import estimate_similarity_transform, warp_face
from vision.config import settings
from vision.core.exceptions import FaceQualityError, InferenceError, NoFaceDetectedError
from vision.core.image_utils import (
    bgr_to_rgb,
    clip_bbox,
    estimate_face_quality,
    expand_bbox,
    hwc_to_chw,
    normalize,
)
from vision.core.logging import logger
from vision.core.onnx_runner import OnnxRunner
from vision.core.types import AlignedFace, BoundingBox, Landmarks


# -----------------------------------------------------------------------------
# Detection result container
# -----------------------------------------------------------------------------
@dataclass(slots=True)
class Detection:
    bbox: BoundingBox
    score: float
    landmarks: NDArray[np.float32]   # (5, 2) in image pixel coords


# -----------------------------------------------------------------------------
# Detector
# -----------------------------------------------------------------------------
class FaceDetector:
    """SCRFD-10GF face detector (5-landmark output, ONNX)."""

    def __init__(
        self,
        model_path: Optional[str | None] = None,
        *,
        input_size: int | None = None,
        conf_thresh: float | None = None,
        nms_thresh: float | None = None,
    ) -> None:
        self.input_size = input_size or settings.scrfd_input_size
        self.conf_thresh = conf_thresh or settings.scrfd_conf_thresh
        self.nms_thresh = nms_thresh or settings.scrfd_nms_thresh
        self.model_path = str(model_path or settings.scrfd_path)
        self.runner = OnnxRunner(self.model_path)
        self._feat_stride_fpn: list[int] = [8, 16, 32]
        self.runner.warmup()
        logger.info(
            "FaceDetector ready · {} · input={} conf={} nms={}",
            self.runner.model_path.name,
            self.input_size,
            self.conf_thresh,
            self.nms_thresh,
        )

    # ---------------------------------------------------------------- detect
    def detect(self, image: NDArray[np.uint8], max_num: int = 5) -> list[Detection]:
        """Run detection. `image` is RGB uint8 (H, W, 3). Returns sorted by score."""
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("detect() expects an RGB image (H, W, 3)")

        h0, w0 = image.shape[:2]
        img, scale = self._preprocess(image)
        outputs = self._infer(img)
        dets, kpss = self._postprocess(outputs, scale, (h0, w0))
        if not dets:
            return []
        keep = self._nms(dets, self.nms_thresh)[:max_num]
        out: list[Detection] = []
        for i in keep:
            x1, y1, x2, y2, score = dets[i]
            kp = kpss[i] if kpss is not None else np.zeros((5, 2), dtype=np.float32)
            out.append(
                Detection(
                    bbox=BoundingBox(float(x1), float(y1), float(x2), float(y2)),
                    score=float(score),
                    landmarks=kp.astype(np.float32),
                )
            )
        return out

    def detect_single(self, image: NDArray[np.uint8]) -> Detection:
        dets = self.detect(image, max_num=1)
        if not dets:
            raise NoFaceDetectedError("No face detected")
        return dets[0]

    # ---------------------------------------------------------------- crop
    def crop_face(
        self,
        image: NDArray[np.uint8],
        detection: Detection,
        margin: float = 0.15,
    ) -> NDArray[np.uint8]:
        x1, y1, x2, y2 = detection.bbox.x1, detection.bbox.y1, detection.bbox.x2, detection.bbox.y2
        h, w = image.shape[:2]
        x1, y1, x2, y2 = expand_bbox((x1, y1, x2, y2), w, h, margin=margin)
        x1, y1, x2, y2 = clip_bbox((x1, y1, x2, y2), w, h)
        return image[y1:y2, x1:x2].copy()

    # ---------------------------------------------------------------- align
    def align_face(
        self,
        image: NDArray[np.uint8],
        detection: Detection,
        output_size: int = 112,
    ) -> AlignedFace:
        aligned = warp_face(image, detection.landmarks, output_size=output_size)
        crop = self.crop_face(image, detection)
        quality = estimate_face_quality(crop)
        pts = np.concatenate(
            [
                detection.landmarks,
                np.zeros((detection.landmarks.shape[0], 1), dtype=np.float32),
            ],
            axis=1,
        ).astype(np.float32)
        return AlignedFace(
            image=aligned,
            bbox=detection.bbox,
            landmarks=Landmarks(points=pts),
            quality_score=quality,
        )

    # ---------------------------------------------------------------- quality
    def quality_check(
        self,
        image: NDArray[np.uint8],
        detection: Detection,
        *,
        min_quality: float = 0.25,
        min_face_px: int = 80,
    ) -> float:
        h, w = image.shape[:2]
        face_w = detection.bbox.width()
        face_h = detection.bbox.height()
        if face_w < min_face_px or face_h < min_face_px:
            raise FaceQualityError(f"Face too small: {face_w:.0f}x{face_h:.0f}px")
        crop = self.crop_face(image, detection, margin=0.10)
        q = estimate_face_quality(crop)
        if q < min_quality:
            raise FaceQualityError(f"Face quality too low: {q:.2f}")
        return q

    # ---------------------------------------------------------------- helpers
    @staticmethod
    def _preprocess(image: NDArray[np.uint8]) -> tuple[NDArray[np.float32], float]:
        h, w = image.shape[:2]
        det = max(w, h)
        scale = 320.0 / det
        nw, nh = int(round(w * scale)), int(round(h * scale))
        resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
        pad_w, pad_h = 320 - nw, 320 - nh
        padded = cv2.copyMakeBorder(
            resized, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=(0, 0, 0)
        )
        x = normalize(padded).astype(np.float32)
        x = hwc_to_chw(x)
        x = x[None, ...]
        return x, scale

    def _infer(self, x: NDArray[np.float32]) -> list[NDArray[np.float32]]:
        try:
            return self.runner.run({self.runner.input_names[0]: x})
        except Exception as exc:  # noqa: BLE001
            raise InferenceError(f"SCRFD inference failed: {exc}") from exc

    def _postprocess(
        self,
        outputs: list[NDArray[np.float32]],
        scale: float,
        img_shape: tuple[int, int],
    ) -> tuple[list[list[float]], Optional[NDArray[np.float32]]]:
        h0, w0 = img_shape
        fmc = len(self._feat_stride_fpn)
        bboxes_list: list[list[float]] = []
        kpss_list: list[NDArray[np.float32]] = []
        for idx, stride in enumerate(self._feat_stride_fpn):
            if idx >= fmc:
                break
            score = outputs[idx]
            bbox = outputs[idx + fmc]
            kps = outputs[idx + fmc * 2] if (idx + fmc * 2) < len(outputs) else None
            anchor_centers = self._anchor_centers(score, stride)
            # Reduce leading dims so we always work on (C, H, W) where C is
            # the per-cell output channel count (1 for score, 4 for bbox,
            # 10 for kps).  This tolerates stubs and real weights alike.
            while score.ndim > 3:
                score = score[0]
            while bbox.ndim > 3:
                bbox = bbox[0]
            if kps is not None:
                while kps.ndim > 3:
                    kps = kps[0]
            if score.ndim == 3:
                score = score[0]
            if bbox.ndim == 3:
                bbox = bbox[0]
            if kps is not None and kps.ndim == 3:
                kps = kps[0]
            inds = np.where(score >= self.conf_thresh)[0]
            if inds.size == 0:
                continue
            for i in inds:
                ax, ay = anchor_centers[i]
                sc = float(score[i])
                dx, dy, dw, dh = bbox[:, i]
                x1 = (ax - dw * 0.5) / scale
                y1 = (ay - dh * 0.5) / scale
                x2 = (ax + dw * 0.5) / scale
                y2 = (ay + dh * 0.5) / scale
                bboxes_list.append([x1, y1, x2, y2, sc])
                if kps is not None:
                    pts = kps[:, i].reshape(5, 2) / scale
                    pts[:, 0] += ax / scale - dw * 0.5 / scale
                    pts[:, 1] += ay / scale - dh * 0.5 / scale
                    kpss_list.append(pts)
        if not bboxes_list:
            return [], None
        kpss_arr = np.array(kpss_list, dtype=np.float32) if kpss_list else None
        return bboxes_list, kpss_arr

    def _anchor_centers(self, score: NDArray[np.float32], stride: int) -> NDArray[np.float32]:
        # Real SCRFD output is (B, num_anchors, H, W); squeeze singleton dims
        # so the anchor grid lines up with (H, W).
        s = score
        while s.ndim > 2:
            s = s[0]
        h, w = s.shape
        xs = (np.arange(w) + 0.5) * stride
        ys = (np.arange(h) + 0.5) * stride
        xv, yv = np.meshgrid(xs, ys)
        return np.stack([xv.flatten(), yv.flatten()], axis=1).astype(np.float32)

    @staticmethod
    def _nms(dets: list[list[float]], iou_thresh: float) -> list[int]:
        if not dets:
            return []
        arr = np.array(dets, dtype=np.float32)
        x1, y1, x2, y2, s = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3], arr[:, 4]
        areas = (x2 - x1) * (y2 - y1)
        order = s.argsort()[::-1]
        keep: list[int] = []
        while order.size > 0:
            i = int(order[0])
            keep.append(i)
            if order.size == 1:
                break
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
            iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
            inds = np.where(iou <= iou_thresh)[0]
            order = order[inds + 1]
        return keep
