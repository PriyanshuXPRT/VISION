"""MediaPipe Face Mesh wrapper (468 landmarks)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from vision.config import settings
from vision.core.exceptions import InferenceError
from vision.core.logging import logger
from vision.core.types import BoundingBox, Landmarks

# Canonical MediaPipe Face Mesh indices.
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
LEFT_IRIS = [468, 469, 470, 471, 472]  # only when refine_landmarks=True
RIGHT_IRIS = [473, 474, 475, 476, 477]
NOSE_TIP = 1
CHIN = 152
FOREHEAD = 10
LEFT_EAR = 234
RIGHT_EAR = 454
MOUTH_LEFT = 61
MOUTH_RIGHT = 291


@dataclass(slots=True)
class LandmarkFrame:
    landmarks: Landmarks
    bbox: BoundingBox
    score: float = 0.0


class LandmarkEngine:
    """Lightweight wrapper around mediapipe.solutions.face_mesh.

    Pure Python — usable on desktop, server, and in unit tests.
    On-device Android uses the equivalent `MlKitFaceMeshProvider` in the
    Android client.
    """

    def __init__(
        self,
        *,
        max_num_faces: int | None = None,
        refine_landmarks: bool | None = None,
        model_complexity: int | None = None,
    ) -> None:
        self.max_num_faces = max_num_faces or settings.mp_max_faces
        self.refine = refine_landmarks if refine_landmarks is not None else settings.mp_refine_landmarks
        self.complexity = model_complexity or settings.mp_model_complexity
        try:
            import mediapipe as mp  # type: ignore
            self._mp = mp
            try:
                self._mesh = mp.solutions.face_mesh.FaceMesh(
                    static_image_mode=False,
                    max_num_faces=self.max_num_faces,
                    refine_landmarks=self.refine,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5,
                )
            except TypeError:
                # older API exposed model_complexity
                self._mesh = mp.solutions.face_mesh.FaceMesh(
                    static_image_mode=False,
                    max_num_faces=self.max_num_faces,
                    refine_landmarks=self.refine,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5,
                    model_complexity=self.complexity,
                )
            self._mode = "mediapipe"
        except ImportError:
            self._mp = None
            self._mesh = None
            self._mode = "fallback"
            logger.warning(
                "LandmarkEngine running in FALLBACK mode (mediapipe not installed). "
                "Blink/eye/head-pose liveness will be disabled; install "
                "'mediapipe>=0.10' for full functionality."
            )
        logger.info(
            "LandmarkEngine ready · mode={} max_faces={} refine={} complexity={}",
            self._mode, self.max_num_faces, self.refine, self.complexity,
        )

    # ---------------------------------------------------------------- process
    def extract_landmarks(
        self,
        image_rgb: NDArray[np.uint8],
    ) -> Optional[Landmarks]:
        """Return 468 (x, y, z) landmarks in pixel coords, or None if no face."""
        if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
            raise ValueError("extract_landmarks expects an RGB image (H, W, 3)")
        if self._mode == "fallback":
            return None
        try:
            res = self._mesh.process(image_rgb)
        except Exception as exc:  # noqa: BLE001
            raise InferenceError(f"MediaPipe FaceMesh failed: {exc}") from exc
        if not res.multi_face_landmarks:
            return None
        lms = res.multi_face_landmarks[0].landmark
        h, w = image_rgb.shape[:2]
        pts = np.zeros((len(lms), 3), dtype=np.float32)
        for i, lm in enumerate(lms):
            pts[i, 0] = lm.x * w
            pts[i, 1] = lm.y * h
            pts[i, 2] = lm.z * w
        return Landmarks(points=pts)

    def track_landmarks(
        self,
        image_rgb: NDArray[np.uint8],
        previous: Optional[Landmarks] = None,
    ) -> Optional[Landmarks]:
        """Same as extract_landmarks; tracking is handled by MediaPipe internally."""
        return self.extract_landmarks(image_rgb)

    # ---------------------------------------------------------------- geometry
    def estimate_face_geometry(
        self,
        landmarks: Landmarks,
        image_shape_hw: tuple[int, int],
    ) -> dict[str, NDArray[np.float32]]:
        h, w = image_shape_hw
        left_eye = np.array([landmarks.points[i, :2] for i in LEFT_EYE])
        right_eye = np.array([landmarks.points[i, :2] for i in RIGHT_EYE])
        return {
            "left_eye": left_eye,
            "right_eye": right_eye,
            "left_iris": np.array([landmarks.points[i, :2] for i in LEFT_IRIS])
            if self.refine else np.zeros((0, 2), dtype=np.float32),
            "right_iris": np.array([landmarks.points[i, :2] for i in RIGHT_IRIS])
            if self.refine else np.zeros((0, 2), dtype=np.float32),
            "nose_tip": landmarks.points[NOSE_TIP, :2],
            "chin": landmarks.points[CHIN, :2],
            "forehead": landmarks.points[FOREHEAD, :2],
            "image_size": np.array([w, h], dtype=np.float32),
        }

    # ---------------------------------------------------------------- bbox
    @staticmethod
    def bbox_from_landmarks(landmarks: Landmarks) -> BoundingBox:
        pts = landmarks.points[:, :2]
        x1, y1 = pts.min(axis=0)
        x2, y2 = pts.max(axis=0)
        return BoundingBox(float(x1), float(y1), float(x2), float(y2))

    def close(self) -> None:
        try:
            self._mesh.close()
        except Exception:  # noqa: BLE001
            pass

    def __del__(self) -> None:  # pragma: no cover
        self.close()
