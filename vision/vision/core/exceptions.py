"""V.I.S.I.O.N. exception hierarchy."""
from __future__ import annotations


class VisionError(Exception):
    """Base exception for all V.I.S.I.O.N. errors."""

    code: str = "VISION_ERROR"

    def __init__(self, message: str = "", *, code: str | None = None) -> None:
        super().__init__(message)
        if code:
            self.code = code


# ---- Configuration --------------------------------------------------------
class ConfigError(VisionError):
    code = "VISION_CONFIG_ERROR"


# ---- Model loading / inference -------------------------------------------
class ModelNotFoundError(VisionError):
    code = "VISION_MODEL_NOT_FOUND"


class ModelLoadError(VisionError):
    code = "VISION_MODEL_LOAD_ERROR"


class InferenceError(VisionError):
    code = "VISION_INFERENCE_ERROR"


# ---- Pipeline errors ------------------------------------------------------
class NoFaceDetectedError(VisionError):
    code = "VISION_NO_FACE"


class MultipleFacesError(VisionError):
    code = "VISION_MULTIPLE_FACES"


class FaceQualityError(VisionError):
    code = "VISION_FACE_QUALITY"


class LivenessError(VisionError):
    code = "VISION_LIVENESS"


class SpoofDetectedError(VisionError):
    code = "VISION_SPOOF"


# ---- Database -------------------------------------------------------------
class DatabaseError(VisionError):
    code = "VISION_DB_ERROR"


class UserNotFoundError(VisionError):
    code = "VISION_USER_NOT_FOUND"


class DuplicateUserError(VisionError):
    code = "VISION_DUPLICATE_USER"


# ---- I/O ------------------------------------------------------------------
class CameraError(VisionError):
    code = "VISION_CAMERA_ERROR"


class AssetError(VisionError):
    code = "VISION_ASSET_ERROR"
