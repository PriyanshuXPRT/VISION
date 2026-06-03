"""Domain-level data structures shared across the AI pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np
from numpy.typing import NDArray


# -----------------------------------------------------------------------------
# Enums
# -----------------------------------------------------------------------------
class Decision(str, Enum):
    """Final authentication decision."""

    ACCEPT = "accept"
    REJECT = "reject"
    INCONCLUSIVE = "inconclusive"


class SpoofKind(str, Enum):
    """Categories of presentation attack we attempt to detect."""

    NONE = "none"
    PRINT = "print"
    MOBILE_SCREEN = "mobile_screen"
    LAPTOP_SCREEN = "laptop_screen"
    REPLAY_VIDEO = "replay_video"
    FACE_SWAP = "face_swap"
    DEEPFAKE = "deepfake"
    MASK = "mask"
    UNKNOWN = "unknown"


class LivenessSignal(str, Enum):
    """Individual liveness sub-signal."""

    PASSIVE_FAS = "passive_fas"
    BLINK = "blink"
    EYE_MOTION = "eye_motion"
    HEAD_POSE = "head_pose"
    TEMPORAL = "temporal"


# -----------------------------------------------------------------------------
# Geometry containers
# -----------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class BoundingBox:
    """Axis-aligned bounding box in (x1, y1, x2, y2) pixel coords."""

    x1: float
    y1: float
    x2: float
    y2: float

    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    def area(self) -> float:
        return self.width() * self.height()

    def as_int(self) -> tuple[int, int, int, int]:
        return int(self.x1), int(self.y1), int(self.x2), int(self.y2)

    def as_xywh(self) -> tuple[float, float, float, float]:
        return self.x1, self.y1, self.width(), self.height()


@dataclass(frozen=True, slots=True)
class Landmarks:
    """MediaPipe Face Mesh landmarks, shape (N, 3) where N is 468 (basic) or
    478 (with iris refinement: 468 + 10 iris landmarks)."""

    points: NDArray[np.float32]  # shape (468, 3) or (478, 3)

    def __post_init__(self) -> None:
        if self.points.ndim != 2 or self.points.shape[1] != 3:
            raise ValueError(f"Expected (N, 3) landmarks, got {self.points.shape}")
        if self.points.shape[0] not in (468, 478):
            raise ValueError(
                f"Expected 468 or 478 landmarks, got {self.points.shape[0]}"
            )

    @property
    def has_iris(self) -> bool:
        """True when iris landmarks are present (478-point mode)."""
        return self.points.shape[0] == 478


@dataclass(frozen=True, slots=True)
class AlignedFace:
    """A face cropped, aligned, and resized to recognition input."""

    image: NDArray[np.uint8]            # (H, W, 3) RGB
    bbox: BoundingBox
    landmarks: Landmarks
    quality_score: float                # 0..1


# -----------------------------------------------------------------------------
# Per-frame analysis output
# -----------------------------------------------------------------------------
@dataclass(slots=True)
class FrameAnalysis:
    """Result of running one frame through the AI pipeline."""

    timestamp_ms: int
    bbox: BoundingBox
    landmarks: Optional[Landmarks] = None
    aligned_face: Optional[NDArray[np.uint8]] = None    # (112, 112, 3)
    passive_liveness: float = 0.0
    is_spoof: bool = False
    spoof_kind: SpoofKind = SpoofKind.NONE
    ear: float = 0.0
    eye_motion: float = 0.0
    yaw: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0
    quality_score: float = 0.0
    embedding: Optional[NDArray[np.float32]] = None     # (512,)


# -----------------------------------------------------------------------------
# Liveness rollup
# -----------------------------------------------------------------------------
@dataclass(slots=True)
class LivenessReport:
    blink_count: int = 0
    blink_score: float = 0.0
    eye_motion_score: float = 0.0
    head_pose_score: float = 0.0
    passive_score: float = 0.0
    temporal_score: float = 0.0
    final_score: float = 0.0
    spoof_kind: SpoofKind = SpoofKind.NONE
    accepted_signals: list[LivenessSignal] = field(default_factory=list)
    rejected_signals: list[LivenessSignal] = field(default_factory=list)


# -----------------------------------------------------------------------------
# Identification
# -----------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class UserMatch:
    user_id: int
    name: str
    similarity: float
    template_id: int


@dataclass(slots=True)
class IdentificationResult:
    best_match: Optional[UserMatch] = None
    candidates: list[UserMatch] = field(default_factory=list)
    is_identified: bool = False


# -----------------------------------------------------------------------------
# Top-level decision
# -----------------------------------------------------------------------------
@dataclass(slots=True)
class AuthenticationResult:
    decision: Decision
    user_id: Optional[int] = None
    user_name: Optional[str] = None
    similarity: float = 0.0
    liveness: LivenessReport = field(default_factory=LivenessReport)
    frames_processed: int = 0
    latency_ms: int = 0
    reason: str = ""
