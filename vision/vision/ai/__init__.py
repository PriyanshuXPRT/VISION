"""AI subsystem top-level package."""
from vision.ai.antispoof import AntiSpoof, LivenessPrediction
from vision.ai.detection import Detection, FaceDetector
from vision.ai.embedding_search import (
    BruteForceIndex,
    FaissIndex,
    SearchHit,
    VectorIndex,
    build_index,
)
from vision.ai.landmarks import LandmarkEngine
from vision.ai.liveness.blink import BlinkDetector
from vision.ai.liveness.eye import EyeTracker
from vision.ai.liveness.headpose import HeadPoseEstimator
from vision.ai.liveness.temporal import TemporalLivenessEngine
from vision.ai.recognition import FaceRecognizer, FaceTemplate

__all__ = [
    "FaceDetector",
    "Detection",
    "FaceRecognizer",
    "FaceTemplate",
    "AntiSpoof",
    "LivenessPrediction",
    "LandmarkEngine",
    "BlinkDetector",
    "EyeTracker",
    "HeadPoseEstimator",
    "TemporalLivenessEngine",
    "VectorIndex",
    "BruteForceIndex",
    "FaissIndex",
    "SearchHit",
    "build_index",
]
