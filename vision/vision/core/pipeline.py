"""
Pipeline factory: builds the full set of engines from a single Database handle.

This is the single point of integration for both the CLI demo and the
hybrid backend.
"""
from __future__ import annotations

from dataclasses import dataclass

from vision.ai import (
    AntiSpoof,
    FaceDetector,
    FaceRecognizer,
    LandmarkEngine,
)
from vision.ai.liveness.temporal import TemporalLivenessEngine
from vision.authentication import Authenticator
from vision.config import settings
from vision.core.logging import logger
from vision.database import Database
from vision.identification import IdentificationService
from vision.registration import RegistrationService


@dataclass(slots=True)
class VisionPipeline:
    db: Database
    detector: FaceDetector
    recognizer: FaceRecognizer
    antispoof: AntiSpoof
    landmarks: LandmarkEngine
    temporal: TemporalLivenessEngine
    identification: IdentificationService
    registration: RegistrationService
    authenticator: Authenticator

    def rebuild_index(self) -> None:
        self.identification.rebuild()


def build_pipeline(
    db: Database | None = None,
    *,
    prefer_index: str = "faiss",
) -> VisionPipeline:
    """Initialise all engines, repositories, and services."""
    from vision.database import Database as _DB
    db = db or _DB()

    logger.info("Booting V.I.S.I.O.N. pipeline …")
    detector = FaceDetector()
    recognizer = FaceRecognizer()
    antispoof = AntiSpoof()
    landmarks = LandmarkEngine()
    temporal = TemporalLivenessEngine()
    identification = IdentificationService(db, recognizer, prefer=prefer_index)
    registration = RegistrationService(db, detector, recognizer, antispoof, landmarks)
    authenticator = Authenticator(db, detector, recognizer, antispoof, landmarks, identification, temporal)
    logger.info("Pipeline ready.")
    return VisionPipeline(
        db=db,
        detector=detector,
        recognizer=recognizer,
        antispoof=antispoof,
        landmarks=landmarks,
        temporal=temporal,
        identification=identification,
        registration=registration,
        authenticator=authenticator,
    )
