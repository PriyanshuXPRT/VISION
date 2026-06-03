"""
User registration pipeline.

Captures a short video, extracts frames, runs SCRFD + alignment + ArcFace,
and stores a single aggregated embedding per user.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
from numpy.typing import NDArray

from vision.ai import (
    AntiSpoof,
    FaceDetector,
    FaceRecognizer,
    LandmarkEngine,
)
from vision.ai.alignment import warp_face
from vision.ai.liveness.blink import BlinkDetector
from vision.ai.liveness.eye import EyeTracker
from vision.ai.liveness.headpose import HeadPoseEstimator
from vision.config import settings
from vision.core.exceptions import (
    FaceQualityError,
    LivenessError,
    NoFaceDetectedError,
    VisionError,
)
from vision.core.logging import logger
from vision.core.types import BoundingBox, Landmarks
from vision.database import Database, FaceTemplateRepository, UserRepository
from vision.database.models import FaceTemplate, User
from vision.registration.video_source import iter_video_frames


@dataclass(slots=True)
class EnrollmentResult:
    user: User
    template: FaceTemplate
    frames_used: int
    mean_quality: float


class RegistrationService:
    def __init__(
        self,
        db: Database,
        detector: FaceDetector,
        recognizer: FaceRecognizer,
        antispoof: Optional[AntiSpoof] = None,
        landmarks: Optional[LandmarkEngine] = None,
    ) -> None:
        self.db = db
        self.detector = detector
        self.recognizer = recognizer
        self.antispoof = antispoof
        self.landmarks = landmarks
        self.users = UserRepository(db)
        self.templates = FaceTemplateRepository(db)

    # ------------------------------------------------------------------ main
    def enroll(
        self,
        *,
        name: str,
        video_source: str | Path | int,
        phone: str = "",
        email: str = "",
        notes: str = "",
        source: str = "self",
        max_seconds: float = 3.0,
        require_liveness: bool = True,
    ) -> EnrollmentResult:
        """Run the 3-second enrollment flow."""
        user = self.users.create(name=name, phone=phone, email=email, notes=notes)
        try:
            template, frames, q = self._enroll_video(
                user_id=user.user_id,
                video_source=video_source,
                max_seconds=max_seconds,
                require_liveness=require_liveness,
            )
        except Exception:
            # Roll back the user row if enrollment fails
            self.users.delete(user.user_id)
            raise
        template_row = self.templates.add(
            user_id=user.user_id,
            embedding=template.embedding,
            quality_score=template.quality_score,
            source=source,
            is_primary=True,
        )
        return EnrollmentResult(user=user, template=template_row, frames_used=frames, mean_quality=q)

    # ------------------------------------------------------------------ batch
    def enroll_from_iter(
        self,
        *,
        user_id: int,
        frames: Iterator[tuple[int, NDArray[np.uint8]]],
        source: str = "self",
        require_liveness: bool = True,
    ) -> tuple[FaceTemplate, int, float]:
        return self._enroll_from_iter(user_id, frames, source, require_liveness)

    # ---------------------------------------------------------------- helpers
    def _enroll_video(
        self,
        *,
        user_id: int,
        video_source: str | Path | int,
        max_seconds: float,
        require_liveness: bool,
    ) -> tuple[FaceTemplate, int, float]:
        return self._enroll_from_iter(
            user_id=user_id,
            frames=iter_video_frames(video_source, max_seconds=max_seconds),
            source="self",
            require_liveness=require_liveness,
        )

    def _enroll_from_iter(
        self,
        *,
        user_id: int,
        frames: Iterator[tuple[int, NDArray[np.uint8]]],
        source: str,
        require_liveness: bool,
    ) -> tuple[FaceTemplate, int, float]:
        embeds: list[NDArray[np.float32]] = []
        qualities: list[float] = []
        blink = BlinkDetector()
        eye = EyeTracker()
        head = HeadPoseEstimator()
        head_var_buf: list[float] = []
        blink_rate_buf: list[int] = []
        last_blink_count = 0
        n_used = 0
        t_start: int | None = None

        for ts_ms, frame in frames:
            if t_start is None:
                t_start = ts_ms
            try:
                det = self.detector.detect_single(frame)
            except NoFaceDetectedError:
                continue
            try:
                q = self.detector.quality_check(frame, det)
            except FaceQualityError:
                continue
            aligned = warp_face(frame, det.landmarks, output_size=112)
            if self.antispoof is not None and require_liveness:
                liveness = self.antispoof.predict_liveness(frame, det.bbox)
                if not liveness.is_real:
                    raise LivenessError(f"Spoof detected during enrollment: {liveness.spoof_kind}")
            if self.landmarks is not None:
                lms = self.landmarks.extract_landmarks(frame)
                if lms is not None:
                    ear = blink.update(lms, ts_ms)
                    eye.update(lms, ts_ms, has_refine=self.landmarks.refine)
                    hp = head.estimate(lms, frame.shape[:2], ts_ms)
                    head_var_buf.append(abs(hp.yaw) + abs(hp.pitch))
                    blink_rate_buf.append(blink.blink_count - last_blink_count)
                    last_blink_count = blink.blink_count
            emb = self.recognizer.generate_embedding(aligned)
            embeds.append(emb)
            qualities.append(q)
            n_used += 1

        if not embeds:
            raise NoFaceDetectedError("No usable frames in enrollment video")

        tpl = self.recognizer.register_embedding(
            embeds,
            user_id=user_id,
            template_id=0,  # assigned by repo
            quality_scores=qualities,
        )
        return tpl, n_used, float(np.mean(qualities))
