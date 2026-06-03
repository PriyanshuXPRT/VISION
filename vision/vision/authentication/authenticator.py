"""
Authentication orchestrator.

End-to-end flow:
    video/frames → face detect → quality → liveness (passive + temporal + blink/eye/pose)
    → embedding → 1:N identification → accept/reject
"""
from __future__ import annotations

import time
from dataclasses import dataclass
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
from vision.ai.embedding_search import VectorIndex
from vision.ai.liveness.blink import BlinkDetector
from vision.ai.liveness.eye import EyeTracker
from vision.ai.liveness.headpose import HeadPoseEstimator
from vision.ai.liveness.temporal import (
    TemporalLivenessEngine,
    build_feature_vector,
)
from vision.config import settings
from vision.core.exceptions import (
    LivenessError,
    NoFaceDetectedError,
    SpoofDetectedError,
)
from vision.core.logging import logger
from vision.core.types import (
    AuthenticationResult,
    Decision,
    FrameAnalysis,
    LivenessReport,
    LivenessSignal,
    SpoofKind,
)
from vision.database import (
    AuthLogRepository,
    Database,
    DeviceRepository,
    FaceTemplateRepository,
    UserRepository,
)
from vision.database.models import AuthLog
from vision.identification import IdentificationService


@dataclass(slots=True)
class AuthConfig:
    min_frames: int = 12
    min_liveness: float = 0.80
    min_similarity: float = 0.45
    require_blink: bool = True
    require_motion: bool = True
    require_pose: bool = False
    spoof_threshold: float = 0.50
    device_id: Optional[str] = None


class Authenticator:
    """Stateless, thread-friendly authentication pipeline."""

    def __init__(
        self,
        db: Database,
        detector: FaceDetector,
        recognizer: FaceRecognizer,
        antispoof: AntiSpoof,
        landmarks: LandmarkEngine,
        identification: IdentificationService,
        temporal: Optional[TemporalLivenessEngine] = None,
    ) -> None:
        self.db = db
        self.detector = detector
        self.recognizer = recognizer
        self.antispoof = antispoof
        self.landmarks = landmarks
        self.identification = identification
        self.temporal = temporal or TemporalLivenessEngine()
        self.users = UserRepository(db)
        self.templates = FaceTemplateRepository(db)
        self.logs = AuthLogRepository(db)
        self.devices = DeviceRepository(db)

    # ------------------------------------------------------------------ main
    def authenticate(
        self,
        frames: Iterator[tuple[int, NDArray[np.uint8]]],
        config: AuthConfig | None = None,
    ) -> AuthenticationResult:
        """Run the full pipeline on a stream of (timestamp_ms, frame_rgb) tuples."""
        cfg = config or AuthConfig()
        t0 = time.perf_counter()
        blink = BlinkDetector()
        eye = EyeTracker()
        head = HeadPoseEstimator()
        self.temporal.reset()

        accepted: list[FrameAnalysis] = []
        rejected_early: list[FrameAnalysis] = []
        last_emb: Optional[NDArray[np.float32]] = None
        last_id: Optional[dict] = None
        reason = ""
        decision = Decision.INCONCLUSIVE
        n_used = 0

        for ts_ms, frame in frames:
            n_used += 1
            analysis = self._analyze_frame(frame, ts_ms, blink, eye, head)
            if analysis is None:
                continue
            if analysis.is_spoof:
                rejected_early.append(analysis)
                continue
            accepted.append(analysis)
            if analysis.embedding is not None:
                last_emb = analysis.embedding
                last_id = self.identification.identify(analysis.embedding, top_k=1)
            # Stop early when we have enough evidence
            if len(accepted) >= cfg.min_frames and self.temporal.is_ready():
                break

        # Aggregate
        liveness_report = self._build_liveness_report(
            accepted, rejected_early, blink, eye, head, cfg
        )
        frames_processed = len(accepted)
        latency_ms = int((time.perf_counter() - t0) * 1000)
        if last_emb is None or last_id is None or last_id.get("best_match") is None:
            decision = Decision.REJECT
            reason = reason or "no_match"
        elif not liveness_report.final_score >= cfg.min_liveness:
            decision = Decision.REJECT
            reason = reason or "liveness_failed"
        elif liveness_report.spoof_kind != SpoofKind.NONE:
            decision = Decision.REJECT
            reason = reason or f"spoof_{liveness_report.spoof_kind.value}"
        else:
            decision = Decision.ACCEPT
            reason = "ok"

        # Persist log
        best = last_id.get("best_match") if last_id else None
        result = AuthenticationResult(
            decision=decision,
            user_id=best.user_id if best else None,
            user_name=best.name if best else None,
            similarity=best.similarity if best else 0.0,
            liveness=liveness_report,
            frames_processed=frames_processed,
            latency_ms=latency_ms,
            reason=reason,
        )
        try:
            self.logs.add(
                user_id=result.user_id,
                liveness_score=liveness_report.final_score,
                similarity_score=result.similarity,
                result=decision.value,
                reason=reason,
                device_id=cfg.device_id,
                frames=frames_processed,
                latency_ms=latency_ms,
                spoof_kind=liveness_report.spoof_kind.value,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to persist auth log: {}", exc)
        return result

    # ------------------------------------------------------------------ per-frame
    def _analyze_frame(
        self,
        frame: NDArray[np.uint8],
        ts_ms: int,
        blink: BlinkDetector,
        eye: EyeTracker,
        head: HeadPoseEstimator,
    ) -> Optional[FrameAnalysis]:
        try:
            det = self.detector.detect_single(frame)
        except NoFaceDetectedError:
            return None
        aligned = warp_face(frame, det.landmarks, output_size=112)
        fas = self.antispoof.predict_liveness(frame, det.bbox)
        is_spoof = (not fas.is_real) or fas.score < 0.50
        landmarks = self.landmarks.extract_landmarks(frame)
        ear = 0.0
        eye_motion = 0.0
        yaw = pitch = roll = 0.0
        if landmarks is not None:
            ear = blink.update(landmarks, ts_ms)
            eye_motion = eye.update(landmarks, ts_ms, has_refine=self.landmarks.refine)
            hp = head.estimate(landmarks, frame.shape[:2], ts_ms)
            yaw, pitch, roll = hp.yaw, hp.pitch, hp.roll

        # Build feature and feed temporal
        head_motion_rms = float(np.sqrt(0.5 * (yaw * yaw + pitch * pitch)) / 30.0)
        elapsed_s = max(0.001, (ts_ms - (self._t0 or ts_ms)) / 1000.0)
        blink_rate = blink.blink_count / elapsed_s
        feat = build_feature_vector(
            passive_liveness=fas.score,
            ear=ear,
            eye_motion=eye_motion,
            yaw=yaw,
            pitch=pitch,
            roll=roll,
            head_motion_rms=head_motion_rms,
            blink_rate_per_sec=blink_rate,
        )
        self._t0 = self._t0 or ts_ms
        self.temporal.push(feat)

        emb: Optional[NDArray[np.float32]] = None
        try:
            emb = self.recognizer.generate_embedding(aligned)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Embedding failed for frame: {}", exc)

        return FrameAnalysis(
            timestamp_ms=ts_ms,
            bbox=det.bbox,
            landmarks=landmarks,
            aligned_face=aligned,
            passive_liveness=fas.score,
            is_spoof=is_spoof,
            spoof_kind=fas.spoof_kind,
            ear=ear,
            eye_motion=eye_motion,
            yaw=yaw,
            pitch=pitch,
            roll=roll,
            quality_score=det.score,
            embedding=emb,
        )

    # ------------------------------------------------------------------ aggregation
    def _build_liveness_report(
        self,
        accepted: list[FrameAnalysis],
        rejected: list[FrameAnalysis],
        blink: BlinkDetector,
        eye: EyeTracker,
        head: HeadPoseEstimator,
        cfg: AuthConfig,
    ) -> LivenessReport:
        passive = float(np.mean([f.passive_liveness for f in accepted])) if accepted else 0.0
        br = blink.result()
        er = eye.result()
        hr = head.result()
        tm = self.temporal.score() if self.temporal.is_ready() else None

        accepted_signals: list[LivenessSignal] = []
        rejected_signals: list[LivenessSignal] = []

        if passive >= 0.5:
            accepted_signals.append(LivenessSignal.PASSIVE_FAS)
        else:
            rejected_signals.append(LivenessSignal.PASSIVE_FAS)
        if br.blink_count >= 1:
            accepted_signals.append(LivenessSignal.BLINK)
        elif cfg.require_blink:
            rejected_signals.append(LivenessSignal.BLINK)
        if er.motion_score >= settings.eye_motion_thresh:
            accepted_signals.append(LivenessSignal.EYE_MOTION)
        elif cfg.require_motion:
            rejected_signals.append(LivenessSignal.EYE_MOTION)
        if hr.head_pose_score >= 0.2 or not cfg.require_pose:
            accepted_signals.append(LivenessSignal.HEAD_POSE)
        elif cfg.require_pose:
            rejected_signals.append(LivenessSignal.HEAD_POSE)
        if tm is not None and tm.is_live:
            accepted_signals.append(LivenessSignal.TEMPORAL)
        else:
            rejected_signals.append(LivenessSignal.TEMPORAL)

        # Final: weighted average
        weights = {
            LivenessSignal.PASSIVE_FAS: 0.35,
            LivenessSignal.BLINK: 0.10,
            LivenessSignal.EYE_MOTION: 0.10,
            LivenessSignal.HEAD_POSE: 0.10,
            LivenessSignal.TEMPORAL: 0.35,
        }
        scores = {
            LivenessSignal.PASSIVE_FAS: passive,
            LivenessSignal.BLINK: br.blink_score,
            LivenessSignal.EYE_MOTION: er.motion_score,
            LivenessSignal.HEAD_POSE: hr.head_pose_score,
            LivenessSignal.TEMPORAL: tm.final_score if tm else 0.0,
        }
        if accepted_signals:
            final = sum(weights[s] * scores[s] for s in accepted_signals)
            denom = sum(weights[s] for s in accepted_signals) or 1.0
            final = final / denom
        else:
            final = 0.0
        spoof = SpoofKind.NONE if passive >= 0.5 else (
            rejected[0].spoof_kind if rejected else SpoofKind.UNKNOWN
        )
        return LivenessReport(
            blink_count=br.blink_count,
            blink_score=br.blink_score,
            eye_motion_score=er.motion_score,
            head_pose_score=hr.head_pose_score,
            passive_score=passive,
            temporal_score=tm.final_score if tm else 0.0,
            final_score=float(np.clip(final, 0.0, 1.0)),
            spoof_kind=spoof,
            accepted_signals=accepted_signals,
            rejected_signals=rejected_signals,
        )

    # ------------------------------------------------------------------ query
    def list_recent_logs(self, limit: int = 100):
        return self.logs.list_recent(limit=limit)
