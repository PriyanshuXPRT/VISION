"""Anti-replay liveness stack — three layers to defeat video / photo attacks.

Layer 1 — Active challenge-response
    A random prompt ("turn head LEFT", "blink 2x", "look UP") is shown to the
    user; the system verifies the response in real time. A pre-recorded video
    cannot predict the random challenge, and a looping video cannot reliably
    reproduce every direction on demand.

Layer 2 — Temporal motion variance
    Real faces exhibit continuous micro-jitter (1-2 px frame-to-frame on
    landmarks, plus natural head sway). Video replays and printed photos
    produce near-zero motion between frames. We measure per-frame landmark
    displacement and require at least N px mean jitter over the window.

Layer 3 — Passive anti-spoof (MiniFASNet V2)
    If real FAS weights are available, runs the MiniFASNetV2 ensemble for
    a passive "real vs printed/screen" score. Falls back to a heuristic
    texture-score when weights aren't present.
"""
from __future__ import annotations

import random
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np
from numpy.typing import NDArray

from vision.ai.liveness.blink import BlinkDetector
from vision.ai.liveness.headpose import HeadPoseEstimator
from vision.ai.landmarks.landmark_engine import LandmarkEngine


# ----------------------------------------------------------------- challenges
CHALLENGES: list[dict] = [
    {"id": "turn_left",  "text": "Turn your head LEFT",  "kind": "yaw",   "target": -1},
    {"id": "turn_right", "text": "Turn your head RIGHT", "kind": "yaw",   "target": +1},
    {"id": "look_up",    "text": "Look UP",              "kind": "pitch", "target": -1},
    {"id": "look_down",  "text": "Look DOWN",            "kind": "pitch", "target": +1},
    {"id": "blink_2x",   "text": "Blink TWICE",          "kind": "blinks", "target": 2},
    {"id": "blink_3x",   "text": "Blink THREE times",    "kind": "blinks", "target": 3},
]


@dataclass(slots=True)
class ChallengeResult:
    challenge_id: str
    text: str
    passed: bool
    detail: str
    yaw_delta: float = 0.0
    pitch_delta: float = 0.0
    blink_delta: int = 0


# ----------------------------------------------------------------- layer 2
@dataclass(slots=True)
class MotionVarianceResult:
    mean_px_jitter: float       # mean frame-to-frame landmark displacement (px)
    max_px_jitter: float
    frames_sampled: int
    passes: bool
    threshold_px: float


class MotionVarianceDetector:
    """Measures per-frame landmark jitter.

    A real face in front of a webcam always shows some pixel-level motion
    on the 468/478 face-mesh points: head micro-sway, blink tremor,
    pupil jitter, breathing.  Even a perfectly still subject moves 0.3-1.5
    px between consecutive frames at 30 fps.  A pre-recorded video played
    back at the same frame rate typically has stretches of 0-0.05 px motion
    (the codec is bandwidth-limited and the subject wasn't perfectly still
    during recording, but the *distribution* of jitter is different).
    """

    def __init__(self, threshold_px: float = 0.20, history: int = 240) -> None:
        self.threshold_px = threshold_px
        self._prev: Optional[NDArray[np.float32]] = None
        self._diffs: deque[float] = deque(maxlen=history)

    def update(self, landmarks) -> None:
        pts = landmarks.points[:, :2]
        if self._prev is not None and pts.shape == self._prev.shape:
            d = float(np.linalg.norm(pts - self._prev, axis=1).mean())
            self._diffs.append(d)
        self._prev = pts

    def reset(self) -> None:
        self._prev = None
        self._diffs.clear()

    def result(self) -> MotionVarianceResult:
        if not self._diffs:
            return MotionVarianceResult(0.0, 0.0, 0, False, self.threshold_px)
        diffs = np.array(self._diffs, dtype=np.float32)
        return MotionVarianceResult(
            mean_px_jitter=float(diffs.mean()),
            max_px_jitter=float(diffs.max()),
            frames_sampled=len(self._diffs),
            passes=float(diffs.mean()) >= self.threshold_px,
            threshold_px=self.threshold_px,
        )


# ----------------------------------------------------------------- layer 3
@dataclass(slots=True)
class AntiSpoofResult:
    score: float        # 0..1, higher = more likely real
    passed: bool
    is_real: bool
    threshold: float
    source: str         # "minifasnet" or "heuristic" or "stub"


class AntiSpoofCheck:
    """Passively estimates real-face probability.

    Tries to load the MiniFASNet V2 ensemble from `models/buffalo_l/`; if
    that fails (no weights or wrong model) it falls back to a luminance
    + Laplacian-variance heuristic that catches the most obvious printed
    photos (which are flat and have low high-frequency content).
    """

    def __init__(self, models_dir: Path, threshold: float = 0.50) -> None:
        self.threshold = threshold
        self._impl = None
        self._source = "heuristic"
        try:
            from vision.ai.antispoof.anti_spoof import AntiSpoof
            self._impl = AntiSpoof(model_dir=models_dir, threshold=threshold)
            self._source = "minifasnet"
        except Exception:
            # weights missing or model shape mismatch — use heuristic
            self._impl = None

    def predict(self, frame_bgr: NDArray[np.uint8], bbox) -> AntiSpoofResult:
        if self._impl is not None:
            try:
                pred = self._impl.predict_liveness(frame_bgr, bbox)
                return AntiSpoofResult(
                    score=pred.score,
                    passed=pred.is_real,
                    is_real=pred.is_real,
                    threshold=self.threshold,
                    source=self._source,
                )
            except Exception:
                pass
        # ----- heuristic fallback -----
        x1, y1, x2, y2 = (int(v) for v in bbox)
        crop = frame_bgr[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
        if crop.size == 0:
            return AntiSpoofResult(0.0, False, False, self.threshold, "heuristic")
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        lap = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        # real skin: typically lap > 60; matte paper: lap ~5-25; phone screen: 30-60
        score = min(1.0, lap / 120.0)
        return AntiSpoofResult(
            score=score,
            passed=score >= self.threshold,
            is_real=score >= self.threshold,
            threshold=self.threshold,
            source="heuristic",
        )


# ----------------------------------------------------------------- orchestrator
@dataclass(slots=True)
class LivenessBundle:
    challenge: ChallengeResult
    motion: MotionVarianceResult
    antispoof: AntiSpoofResult
    blink_count: int
    mean_ear: float
    final_pass: bool
    score: float
    reasons: list[str] = field(default_factory=list)


def pick_challenge(rng: random.Random | None = None) -> dict:
    rng = rng or random.Random()
    return rng.choice(CHALLENGES)


def _check_challenge(
    challenge: dict,
    yaw_series: list[float],
    pitch_series: list[float],
    blink_count: int,
    blink_at_start: int,
) -> ChallengeResult:
    cid = challenge["id"]
    text = challenge["text"]
    if not yaw_series or not pitch_series:
        return ChallengeResult(cid, text, False, "no head-pose data")
    yaw_start, yaw_end = yaw_series[0], yaw_series[-1]
    pitch_start, pitch_end = pitch_series[0], pitch_series[-1]
    yaw_delta = yaw_end - yaw_start
    pitch_delta = pitch_end - pitch_start
    blink_delta = blink_count - blink_at_start

    if challenge["kind"] == "yaw":
        ok = (yaw_delta <= -12.0) if challenge["target"] < 0 else (yaw_delta >= 12.0)
        return ChallengeResult(
            cid, text, ok,
            f"yaw moved {yaw_delta:+.1f} deg (need {'<= -12' if challenge['target'] < 0 else '>= +12'})",
            yaw_delta=yaw_delta, pitch_delta=pitch_delta, blink_delta=blink_delta,
        )
    if challenge["kind"] == "pitch":
        ok = (pitch_delta <= -8.0) if challenge["target"] < 0 else (pitch_delta >= 8.0)
        return ChallengeResult(
            cid, text, ok,
            f"pitch moved {pitch_delta:+.1f} deg (need {'<= -8' if challenge['target'] < 0 else '>= +8'})",
            yaw_delta=yaw_delta, pitch_delta=pitch_delta, blink_delta=blink_delta,
        )
    if challenge["kind"] == "blinks":
        ok = blink_delta >= challenge["target"]
        return ChallengeResult(
            cid, text, ok,
            f"blinked {blink_delta} time(s) (need >= {challenge['target']})",
            yaw_delta=yaw_delta, pitch_delta=pitch_delta, blink_delta=blink_delta,
        )
    return ChallengeResult(cid, text, False, "unknown challenge type")


def run_auth_liveness(
    cap: cv2.VideoCapture,
    detector,            # SCRFD
    recognizer,          # ArcFace
    landmarks: LandmarkEngine,
    blink: BlinkDetector,
    head: HeadPoseEstimator,
    antispoof: AntiSpoofCheck,
    *,
    challenge_seconds: float = 3.0,
    capture_seconds: float = 1.5,
    target_fps: float = 12.0,
) -> tuple[ChallengeResult, MotionVarianceResult, AntiSpoofResult, list[NDArray[np.float32]], int, float, list[float], list[float], list[NDArray[np.uint8]]]:
    """Run the full anti-replay liveness flow. Returns:
        (challenge, motion, antispoof, embeddings, blink_count, mean_ear,
         yaw_series, pitch_series, crops)
    """
    motion = MotionVarianceDetector(threshold_px=0.20)
    yaw_series: list[float] = []
    pitch_series: list[float] = []
    embeddings: list[NDArray[np.float32]] = []
    crops: list[NDArray[np.uint8]] = []
    period = 1.0 / target_fps
    last_t = 0.0

    # Pick a random challenge at the start of the flow.
    challenge = pick_challenge()
    print(f"[liveness] challenge: {challenge['text']}", flush=True)
    blink_at_start = blink.blink_count
    challenge_start_t = time.time()
    challenge_done = False
    challenge_result: Optional[ChallengeResult] = None
    # Mark on the window
    last_status = f"CHALLENGE: {challenge['text']}"

    while True:
        now = time.time()
        if now - last_t < period:
            time.sleep(0.001)
            continue
        last_t = now
        ok, frame = grab(cap)
        if not ok:
            break
        dets = detector.detect(frame)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Always update motion + blink + head
        if dets:
            best = max(dets, key=lambda d: (d["bbox"][2] - d["bbox"][0]) * (d["bbox"][3] - d["bbox"][1]))
            x1, y1, x2, y2 = (int(v) for v in best["bbox"])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            lms = landmarks.extract_landmarks(rgb)
            if lms is not None:
                motion.update(lms)
                ear = blink.update(lms, int(now * 1000))
                hp = head.estimate(lms, frame.shape[:2], int(now * 1000))
                yaw_series.append(hp.yaw)
                pitch_series.append(hp.pitch)
            # Anti-spoof on the largest face
            from vision.core.types import BoundingBox
            bbox = best["bbox"]
            if (bbox[2] - bbox[0]) > 60 and (bbox[3] - bbox[1]) > 60:
                asp = antispoof.predict(frame, bbox)

        # ----- phase 1: challenge (challenge_seconds) -----
        elapsed = now - challenge_start_t
        if not challenge_done:
            # also collect embeddings during challenge
            if dets:
                emb = recognizer.embed(frame, best["kps"])
                embeddings.append(emb)
                crop = frame[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
                crops.append(crop if crop.size else np.zeros((112, 112, 3), dtype=np.uint8))
            cv2.putText(
                frame, f"CHALLENGE: {challenge['text']}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 255), 2,
            )
            cv2.putText(
                frame, f"time left: {max(0, challenge_seconds - elapsed):.1f}s",
                (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2,
            )
            cv2.putText(
                frame, f"yaw: {yaw_series[-1] if yaw_series else 0:+.1f}  pitch: {pitch_series[-1] if pitch_series else 0:+.1f}  blinks: {blink.blink_count - blink_at_start}",
                (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2,
            )
            cv2.imshow("V.I.S.I.O.N. PC Test", frame)
            cv2.waitKey(1)
            if elapsed >= challenge_seconds:
                challenge_result = _check_challenge(
                    challenge, yaw_series, pitch_series,
                    blink.blink_count, blink_at_start,
                )
                challenge_done = True
                print(f"[liveness] challenge result: passed={challenge_result.passed} ({challenge_result.detail})", flush=True)
                capture_start = now
            continue

        # ----- phase 2: capture (capture_seconds) -----
        capture_elapsed = now - capture_start
        if dets and capture_elapsed <= capture_seconds:
            emb = recognizer.embed(frame, best["kps"])
            embeddings.append(emb)
            crop = frame[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
            crops.append(crop if crop.size else np.zeros((112, 112, 3), dtype=np.uint8))
        cv2.putText(
            frame, "HOLD STILL... matching", (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2,
        )
        cv2.putText(
            frame, f"capture {min(len(embeddings), int(capture_seconds * target_fps))}/{int(capture_seconds * target_fps)}",
            (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
        )
        cv2.imshow("V.I.S.I.O.N. PC Test", frame)
        cv2.waitKey(1)
        if capture_elapsed > capture_seconds and len(embeddings) >= int(capture_seconds * target_fps * 0.5):
            break
        if capture_elapsed > capture_seconds + 1.5:
            break  # safety timeout

    # Final motion + antispoof summary across the whole window
    motion_res = motion.result()
    # Use the most recent face bbox for a final antispoof
    last_bbox = None
    if dets:
        last_bbox = max(dets, key=lambda d: (d["bbox"][2] - d["bbox"][0]) * (d["bbox"][3] - d["bbox"][1]))["bbox"]
    if last_bbox is not None:
        asp_res = antispoof.predict(frame, last_bbox)
    else:
        asp_res = AntiSpoofResult(0.0, False, False, antispoof.threshold, antispoof._source)
    # ear mean
    if blink.ear_history:
        mean_ear = float(np.mean(list(blink.ear_history)))
    else:
        mean_ear = 0.0

    if challenge_result is None:
        challenge_result = ChallengeResult(
            challenge["id"], challenge["text"], False, "no challenge result (timeout)"
        )
    return (
        challenge_result, motion_res, asp_res, embeddings,
        blink.blink_count, mean_ear, yaw_series, pitch_series, crops,
    )


def grab(cap: cv2.VideoCapture) -> tuple[bool, NDArray[np.uint8]]:
    ok, frame = cap.read()
    if not ok:
        return False, np.zeros((1, 1, 3), dtype=np.uint8)
    return True, cv2.flip(frame, 1)
