"""Smoke test for the anti-replay auth flow (headless, no UI)."""
import os
os.environ["ORT_LOG_LEVEL"] = "ERROR"
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["VISION_DOWNLOAD_PRETRAINED"] = "false"

from pathlib import Path
import numpy as np
import cv2

from scripts.pc_test import (
    SCRFD, ArcFace, Registry, BlinkDetector, HeadPoseEstimator, LandmarkEngine,
    AntiSpoofCheck, REGISTRY,
)
from scripts.liveness_runtime import run_auth_liveness

det = SCRFD(Path("models/buffalo_l/det_10g.onnx"))
rec = ArcFace(Path("models/buffalo_l/w600k_r50.onnx"))
land = LandmarkEngine()
blink = BlinkDetector()
head = HeadPoseEstimator()
asp = AntiSpoofCheck(Path("models/buffalo_l"), threshold=0.30)
reg = Registry(REGISTRY)
print(f"antispoof source: {asp._source}, threshold: {asp.threshold}")
print(f"users in registry: {reg.list_names()}")

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise SystemExit("webcam not opened")
print("webcam opened — running liveness flow (10s) ...")

# Override the active challenges to a fixed one for the smoke test.
import scripts.liveness_runtime as lr
lr.CHALLENGES = [
    {"id": "turn_left", "text": "Turn your head LEFT", "kind": "yaw", "target": -1},
    {"id": "blink_2x",  "text": "Blink TWICE",         "kind": "blinks", "target": 2},
]

challenge, motion, spoof, embs, blink_count, mean_ear, yaws, pitches, crops = (
    run_auth_liveness(
        cap, det, rec, land, blink, head, asp,
        challenge_seconds=4.0, capture_seconds=1.5, target_fps=10.0,
    )
)
cap.release()

print(f"\n=== RESULT ===")
print(f"challenge: {challenge.text}  -> {'PASS' if challenge.passed else 'FAIL'}  ({challenge.detail})")
print(f"motion:    mean={motion.mean_px_jitter:.3f}px max={motion.max_px_jitter:.3f}px n={motion.frames_sampled}  -> {'PASS' if motion.passes else 'FAIL'}")
print(f"antispoof: score={spoof.score:.3f} src={spoof.source} thresh={spoof.threshold}  -> {'PASS' if spoof.passed else 'FAIL'}")
print(f"blinks:    {blink_count}, mean_ear: {mean_ear:.3f}")
print(f"embeddings captured: {len(embs)}")

if embs and reg.users:
    agg = np.mean(np.stack(embs), axis=0)
    agg /= np.linalg.norm(agg) + 1e-9
    name, sim = reg.match(agg, threshold=0.45)
    print(f"match: {name} sim={sim:.3f}")
