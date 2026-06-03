"""Smoke test for PASSIVE anti-replay liveness (no user interaction)."""
import os
os.environ["ORT_LOG_LEVEL"] = "ERROR"
os.environ["PYTHONWARNINGS"] = "ignore"

from pathlib import Path
import numpy as np
import cv2

from scripts.pc_test import (
    SCRFD, ArcFace, Registry, AntiSpoofCheck, PassiveLivenessDetector,
    REGISTRY,
)
from scripts.pc_test import run_passive_auth

det = SCRFD(Path("models/buffalo_l/det_10g.onnx"))
rec = ArcFace(Path("models/buffalo_l/w600k_r50.onnx"))
asp = AntiSpoofCheck(Path("models/buffalo_l"), threshold=0.30)
passive = PassiveLivenessDetector(threshold=0.45)
reg = Registry(REGISTRY)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise SystemExit("webcam not opened")

print("running PASSIVE liveness — just look at the camera (3s) ...")
res, spoof, embs, frames = run_passive_auth(
    cap, det, rec, passive, asp, n_seconds=3.0, target_fps=10.0,
)
cap.release()

print(f"\n=== PASSIVE RESULT ===")
print(f"score:       {res.score:.3f}  (thresh {res.threshold})  -> {'PASS' if res.passed else 'FAIL'}")
print(f"  motion:    {res.motion_score:.3f}")
print(f"  texture:   {res.texture_score:.3f}")
print(f"  color:     {res.color_score:.3f}")
print(f"  edges:     {res.edge_score:.3f}")
print(f"  refresh:   {res.refresh_score:.3f}  (1.0 = no screen peak)")
print(f"  frames:    {res.frames_sampled}")
print(f"  notes:     {res.notes}")
print(f"antispoof:   {spoof.score:.3f}  (src={spoof.source})  -> {'PASS' if spoof.passed else 'FAIL'}")
print(f"embeddings:  {len(embs)}  frames_captured: {frames}")

if embs and reg.users:
    agg = np.mean(np.stack(embs), axis=0)
    agg /= np.linalg.norm(agg) + 1e-9
    name, sim = reg.match(agg, threshold=0.45)
    liveness_pass = res.passed and spoof.passed
    decision = "ACCEPT" if (name != "?" and liveness_pass) else "REJECT"
    print(f"match: {name} sim={sim:.3f}")
    print(f"decision: {decision}")
