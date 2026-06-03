"""Quick webcam smoke test."""
from pathlib import Path

import cv2
import numpy as np

from scripts.pc_test import SCRFD, ArcFace, Registry

det = SCRFD(Path("models/buffalo_l/det_10g.onnx"))
rec = ArcFace(Path("models/buffalo_l/w600k_r50.onnx"))
print("opening webcam ...")
cap = cv2.VideoCapture(0)
ok, frame = cap.read()
print("webcam read:", ok, "shape:", frame.shape if ok else None)
cap.release()
if not ok:
    raise SystemExit(1)
frame = cv2.flip(frame, 1)
dets = det.detect(frame)
print(f"detected {len(dets)} face(s) in first frame")
if dets:
    best = max(dets, key=lambda d: d["score"])
    score = best["score"]
    bb = best["bbox"]
    print(f"  best score: {score:.3f}  bbox: ({bb[0]:.0f},{bb[1]:.0f}) -> ({bb[2]:.0f},{bb[3]:.0f})")
    emb = rec.embed(frame, best["kps"])
    print(f"  embedding shape: {emb.shape}, norm: {np.linalg.norm(emb):.3f}")
    reg = Registry(Path("registry/registry.json"))
    name, sim = reg.match(emb, threshold=0.45)
    print(f"  match: {name}  sim={sim:.3f}")
print("OK")
