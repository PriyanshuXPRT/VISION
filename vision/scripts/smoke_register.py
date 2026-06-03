"""End-to-end test: register 1 user, then match against fresh frame."""
import os
os.environ["ORT_LOG_LEVEL"] = "ERROR"
os.environ["PYTHONWARNINGS"] = "ignore"

from pathlib import Path
import numpy as np
import cv2

from scripts.pc_test import SCRFD, ArcFace, Registry

det = SCRFD(Path("models/buffalo_l/det_10g.onnx"))
rec = ArcFace(Path("models/buffalo_l/w600k_r50.onnx"))
reg = Registry(Path("registry/registry.json"))

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise SystemExit("webcam not opened")

# collect 10 embeddings
embs = []
for i in range(10):
    ok, frame = cap.read()
    if not ok:
        break
    frame = cv2.flip(frame, 1)
    dets = det.detect(frame)
    if not dets:
        print(f"  frame {i}: no face")
        continue
    best = max(dets, key=lambda d: (d["bbox"][2] - d["bbox"][0]) * (d["bbox"][3] - d["bbox"][1]))
    emb = rec.embed(frame, best["kps"])
    embs.append(emb)
    print(f"  frame {i}: face detected score={best['score']:.3f} bbox={tuple(int(v) for v in best['bbox'])}")
cap.release()

if not embs:
    raise SystemExit("no faces captured in 10 frames")

# Register
name = "smoke_user"
reg.delete(name)
reg.add_embeddings(name, embs)
print(f"\nRegistered '{name}' with {len(embs)} embeddings")

# Match
agg = np.mean(np.stack(embs), axis=0)
agg /= np.linalg.norm(agg) + 1e-9
match_name, sim = reg.match(agg, threshold=0.45)
print(f"Match: {match_name} sim={sim:.3f}")
assert match_name == name, f"self-match failed (got {match_name})"
print("OK - registry + index working")
