"""Diagnostic: dump every SCRFD detection for 10 webcam frames."""
import os
os.environ["ORT_LOG_LEVEL"] = "ERROR"
os.environ["PYTHONWARNINGS"] = "ignore"

import cv2
import numpy as np
from pathlib import Path

from scripts.pc_test import SCRFD

det = SCRFD(Path("models/buffalo_l/det_10g.onnx"), conf_thresh=0.30)
print(f"SCRFD loaded. conf=0.30 (low for diagnostic)")
print(f"Will print every raw detection in 10 frames.")
print(f"Output 'r 1234 67 89 450 230 0.95' = score=0.95, bbox=(1234,67)-(1289,230)")

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise SystemExit("ERROR: webcam not opened")
for i in range(10):
    ok, frame = cap.read()
    if not ok:
        print(f"frame {i}: read failed")
        continue
    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]
    # Manually re-run detect internals at low conf
    blob, scale = det._preprocess(frame)
    outs = det.session.run(None, {det.input_name: blob})
    raw = 0
    for idx, stride in enumerate(det.feat_stride_fpn):
        sc = outs[idx].reshape(-1)
        bb = outs[idx + 3].reshape(-1, 4)
        inds = np.where(sc >= 0.05)[0]  # extremely low
        for k in inds[:5]:  # cap per scale
            cx = (k % (2 * (det.input_size // stride))) // 2 * stride + stride // 2
            cy = (k // (2 * (det.input_size // stride))) * stride + stride // 2
            dw, dh = bb[k, 2], bb[k, 3]
            x1 = max(0, int((cx - dw * 0.5) / scale))
            y1 = max(0, int((cy - dh * 0.5) / scale))
            x2 = min(w, int((cx + dw * 0.5) / scale))
            y2 = min(h, int((cy + dh * 0.5) / scale))
            print(f"  f{i:02d} s{stride:02d} score={sc[k]:.3f} bbox=({x1:4d},{y1:4d})-({x2:4d},{y2:4d})  size={x2-x1}x{y2-y1}")
            raw += 1
    print(f"frame {i}: {raw} raw detections >=0.05 (showing top 5 per scale)")
cap.release()
print("done")
