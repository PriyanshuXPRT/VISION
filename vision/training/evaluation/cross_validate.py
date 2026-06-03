"""
Cross-validate a recognition model on a labelled pair list.

Pair list format (text):
    <probe_path>\t<gallery_path>\t<is_genuine_int>

Usage:
    python -m training.evaluation.cross_validate --pairs eval/lfw_pairs.txt --onnx onnx_models/recognition/arcface_r100.onnx
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from vision.ai.alignment import warp_face
from vision.ai.detection import FaceDetector
from vision.ai.recognition import FaceRecognizer
from training.evaluation.metrics import tar_at_far


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--pairs", type=Path, required=True)
    p.add_argument("--recognizer", type=str, default="insightface")
    p.add_argument("--max", type=int, default=0)
    args = p.parse_args()

    detector = FaceDetector()
    recognizer = FaceRecognizer()
    genuine, impostor = [], []
    lines = args.pairs.read_text().strip().splitlines()
    if args.max:
        lines = lines[: args.max]
    for line in tqdm(lines, desc="pairs"):
        a, b, label = line.split("\t")
        is_genuine = int(label) == 1
        img_a = cv2.cvtColor(cv2.imread(a), cv2.COLOR_BGR2RGB)
        img_b = cv2.cvtColor(cv2.imread(b), cv2.COLOR_BGR2RGB)
        try:
            ea = recognizer.generate_embedding(warp_face(img_a, detector.detect_single(img_a).landmarks, 112))
            eb = recognizer.generate_embedding(warp_face(img_b, detector.detect_single(img_b).landmarks, 112))
        except Exception:
            continue
        sim = float(np.dot(ea, eb))
        (genuine if is_genuine else impostor).append(sim)
    g = np.array(genuine, dtype=np.float32)
    i = np.array(impostor, dtype=np.float32)
    for far in (1e-2, 1e-3, 1e-4):
        tar, thr = tar_at_far(g, i, far_target=far)
        print(f"TAR@FAR={far:.0e} → TAR={tar:.4f}  threshold={thr:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
