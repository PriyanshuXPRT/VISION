"""
Security benchmark suite.

Each attack is a directory of (image|video) files known to be spoofs.
The runner evaluates the liveness module on each sample and reports FAR,
ACER, and per-attack-kind statistics.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from training.evaluation.metrics import evaluate_binary
from vision.registration.video_source import iter_video_frames


def _load_first_frame(path: Path) -> np.ndarray | None:
    if path.suffix.lower() in {".mp4", ".mov", ".avi"}:
        for _ts, frame in iter_video_frames(path, target_fps=2, max_seconds=1.0):
            return frame
        return None
    img = cv2.imread(str(path))
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img is not None else None


def _label_for(path: Path) -> int:
    p = path.as_posix().lower()
    if "real" in p or "live" in p:
        return 1
    return 0


def run(root: Path, pipeline) -> dict:
    """Run the benchmark. Returns a dict with per-attack-kind and overall stats."""
    by_kind_scores: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for sample in tqdm(list(root.rglob("*")), desc="samples"):
        if not sample.is_file():
            continue
        if sample.suffix.lower() not in {".jpg", ".png", ".mp4", ".mov", ".avi"}:
            continue
        frame = _load_first_frame(sample)
        if frame is None:
            continue
        try:
            det = pipeline.detector.detect_single(frame)
            pred = pipeline.antispoof.predict_liveness(frame, det.bbox)
        except Exception:
            continue
        kind = sample.relative_to(root).parts[0] if sample.relative_to(root).parts else "unknown"
        by_kind_scores[kind].append((_label_for(sample), float(pred.score)))

    per_kind = {}
    all_y, all_s = [], []
    for k, lst in by_kind_scores.items():
        y = np.array([t[0] for t in lst], dtype=np.int32)
        s = np.array([t[1] for t in lst], dtype=np.float32)
        rep = evaluate_binary(y, s)
        per_kind[k] = {"n": int(s.size), "label_pos_rate": float(y.mean()) if y.size else 0.0, **rep.__dict__}
        all_y.extend(y.tolist()); all_s.extend(s.tolist())
    overall = evaluate_binary(np.array(all_y, dtype=np.int32), np.array(all_s, dtype=np.float32)).__dict__
    return {"overall": overall, "per_kind": per_kind}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    from vision.core.pipeline import build_pipeline
    pipeline = build_pipeline()
    report = run(args.root, pipeline)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(json.dumps({"overall": report["overall"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
