"""
Run the full evaluation suite on a trained pipeline.

Loads a VisionPipeline (or builds one), runs the antispoof + recognition eval
on a labelled set, and writes a JSON report.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

from training.evaluation.metrics import evaluate_binary, tar_at_far


def _eval_antispoof(pipeline, root: Path) -> dict:
    y, s = [], []
    for label, cls in [(1, "real"), (0, "print"), (0, "mobile_screen"), (0, "laptop_screen"), (0, "replay_video")]:
        d = root / cls
        if not d.is_dir():
            continue
        for img in tqdm(list(d.rglob("*.jpg")), desc=f"fas/{cls}"):
            import cv2
            arr = cv2.cvtColor(cv2.imread(str(img)), cv2.COLOR_BGR2RGB)
            try:
                det = pipeline.detector.detect_single(arr)
            except Exception:
                continue
            pred = pipeline.antispoof.predict_liveness(arr, det.bbox)
            y.append(label); s.append(pred.score)
    rep = evaluate_binary(np.array(y), np.array(s))
    return rep.__dict__


def _eval_recognition(pipeline, pairs: Path) -> dict:
    from vision.ai.alignment import warp_face
    g, i = [], []
    for line in pairs.read_text().splitlines():
        a, b, lab = line.split("\t")
        import cv2
        aa = cv2.cvtColor(cv2.imread(a), cv2.COLOR_BGR2RGB)
        bb = cv2.cvtColor(cv2.imread(b), cv2.COLOR_BGR2RGB)
        try:
            ea = pipeline.recognizer.generate_embedding(warp_face(aa, pipeline.detector.detect_single(aa).landmarks, 112))
            eb = pipeline.recognizer.generate_embedding(warp_face(bb, pipeline.detector.detect_single(bb).landmarks, 112))
        except Exception:
            continue
        sim = float(np.dot(ea, eb))
        (g if int(lab) == 1 else i).append(sim)
    g_arr, i_arr = np.array(g, dtype=np.float32), np.array(i, dtype=np.float32)
    return {
        "tars": {
            str(far): {
                "tar": tar_at_far(g_arr, i_arr, far_target=far)[0],
                "threshold": tar_at_far(g_arr, i_arr, far_target=far)[1],
            } for far in (1e-2, 1e-3, 1e-4)
        },
        "n_genuine": int(g_arr.size),
        "n_impostor": int(i_arr.size),
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--antispoof-root", type=Path, default=None)
    p.add_argument("--pairs", type=Path, default=None)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    from vision.core.pipeline import build_pipeline
    pipeline = build_pipeline()
    report = {}
    if args.antispoof_root:
        report["anti_spoof"] = _eval_antispoof(pipeline, args.antispoof_root)
    if args.pairs:
        report["recognition"] = _eval_recognition(pipeline, args.pairs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
