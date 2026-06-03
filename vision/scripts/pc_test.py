"""
Interactive webcam test for V.I.S.I.O.N.

Opens your PC webcam, runs real SCRFD + ArcFace + MediaPipe liveness,
and lets you register / identify / delete users live.

Keys (in the OpenCV window):
    r  -> register a new user (you'll be prompted for a name in the console)
    a  -> authenticate (captures ~3 s and prints the decision)
    l  -> list registered users
    d  -> delete a user (prompted in the console)
    q  -> quit

Storage:  vision/registry/registry.json
Real weights: models/buffalo_l/{det_10g.onnx,w600k_r50.onnx}
"""
from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np
from numpy.typing import NDArray

# Silence insightface / onnxruntime chatty logs before importing.
os.environ.setdefault("INSIGHTFACE_LOG_LEVEL", "ERROR")
os.environ.setdefault("ORT_LOG_LEVEL", "ERROR")
os.environ.setdefault("PYTHONWARNINGS", "ignore::UserWarning,ignore::DeprecationWarning")
# Suppress the harmless dynamic-batch shape warnings on stderr.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
# MiniFASNet mirrors are all dead — don't try to download.  Place files
# in models/buffalo_l/ manually to use the real model.
os.environ.setdefault("VISION_DOWNLOAD_PRETRAINED", "false")

from vision.ai.liveness.blink import BlinkDetector
from vision.ai.liveness.eye import EyeTracker
from vision.ai.liveness.headpose import HeadPoseEstimator
from vision.ai.landmarks.landmark_engine import LandmarkEngine
from vision.core.logging import setup_logging
from vision.ai.embedding_search.vector_index import BruteForceIndex
from scripts.liveness_runtime import (
    AntiSpoofCheck, LivenessBundle, MotionVarianceDetector, run_auth_liveness,
)
from scripts.passive_liveness import PassiveLivenessDetector, PassiveLivenessResult

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models" / "buffalo_l"
DET_PATH = MODELS / "det_10g.onnx"
REC_PATH = MODELS / "w600k_r50.onnx"
REGISTRY = ROOT / "registry" / "registry.json"
REGISTRY.parent.mkdir(parents=True, exist_ok=True)

# Face alignment template (ArcFace R100 / R50 standard 112×112)
SRC = np.array(
    [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
     [41.5493, 92.3655], [70.7299, 92.2041]],
    dtype=np.float32,
)


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class StoredUser:
    name: str
    embeddings: list[list[float]]
    enrolled_at: float
    samples: int

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "embeddings": self.embeddings,
            "enrolled_at": self.enrolled_at,
            "samples": self.samples,
        }

    @staticmethod
    def from_dict(d: dict) -> "StoredUser":
        return StoredUser(
            name=d["name"],
            embeddings=[list(map(float, e)) for e in d["embeddings"]],
            enrolled_at=float(d.get("enrolled_at", 0.0)),
            samples=int(d.get("samples", len(d["embeddings"]))),
        )


class Registry:
    """Tiny JSON-backed user store; cos-sim matching via BruteForceIndex."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.users: dict[str, StoredUser] = {}
        self._load()
        self._rebuild_index()

    # ---- I/O ----
    def _load(self) -> None:
        if not self.path.is_file():
            return
        with self.path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        for entry in data.get("users", []):
            u = StoredUser.from_dict(entry)
            self.users[u.name] = u

    def save(self) -> None:
        payload = {
            "version": 1,
            "users": [u.to_dict() for u in self.users.values()],
        }
        with self.path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)

    # ---- ops ----
    def list_names(self) -> list[str]:
        return sorted(self.users.keys())

    def add_embeddings(self, name: str, embeddings: list[NDArray[np.float32]]) -> None:
        if name in self.users:
            existing = self.users[name]
            existing.embeddings.extend(e.tolist() for e in embeddings)
            existing.samples = len(existing.embeddings)
            existing.enrolled_at = time.time()
        else:
            self.users[name] = StoredUser(
                name=name,
                embeddings=[e.tolist() for e in embeddings],
                enrolled_at=time.time(),
                samples=len(embeddings),
            )
        self.save()
        self._rebuild_index()

    def delete(self, name: str) -> bool:
        if name in self.users:
            del self.users[name]
            self.save()
            self._rebuild_index()
            return True
        return False

    def _rebuild_index(self) -> None:
        self._index = BruteForceIndex(dim=512)
        self._owner: list[str] = []
        all_embs: list[NDArray[np.float32]] = []
        all_ids: list[int] = []
        for u in self.users.values():
            for e in u.embeddings:
                all_ids.append(len(self._owner))
                self._owner.append(u.name)
                all_embs.append(np.asarray(e, dtype=np.float32))
        if all_embs:
            arr = np.stack(all_embs, axis=0)
            self._index.add(all_ids, arr)

    def match(self, embedding: NDArray[np.float32], threshold: float = 0.45) -> tuple[str, float]:
        if self._index.ntotal() == 0:
            return "?", 0.0
        hits = self._index.search(embedding, top_k=1)
        if not hits:
            return "?", 0.0
        owner, sim = self._owner[hits[0].id], float(hits[0].score)
        if sim < threshold:
            return "?", sim
        return owner, sim


# ---------------------------------------------------------------------------
# Wrappers around the project models
# ---------------------------------------------------------------------------
class SCRFD:
    """Thin ONNX-Runtime wrapper for the insightface det_10g graph.

    Output layout (per the actual buffalo_l model):
        9 outputs in interleaved groups of 3 (score, bbox, kps) at strides
        8, 16, 32.  Each output is a 2D tensor of shape (N_anchors, C)
        where C is 1 / 4 / 10 respectively.  The first axis already covers
        the whole anchor grid; we just reshape using the stride.
    """

    def __init__(self, path: Path, input_size: int = 320, conf_thresh: float = 0.5) -> None:
        import onnxruntime as ort
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = max(1, os.cpu_count() or 1)
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        if hasattr(opts, "enable_mem_arena"):
            opts.enable_mem_arena = True
        # Disable ORT logging entirely.
        try:
            opts.log_severity_level = 3
        except Exception:
            pass
        self.session = ort.InferenceSession(
            str(path), sess_options=opts, providers=["CPUExecutionProvider"],
            # disable telemetry + warning spam
        )
        self.input_name = self.session.get_inputs()[0].name
        self.input_size = input_size
        self.conf = conf_thresh
        self.min_face = 60  # reject bboxes smaller than this many pixels
        self.feat_stride_fpn = [8, 16, 32]

    def _preprocess(self, img: NDArray[np.uint8]) -> tuple[NDArray[np.float32], float]:
        h0, w0 = img.shape[:2]
        scale = self.input_size / max(h0, w0)
        resized = cv2.resize(img, (int(w0 * scale), int(h0 * scale)))
        pad_h = self.input_size - resized.shape[0]
        pad_w = self.input_size - resized.shape[1]
        padded = cv2.copyMakeBorder(
            resized, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=(0, 0, 0)
        )
        blob = padded.astype(np.float32)
        blob = (blob - 127.5) / 128.0
        blob = blob.transpose(2, 0, 1)[None]  # (1, 3, H, W)
        return blob, scale

    def _anchor_centers(self, h: int, w: int, stride: int) -> NDArray[np.float32]:
        # 2 anchors per spatial cell, flattened in (y, x, anchor) order.
        xs = (np.arange(w) + 0.5) * stride
        ys = (np.arange(h) + 0.5) * stride
        xv, yv = np.meshgrid(xs, ys)
        grid = np.stack(
            [xv.flatten().repeat(2), yv.flatten().repeat(2)], axis=1
        ).astype(np.float32)
        return grid

    def detect(self, img: NDArray[np.uint8]) -> list[dict]:
        blob, scale = self._preprocess(img)
        outs = self.session.run(None, {self.input_name: blob})
        fmc = len(self.feat_stride_fpn)
        boxes, kpss = [], []
        for idx, stride in enumerate(self.feat_stride_fpn):
            score = outs[idx]                  # (N, 1)
            bbox = outs[idx + fmc]             # (N, 4)  in units of `stride`
            kps = outs[idx + fmc * 2]          # (N, 10) in units of `stride`
            score = score.reshape(-1)
            bbox = bbox.reshape(-1, 4)
            kps = kps.reshape(-1, 10)
            h = (blob.shape[2]) // stride
            w = (blob.shape[3]) // stride
            anchors = self._anchor_centers(h, w, stride)
            inds = np.where(score >= self.conf)[0]
            for i in inds:
                ax, ay = anchors[i]
                s = float(score[i])
                dx, dy, dw, dh = bbox[i] * stride  # bring to input pixel space
                x1 = (ax - dx) / scale
                y1 = (ay - dy) / scale
                x2 = (ax + dw) / scale
                y2 = (ay + dh) / scale
                pts = (kps[i] * stride).reshape(5, 2)  # kps in input pixel space
                pts[:, 0] += ax
                pts[:, 1] += ay
                pts /= scale
                boxes.append([x1, y1, x2, y2, s])
                kpss.append(pts)
        if not boxes:
            return []
        # drop tiny spurious detections (false positives look like 4x4 pixels)
        keep = [
            (b, k) for b, k in zip(boxes, kpss)
            if (b[2] - b[0]) >= self.min_face and (b[3] - b[1]) >= self.min_face
        ]
        if not keep:
            return []
        boxes, kpss = zip(*keep)
        boxes, kpss = list(boxes), list(kpss)
        # class-agnostic NMS
        order = np.argsort([-b[4] for b in boxes]).tolist()
        keep = []
        while order:
            i = order.pop(0)
            keep.append(i)
            rest = []
            for j in order:
                if self._iou(boxes[i], boxes[j]) < 0.4:
                    rest.append(j)
            order = rest
        return [{"bbox": boxes[k][:4], "score": boxes[k][4], "kps": kpss[k]} for k in keep]

    @staticmethod
    def _iou(a: list[float], b: list[float]) -> float:
        x1, y1 = max(a[0], b[0]), max(a[1], b[1])
        x2, y2 = min(a[2], b[2]), min(a[3], b[3])
        inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        area_a = (a[2] - a[0]) * (a[3] - a[1])
        area_b = (b[2] - b[0]) * (b[3] - b[1])
        return inter / (area_a + area_b - inter + 1e-9)


class ArcFace:
    """ONNX-Runtime wrapper for the insightface w600k_r50 graph (input 112×112)."""

    def __init__(self, path: Path) -> None:
        import onnxruntime as ort
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = max(1, os.cpu_count() or 1)
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        if hasattr(opts, "enable_mem_arena"):
            opts.enable_mem_arena = True
        try:
            opts.log_severity_level = 3
        except Exception:
            pass
        self.session = ort.InferenceSession(
            str(path), sess_options=opts, providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name

    @staticmethod
    def align(img: NDArray[np.uint8], kps: NDArray[np.float32], size: int = 112) -> NDArray[np.float32]:
        dst = SRC * (size / 112.0)
        M, _ = cv2.estimateAffinePartial2D(kps.astype(np.float32), dst, method=cv2.LMEDS)
        if M is None:
            return np.zeros((3, size, size), dtype=np.float32)
        warped = cv2.warpAffine(img, M, (size, size), borderValue=0.0)
        blob = warped.astype(np.float32)
        blob = (blob - 127.5) / 128.0
        return blob.transpose(2, 0, 1)

    def embed(self, img: NDArray[np.uint8], kps: NDArray[np.float32]) -> NDArray[np.float32]:
        face = self.align(img, kps, 112)
        out = self.session.run(None, {self.input_name: face[None]})[0][0]
        n = float(np.linalg.norm(out)) + 1e-9
        return (out / n).astype(np.float32)


# ---------------------------------------------------------------------------
# Frame capture helpers
# ---------------------------------------------------------------------------
def grab(cap: cv2.VideoCapture) -> tuple[bool, NDArray[np.uint8]]:
    ok, frame = cap.read()
    if not ok:
        return False, np.zeros((1, 1, 3), dtype=np.uint8)
    return True, cv2.flip(frame, 1)  # mirror for natural interaction


def capture_window(
    cap: cv2.VideoCapture,
    detector: SCRFD,
    recognizer: ArcFace,
    n_seconds: float = 3.0,
    target_fps: float = 10.0,
    on_progress=None,
) -> tuple[list[NDArray[np.float32]], list[NDArray[np.uint8]], list[dict]]:
    """Capture `n_seconds` of frames and return embeddings + face crops + face meta."""
    n_target = int(n_seconds * target_fps)
    period = 1.0 / target_fps
    embeddings, crops, faces_meta = [], [], []
    last = 0.0
    start = time.time()
    while len(embeddings) < n_target:
        now = time.time()
        if now - last < period:
            time.sleep(0.001)
            continue
        last = now
        ok, frame = grab(cap)
        if not ok:
            break
        dets = detector.detect(frame)
        if not dets:
            cv2.putText(
                frame, "No face - look at the camera", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2,
            )
            cv2.imshow("V.I.S.I.O.N. PC Test", frame)
            cv2.waitKey(1)
            continue
        best = max(dets, key=lambda d: (d["bbox"][2] - d["bbox"][0]) * (d["bbox"][3] - d["bbox"][1]))
        x1, y1, x2, y2 = (int(v) for v in best["bbox"])
        crop = frame[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
        emb = recognizer.embed(frame, best["kps"])
        embeddings.append(emb)
        crops.append(crop if crop.size else np.zeros((112, 112, 3), dtype=np.uint8))
        faces_meta.append(best)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            frame, f"captured {len(embeddings)}/{n_target}  sim={0.0:.2f}", (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2,
        )
        cv2.putText(
            frame, f"elapsed {now - start:0.1f}s", (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2,
        )
        cv2.imshow("V.I.S.I.O.N. PC Test", frame)
        if on_progress is not None:
            on_progress(len(embeddings), n_target)
        cv2.waitKey(1)
    return embeddings, crops, faces_meta


def liveness_for_capture(
    cap: cv2.VideoCapture,
    detector: SCRFD,
    landmarks: LandmarkEngine,
    blink: BlinkDetector,
    eye: EyeTracker,
    head: HeadPoseEstimator,
    n_seconds: float = 3.0,
) -> dict:
    """Stream frames and run blink / eye / head liveness; returns summary stats."""
    n_target = int(n_seconds * 12)
    period = 1.0 / 12.0
    last = 0.0
    yaw_buf: list[float] = []
    pitch_buf: list[float] = []
    ear_buf: list[float] = []
    blink_count = 0
    last_blink = 0
    captured = 0
    while captured < n_target:
        now = time.time()
        if now - last < period:
            time.sleep(0.001)
            continue
        last = now
        ok, frame = grab(cap)
        if not ok:
            break
        dets = detector.detect(frame)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if not dets:
            cv2.putText(
                frame, "No face - look at the camera", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2,
            )
            cv2.imshow("V.I.S.I.O.N. PC Test", frame)
            cv2.waitKey(1)
            continue
        best = max(dets, key=lambda d: (d["bbox"][2] - d["bbox"][0]) * (d["bbox"][3] - d["bbox"][1]))
        lms = landmarks.extract_landmarks(rgb)
        if lms is not None:
            ear = blink.update(lms, int(now * 1000))
            eye.update(lms, int(now * 1000), has_refine=landmarks.refine)
            hp = head.estimate(lms, frame.shape[:2], int(now * 1000))
            yaw_buf.append(hp.yaw)
            pitch_buf.append(hp.pitch)
            ear_buf.append(ear)
            if blink.blink_count > last_blink:
                last_blink = blink.blink_count
            blink_count = blink.blink_count
        x1, y1, x2, y2 = (int(v) for v in best["bbox"])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            frame,
            f"blinks={blink_count}  mean_ear={np.mean(ear_buf) if ear_buf else 0:.2f}",
            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
        )
        if yaw_buf:
            cv2.putText(
                frame,
                f"yaw={np.mean(yaw_buf):.1f}  pitch={np.mean(pitch_buf):.1f}",
                (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
            )
        cv2.putText(
            frame, "Blink naturally for liveness", (20, 110),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2,
        )
        cv2.imshow("V.I.S.I.O.N. PC Test", frame)
        captured += 1
        cv2.waitKey(1)
    return {
        "blink_count": blink_count,
        "mean_ear": float(np.mean(ear_buf)) if ear_buf else 0.0,
        "mean_yaw": float(np.mean(np.abs(yaw_buf))) if yaw_buf else 0.0,
        "mean_pitch": float(np.mean(np.abs(pitch_buf))) if pitch_buf else 0.0,
    }


def run_passive_auth(
    cap: cv2.VideoCapture,
    detector,
    recognizer,
    passive: PassiveLivenessDetector,
    antispoof: AntiSpoofCheck,
    *,
    n_seconds: float = 3.0,
    target_fps: float = 10.0,
) -> tuple[PassiveLivenessResult, AntiSpoofResult, list[NDArray[np.float32]], int]:
    """No user interaction. Just sit still and look at the camera.

    Collects 3 s of frames; computes:
      - passive liveness score (motion + texture + color + edges + screen-refresh)
      - MiniFASNet / heuristic anti-spoof score
      - 512-d ArcFace embeddings for matching

    Returns (passive_res, spoof_res, embeddings, frames_captured).
    """
    passive.reset()
    period = 1.0 / target_fps
    last_t = 0.0
    start = time.time()
    embeddings: list[NDArray[np.float32]] = []
    last_face_bbox: tuple[float, float, float, float] | None = None
    last_frame: NDArray[np.uint8] | None = None
    frames_captured = 0
    while time.time() - start < n_seconds:
        now = time.time()
        if now - last_t < period:
            time.sleep(0.001)
            continue
        last_t = now
        ok, frame = grab(cap)
        if not ok:
            break
        dets = detector.detect(frame)
        if not dets:
            cv2.putText(
                frame, "No face - look at the camera", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2,
            )
            cv2.putText(
                frame, f"passive liveness... {n_seconds - (now - start):.1f}s",
                (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2,
            )
            cv2.imshow("V.I.S.I.O.N. PC Test", frame)
            cv2.waitKey(1)
            continue
        best = max(dets, key=lambda d: (d["bbox"][2] - d["bbox"][0]) * (d["bbox"][3] - d["bbox"][1]))
        x1, y1, x2, y2 = (int(v) for v in best["bbox"])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        passive.update(frame, best["bbox"], t=now)
        emb = recognizer.embed(frame, best["kps"])
        embeddings.append(emb)
        last_face_bbox = (x1, y1, x2, y2)
        last_frame = frame
        frames_captured += 1
        cv2.putText(
            frame, f"passive liveness... {n_seconds - (now - start):.1f}s",
            (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2,
        )
        cv2.imshow("V.I.S.I.O.N. PC Test", frame)
        cv2.waitKey(1)
    # final scores
    passive_res = passive.result()
    if last_face_bbox is not None and last_frame is not None:
        spoof_res = antispoof.predict(last_frame, last_face_bbox)
    else:
        spoof_res = AntiSpoofResult(0.0, False, False, antispoof.threshold, antispoof._source)
    return passive_res, spoof_res, embeddings, frames_captured


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
HELP = (
    "V.I.S.I.O.N. PC test\n"
    "  r = register        \n"
    "  a = authenticate    (passive, no user interaction)\n"
    "  A = authenticate    (STRICT: random challenge, head/eye)\n"
    "  l = list users   d = delete   q = quit\n"
    "Look straight at the camera.  Press a key in the video window."
)


def main() -> int:
    setup_logging()
    print(HELP)
    if not DET_PATH.is_file() or not REC_PATH.is_file():
        print(f"ERROR: missing weights.  expected:\n  {DET_PATH}\n  {REC_PATH}", file=sys.stderr)
        return 1
    print("[load] SCRFD  det_10g.onnx ...", flush=True)
    detector = SCRFD(DET_PATH)
    print("[load] ArcFace  w600k_r50.onnx ...", flush=True)
    recognizer = ArcFace(REC_PATH)
    print("[load] MediaPipe FaceMesh ...", flush=True)
    landmarks = LandmarkEngine()
    registry = Registry(REGISTRY)
    print(f"[load] Registry: {len(registry.users)} user(s)  ->  {REGISTRY}")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: could not open webcam (index 0).", file=sys.stderr)
        return 1
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cv2.namedWindow("V.I.S.I.O.N. PC Test", cv2.WINDOW_NORMAL)
    print("[ready] window opened.  Press 'h' for help inside the window.")
    last_status = ""
    blink = BlinkDetector()
    eye = EyeTracker()
    head = HeadPoseEstimator()
    antispoof = AntiSpoofCheck(models_dir=MODELS, threshold=0.50)
    print(f"[load] AntiSpoofCheck: source={antispoof._source} threshold={antispoof.threshold}")
    if antispoof._source == "heuristic":
        print("[note] Using Laplacian-variance heuristic for anti-spoof.")
        print("       (Real MiniFASNet V2 weights are not publicly hosted;")
        print("        drop 2.7_80x80 + 4_0_80x80 ONNX into models/buffalo_l/")
        print("        to enable the real model.)")
    passive = PassiveLivenessDetector(threshold=0.45)
    print(f"[load] PassiveLivenessDetector: threshold={passive.threshold}")
    try:
        while True:
            ok, frame = grab(cap)
            if not ok:
                break
            dets = detector.detect(frame)
            annotated = frame.copy()
            cv2.putText(annotated, last_status, (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(annotated,
                        "r register  a auth (passive)  A auth (strict)  q quit",
                        (20, annotated.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
            for d in dets:
                x1, y1, x2, y2 = (int(v) for v in d["bbox"])
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(annotated, f"{d['score']:.2f}", (x1, y1 - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.imshow("V.I.S.I.O.N. PC Test", annotated)
            key = cv2.waitKey(30) & 0xFF

            if key == ord("q"):
                break
            elif key == ord("h"):
                print(HELP)
                last_status = "help printed to console"
            elif key == ord("l"):
                names = registry.list_names()
                print(f"[list] {len(names)} user(s): {names}")
                last_status = f"users: {', '.join(names) if names else '(none)'}"
            elif key == ord("r"):
                name = input("Enter name to register: ").strip()
                if not name:
                    last_status = "register: empty name"
                else:
                    print(f"[register] capturing 3 s for '{name}' ... look at the camera")
                    embs, _, _ = capture_window(
                        cap, detector, recognizer, n_seconds=3.0, target_fps=8.0,
                    )
                    if len(embs) < 8:
                        print(f"[register] aborted: only {len(embs)} frames captured")
                        last_status = f"register failed for {name}"
                    else:
                        registry.add_embeddings(name, embs)
                        print(f"[register] stored {len(embs)} embeddings for '{name}'")
                        last_status = f"registered {name} ({len(embs)} samples)"
            elif key == ord("d"):
                name = input("Delete which user? ").strip()
                if registry.delete(name):
                    print(f"[delete] removed '{name}'")
                    last_status = f"deleted {name}"
                else:
                    print(f"[delete] no such user: {name}")
                    last_status = "delete: not found"
            elif key == ord("a"):
                if not registry.users:
                    print("[auth] no users registered yet - press 'r' first")
                    last_status = "no users registered"
                    continue
                print("[auth] PASSIVE liveness — just look at the camera (3 s) ...")
                passive_res, spoof_res, embs, frames = run_passive_auth(
                    cap, detector, recognizer, passive, antispoof,
                    n_seconds=3.0, target_fps=10.0,
                )
                if not embs:
                    print("[auth] aborted: no face in capture")
                    last_status = "auth: no face"
                    continue
                agg = np.mean(np.stack(embs, axis=0), axis=0)
                agg /= (np.linalg.norm(agg) + 1e-9)
                name, sim = registry.match(agg, threshold=0.45)
                reasons = []
                if not passive_res.passed:
                    reasons.append(
                        f"passive_liveness:score={passive_res.score:.2f}<{passive_res.threshold}"
                    )
                if not spoof_res.passed:
                    reasons.append(
                        f"spoof:score={spoof_res.score:.2f}<{spoof_res.threshold}({spoof_res.source})"
                    )
                liveness_pass = passive_res.passed and spoof_res.passed
                decision = "ACCEPT" if (name != "?" and liveness_pass) else "REJECT"
                print(
                    f"[auth] decision={decision} user={name} sim={sim:.3f}\n"
                    f"       passive_liveness: score={passive_res.score:.2f} (thresh={passive_res.threshold}) -> {passive_res.passed}\n"
                    f"                    motion={passive_res.motion_score:.2f} texture={passive_res.texture_score:.2f} "
                    f"color={passive_res.color_score:.2f} edges={passive_res.edge_score:.2f} refresh={passive_res.refresh_score:.2f} n={passive_res.frames_sampled}\n"
                    f"       antispoof:        score={spoof_res.score:.2f} src={spoof_res.source} -> {spoof_res.passed}\n"
                    f"       notes:            {'; '.join(passive_res.notes) if passive_res.notes else 'none'}\n"
                    f"       reasons:          {'; '.join(reasons) if reasons else 'none'}"
                )
                last_status = f"{decision} {name} sim={sim:.2f}"
            elif key == ord("A"):
                if not registry.users:
                    print("[auth] no users registered yet - press 'r' first")
                    last_status = "no users registered"
                    continue
                print("[auth] running anti-replay liveness flow (challenge + motion + spoof) ...")
                # reset blink history so this run starts fresh
                blink.reset()
                head.reset()
                challenge_res, motion_res, asp_res, embs, blink_count, mean_ear, _, _, _ = (
                    run_auth_liveness(
                        cap, detector, recognizer, landmarks, blink, head, antispoof,
                        challenge_seconds=3.0,
                        capture_seconds=1.5,
                        target_fps=12.0,
                    )
                )
                if not embs:
                    print("[auth] aborted: no face in capture")
                    last_status = "auth: no face"
                    continue
                agg = np.mean(np.stack(embs, axis=0), axis=0)
                agg /= (np.linalg.norm(agg) + 1e-9)
                name, sim = registry.match(agg, threshold=0.45)
                # ---- gate on all three layers ----
                reasons = []
                if not challenge_res.passed:
                    reasons.append(f"challenge:{challenge_res.detail}")
                if not motion_res.passes:
                    reasons.append(
                        f"motion:mean={motion_res.mean_px_jitter:.2f}px<thresh={motion_res.threshold_px}"
                    )
                if not asp_res.passed:
                    reasons.append(f"spoof:score={asp_res.score:.2f}<{asp_res.threshold}({asp_res.source})")
                liveness_pass = challenge_res.passed and motion_res.passes and asp_res.passed
                decision = "ACCEPT" if (name != "?" and liveness_pass) else "REJECT"
                print(
                    f"[auth] decision={decision} user={name} sim={sim:.3f}\n"
                    f"       challenge: {challenge_res.text} -> {challenge_res.passed} ({challenge_res.detail})\n"
                    f"       motion:    mean={motion_res.mean_px_jitter:.2f}px (n={motion_res.frames_sampled}, thresh={motion_res.threshold_px}) -> {motion_res.passes}\n"
                    f"       antispoof: score={asp_res.score:.2f} thresh={asp_res.threshold} src={asp_res.source} -> {asp_res.passed}\n"
                    f"       reasons:   {'; '.join(reasons) if reasons else 'none'}"
                )
                last_status = f"{decision} {name} sim={sim:.2f} ch={'P' if challenge_res.passed else 'F'}"
    finally:
        cap.release()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
