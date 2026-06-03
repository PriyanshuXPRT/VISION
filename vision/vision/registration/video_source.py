"""Video frame extraction helpers used during enrollment."""
from __future__ import annotations

from pathlib import Path
from typing import Iterator

import cv2
import numpy as np
from numpy.typing import NDArray


def iter_video_frames(
    source: str | Path | int,
    *,
    target_fps: float = 10.0,
    max_seconds: float = 3.0,
    rgb: bool = True,
) -> Iterator[tuple[int, NDArray[np.uint8]]]:
    """Yield (timestamp_ms, frame_rgb_uint8) tuples from a video file or camera index."""
    cap = cv2.VideoCapture(int(source) if isinstance(source, (int, str)) and str(source).isdigit() else str(source))
    if not cap.isOpened():
        raise IOError(f"Could not open video source: {source}")
    try:
        native_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        period_ms = int(1000.0 / max(1.0, target_fps))
        max_frames = int(max_seconds * target_fps)
        idx = 0
        ts_ms = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % max(1, int(round(native_fps / target_fps))) == 0:
                if rgb:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                yield ts_ms, frame
                ts_ms += period_ms
                if (idx // max(1, int(round(native_fps / target_fps)))) >= max_frames:
                    break
            idx += 1
    finally:
        cap.release()
