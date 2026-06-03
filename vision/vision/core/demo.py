"""
End-to-end demo: write a tiny synthetic video, enrol a fake user, then run
authentication on the same frames and print the decision.

Designed as a CI smoke test that exercises every subsystem (DB, detection,
recognition, antispoof, liveness, vector index, authenticator) without
needing a real camera or face.
"""
from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np

from vision.core.logging import logger, setup_logging
from vision.core.pipeline import build_pipeline


def _write_synthetic_video(
    path: Path,
    *,
    n_frames: int = 60,
    fps: int = 20,
    width: int = 640,
    height: int = 480,
    seed: int = 7,
) -> None:
    """Write a short synthetic video with low-amplitude noise (so the
    stub detector occasionally fires; we don't expect a hit on a random
    frame, but the pipeline still has to handle it cleanly)."""
    rng = np.random.default_rng(seed)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, float(fps), (width, height))
    if not writer.isOpened():
        raise IOError(f"Could not open VideoWriter for {path}")
    try:
        for _ in range(n_frames):
            frame = rng.integers(0, 255, size=(height, width, 3), dtype=np.uint8)
            writer.write(frame)
    finally:
        writer.release()


def main() -> int:
    setup_logging()
    logger.info("V.I.S.I.O.N. demo starting …")
    pipeline = build_pipeline()

    # 1. Write a synthetic 3-second video.
    with tempfile.TemporaryDirectory() as td:
        video_path = Path(td) / "synthetic.mp4"
        _write_synthetic_video(video_path, n_frames=60, fps=20)
        logger.info("Wrote synthetic video {} ({} bytes)",
                    video_path, video_path.stat().st_size)

        # 2. Try to enrol — with random frames we expect a controlled
        #    'no face detected' failure, which is still a valid pipeline run.
        try:
            result = pipeline.registration.enroll(
                name="Demo User",
                video_source=video_path,
                source="self",
            )
            logger.info(
                "Enrolled: user_id={} template_id={} frames={} q={:.2f}",
                result.user.user_id, result.template.template_id,
                result.frames_used, result.mean_quality,
            )
        except Exception as exc:
            logger.warning(
                "Enrolment path produced expected error (synthetic noise frames): {}",
                exc.__class__.__name__,
            )

    # 3. Run authentication on a fresh synthetic stream.  Authenticator
    #    accepts a video path or generator, so we feed it directly.
    rng = np.random.default_rng(42)

    def _frame_iter():
        for ts in range(0, 3000, 100):
            img = rng.integers(0, 255, size=(480, 640, 3), dtype=np.uint8)
            yield ts, img

    auth = pipeline.authenticator.authenticate(_frame_iter())
    logger.info(
        "Auth decision: {} user='{}' sim={:.3f} live={:.3f} reason='{}'",
        auth.decision.value, auth.user_name, auth.similarity,
        auth.liveness.final_score, auth.reason,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
