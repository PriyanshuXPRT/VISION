"""Benchmark a single attack: printed photos held up to a webcam."""
from __future__ import annotations

from pathlib import Path

from tests.security.benchmark_liveness import run


def main() -> int:
    from vision.core.pipeline import build_pipeline
    pipeline = build_pipeline()
    report = run(Path("datasets/security/printed_photo"), pipeline)
    print(report["overall"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
