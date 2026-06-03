"""ONNX graph validation utilities."""
from __future__ import annotations

from pathlib import Path

import onnx
from onnx import checker

from vision.core.exceptions import ModelLoadError
from vision.core.logging import logger


def validate(path: Path) -> bool:
    """Run the ONNX checker on a model. Returns True on success."""
    if not path.is_file():
        raise ModelLoadError(f"Model not found: {path}")
    model = onnx.load(str(path))
    try:
        checker.check_model(model, full_check=True)
    except Exception as exc:  # noqa: BLE001
        raise ModelLoadError(f"Validation failed for {path}: {exc}") from exc
    n_params = sum(t.data_size for t in model.graph.initializer) if model.graph.initializer else 0
    logger.info("✓ {} (initializers ≈ {:.1f} MB)", path.name, n_params / 1e6)
    return True


def simplify(path: Path, out: Path | None = None) -> Path:
    """Run onnx-simplifier on a model. Useful before quantisation."""
    import onnxsim  # type: ignore

    out = out or path
    model = onnx.load(str(path))
    simplified, ok = onnxsim.simplify(model)
    if not ok:
        logger.warning("onnxsim could not simplify {}", path)
        return path
    onnx.save(simplified, str(out))
    logger.info("Simplified → {}", out)
    return out
