"""
ONNX quantisation utilities.

FP16: convert weights to half precision (memory ~ 1/2, accuracy ~ identical).
INT8 (static): calibrate on a small dataset and quantise weights/activations
     (memory ~ 1/4, accuracy hit ~ 1–2% on face tasks; usually fine).

Usage:
    python -m vision.onnx.quantization.quantize --in model.onnx --out model.int8.onnx --int8
    python -m vision.onnx.quantization.quantize --in model.onnx --out model.fp16.onnx --fp16
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
from numpy.typing import NDArray

from vision.core.logging import logger


# -----------------------------------------------------------------------------
# FP16
# -----------------------------------------------------------------------------
def to_fp16(src: Path, dst: Path) -> Path:
    import onnx
    from onnxconverter_common import float16  # type: ignore

    dst.parent.mkdir(parents=True, exist_ok=True)
    model = onnx.load(str(src))
    model_fp16 = float16.convert_float_to_float16(model, keep_io_types=True)
    onnx.save(model_fp16, str(dst))
    logger.info("FP16 → {} ({:.1f} MB)", dst, dst.stat().st_size / 1e6)
    return dst


# -----------------------------------------------------------------------------
# INT8 (static, using onnxruntime.quant)
# -----------------------------------------------------------------------------
def to_int8(
    src: Path,
    dst: Path,
    *,
    calibration_data: Iterable[NDArray[np.float32]] | None = None,
    input_name: str = "input",
) -> Path:
    from onnxruntime.quantization import (
        CalibrationDataReader,
        QuantType,
        quantize_static,
    )

    dst.parent.mkdir(parents=True, exist_ok=True)

    class _DataReader(CalibrationDataReader):
        def __init__(self, data: Iterable[NDArray[np.float32]] | None, name: str) -> None:
            self._iter: Iterator | None = iter(data) if data is not None else None
            self._name = name

        def get_next(self):
            if self._iter is None:
                return None
            try:
                x = next(self._iter)
            except StopIteration:
                return None
            return {self._name: x.astype(np.float32)}

        def rewind(self):
            if self._iter is not None and isinstance(self._iter, list):
                self._iter = iter(self._iter)

    reader = _DataReader(calibration_data, input_name)
    quantize_static(
        model_input=str(src),
        model_output=str(dst),
        calibration_data_reader=reader,
        quant_format=QuantType.QInt8,
        per_channel=True,
        reduce_range=False,
        weight_type=QuantType.QInt8,
    )
    logger.info("INT8 → {} ({:.1f} MB)", dst, dst.stat().st_size / 1e6)
    return dst


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="src", type=Path, required=True)
    p.add_argument("--out", dest="dst", type=Path, required=True)
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--int8", action="store_true")
    p.add_argument("--calib-n", type=int, default=64)
    p.add_argument("--input-name", type=str, default="input")
    p.add_argument("--input-shape", type=str, default="1,3,112,112")
    args = p.parse_args()

    if args.fp16:
        to_fp16(args.src, args.dst)
    elif args.int8:
        shape = tuple(int(s) for s in args.input_shape.split(","))
        rng = np.random.default_rng(0)
        data = [rng.standard_normal(shape).astype(np.float32) for _ in range(args.calib_n)]
        to_int8(args.src, args.dst, calibration_data=data, input_name=args.input_name)
    else:
        logger.error("Specify --fp16 or --int8")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
