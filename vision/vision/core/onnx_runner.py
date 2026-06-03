"""Reusable ONNX Runtime session wrapper with sane defaults."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
import onnxruntime as ort

from vision.core.exceptions import ModelLoadError, ModelNotFoundError
from vision.core.logging import logger


class OnnxRunner:
    """Lightweight ONNX Runtime wrapper.

    Centralises provider selection, input validation, and warm-up.
    """

    def __init__(
        self,
        model_path: str | Path,
        *,
        providers: Sequence[str] | None = None,
        intra_op_threads: int = 0,
        graph_optimization_level: ort.GraphOptimizationLevel | None = None,
    ) -> None:
        self.model_path = Path(model_path)
        if not self.model_path.is_file():
            raise ModelNotFoundError(f"ONNX model not found: {self.model_path}")

        opts = ort.SessionOptions()
        if graph_optimization_level is not None:
            opts.graph_optimization_level = graph_optimization_level
        if intra_op_threads > 0:
            opts.intra_op_num_threads = intra_op_threads
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        if hasattr(opts, "enable_mem_arena"):
            opts.enable_mem_arena = True

        chosen = list(providers) if providers else ort.get_available_providers()
        # Prefer CoreML on iOS, NNAPI/CUDA elsewhere; fall back to CPU.
        preferred = self._order_providers(chosen)

        try:
            self.session: ort.InferenceSession = ort.InferenceSession(
                str(self.model_path),
                sess_options=opts,
                providers=preferred,
            )
        except Exception as exc:  # noqa: BLE001
            raise ModelLoadError(f"Failed to load {self.model_path}: {exc}") from exc

        self.input_meta = self.session.get_inputs()
        self.output_meta = self.session.get_outputs()
        logger.debug(
            "Loaded ONNX model: {} providers={} inputs={} outputs={}",
            self.model_path.name,
            self.session.get_providers(),
            [m.name for m in self.input_meta],
            [m.name for m in self.output_meta],
        )

    @staticmethod
    def _order_providers(available: Sequence[str]) -> list[str]:
        rank = ["TensorrtExecutionProvider", "CUDAExecutionProvider",
                "CoreMLExecutionProvider", "NNAPIExecutionProvider",
                "DmlExecutionProvider", "CPUExecutionProvider"]
        ordered = [p for p in rank if p in available]
        return ordered or ["CPUExecutionProvider"]

    # ---- introspection ----
    @property
    def input_names(self) -> list[str]:
        return [m.name for m in self.input_meta]

    @property
    def output_names(self) -> list[str]:
        return [m.name for m in self.output_meta]

    def input_shape(self, name: str) -> tuple[int, ...]:
        for m in self.input_meta:
            if m.name == name:
                return tuple(m.shape)  # type: ignore[return-value]
        raise KeyError(name)

    # ---- inference ----
    def run(
        self,
        feeds: dict[str, np.ndarray],
        output_names: Sequence[str] | None = None,
    ) -> list[np.ndarray]:
        """Run inference. All inputs must be float32/float16/int32/int64."""
        outs = output_names or self.output_names
        for name, arr in feeds.items():
            if name not in self.input_names:
                raise KeyError(f"Unknown input '{name}' for {self.model_path.name}")
            if not isinstance(arr, np.ndarray):
                raise TypeError(f"Input '{name}' must be ndarray, got {type(arr)}")
        return self.session.run(list(outs), feeds)

    def warmup(self, sample: dict[str, np.ndarray] | None = None, n: int = 2) -> None:
        """Run a few dummy iterations to stabilise latency."""
        if sample is None:
            sample = self._dummy_sample()
        for _ in range(n):
            self.run(sample)
        logger.debug("Warmed up {} ({} iters)", self.model_path.name, n)

    def _dummy_sample(self) -> dict[str, np.ndarray]:
        feeds: dict[str, np.ndarray] = {}
        for meta in self.input_meta:
            shape = [1 if isinstance(d, str) or d is None else int(d) for d in meta.shape]
            dtype = self._onnx_type_to_numpy(meta.type)
            feeds[meta.name] = np.zeros(shape, dtype=dtype)
        return feeds

    @staticmethod
    def _onnx_type_to_numpy(onnx_type: str) -> Any:
        t = onnx_type.lower()
        if "float16" in t:
            return np.float16
        if "float" in t:
            return np.float32
        if "int64" in t:
            return np.int64
        if "int32" in t or "int" in t:
            return np.int32
        if "bool" in t:
            return np.bool_
        return np.float32
