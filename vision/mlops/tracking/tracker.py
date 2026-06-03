"""
Experiment tracking shim — wraps Weights & Biases (or MLflow) with a
no-op fallback so the training scripts can run in CI without credentials.
"""
from __future__ import annotations

import os
from typing import Any

from vision.core.logging import logger


class ExperimentTracker:
    def __init__(self, project: str, run_name: str, *, backend: str = "auto") -> None:
        self.project = project
        self.run_name = run_name
        self.backend = backend
        self._impl = self._init(backend)

    def _init(self, backend: str) -> Any:
        backend = backend.lower()
        if backend == "none" or not os.environ.get("VISION_TRACKING", "1") == "1":
            return _Noop()
        if backend in ("auto", "wandb") and os.environ.get("WANDB_API_KEY"):
            try:
                import wandb  # type: ignore
                wandb.init(project=self.project, name=self.run_name)
                return wandb
            except Exception as exc:  # noqa: BLE001
                logger.warning("wandb unavailable: {}", exc)
        if backend in ("auto", "mlflow"):
            try:
                import mlflow  # type: ignore
                mlflow.set_experiment(self.project)
                mlflow.start_run(run_name=self.run_name)
                return mlflow
            except Exception as exc:  # noqa: BLE001
                logger.warning("mlflow unavailable: {}", exc)
        return _Noop()

    def log(self, key: str, value: Any, step: int | None = None) -> None:
        if hasattr(self._impl, "log"):
            try:
                if step is None:
                    self._impl.log({key: value})
                else:
                    self._impl.log({key: value}, step=step)
            except Exception as exc:  # noqa: BLE001
                logger.debug("tracker.log failed: {}", exc)
        else:
            logger.info("track: {}={} step={}", key, value, step)

    def finish(self) -> None:
        if hasattr(self._impl, "finish"):
            try:
                self._impl.finish()
            except Exception:  # noqa: BLE001
                pass


class _Noop:
    def log(self, *_, **__) -> None: ...
    def finish(self) -> None: ...
