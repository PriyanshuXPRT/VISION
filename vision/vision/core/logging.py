"""
Logging setup. Uses loguru with a console sink and a rotating file sink.
"""
from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger as _logger

from vision.config import settings


def setup_logging() -> None:
    """Configure loguru sinks. Idempotent."""
    _logger.remove()
    _logger.add(
        sys.stderr,
        level=settings.log_level,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        ),
        backtrace=True,
        diagnose=settings.env == "development",
    )
    log_file: Path = settings.log_dir / "vision.log"
    _logger.add(
        log_file,
        level=settings.log_level,
        rotation="50 MB",
        retention="30 days",
        compression="zip",
        enqueue=True,
        backtrace=True,
        diagnose=False,
    )
    _logger.info("Logging initialised · env={} · level={}", settings.env, settings.log_level)


# Module-level logger used everywhere
logger = _logger
