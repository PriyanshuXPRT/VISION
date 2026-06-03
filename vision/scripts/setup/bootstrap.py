"""Setup helpers — install deps, create dirs, etc."""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from vision.config import settings
from vision.core.logging import logger, setup_logging


def main() -> int:
    setup_logging()
    settings.ensure_dirs()
    logger.info("Project root: {}", settings.models_dir.parent)
    # Pull the Python deps
    if shutil.which("pip"):
        subprocess.run(["pip", "install", "-r", "requirements.txt"], check=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
