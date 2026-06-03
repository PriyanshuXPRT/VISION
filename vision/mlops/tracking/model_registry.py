"""
Model registry — a tiny JSON-based manifest of available ONNX models and
their hashes, used to validate the integrity of the model bundle shipped
with the Android app.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from vision.core.logging import logger


@dataclass(slots=True)
class ModelEntry:
    name: str
    path: str
    version: str
    size_bytes: int
    sha256: str
    created_at: str
    notes: str = ""


def sha256_file(p: Path, *, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def build_registry(root: Path, out: Path) -> list[ModelEntry]:
    """Scan `root` for `.onnx` files, compute SHA-256, write a JSON manifest."""
    entries: list[ModelEntry] = []
    for p in sorted(root.rglob("*.onnx")):
        try:
            sha = sha256_file(p)
        except OSError as exc:
            logger.warning("Skipping {}: {}", p, exc)
            continue
        entry = ModelEntry(
            name=p.stem,
            path=str(p.relative_to(root.parent)),
            version=_read_version(p),
            size_bytes=p.stat().st_size,
            sha256=sha,
            created_at=datetime.utcnow().isoformat(timespec="seconds"),
        )
        entries.append(entry)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps([asdict(e) for e in entries], indent=2))
    logger.info("Wrote registry with {} models → {}", len(entries), out)
    return entries


def _read_version(p: Path) -> str:
    """Extract a version token from a sidecar `<model>.version` file if present."""
    side = p.with_suffix(p.suffix + ".version")
    if side.is_file():
        return side.read_text().strip()
    return "0.1.0"


def verify(root: Path, manifest: Path) -> bool:
    """Re-check all hashes against the manifest. Returns True if every model matches."""
    data = json.loads(manifest.read_text())
    ok = True
    for entry in data:
        path = root / entry["path"]
        if not path.is_file():
            logger.error("Missing model: {}", path)
            ok = False
            continue
        sha = sha256_file(path)
        if sha != entry["sha256"]:
            logger.error("Hash mismatch for {}: {} != {}", path, sha, entry["sha256"])
            ok = False
    return ok
