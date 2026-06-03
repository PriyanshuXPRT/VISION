"""
ONNX export dispatcher — run all four export scripts with one command.
"""
from __future__ import annotations

from pathlib import Path

import typer

from vision.core.logging import logger, setup_logging
from vision.onnx.exporters.export_arcface import export_arcface
from vision.onnx.exporters.export_minifasnet import export_minifasnet
from vision.onnx.exporters.export_scrfd import export_scrfd
from vision.onnx.exporters.export_temporal import export_temporal

app = typer.Typer(add_completion=False)


@app.command()
def all(
    out_dir: Path = typer.Option(Path("onnx_models"), "--out-dir"),
    opset: int = typer.Option(12, "--opset"),
) -> None:
    """Export SCRFD + ArcFace + MiniFASNet + Temporal to `out_dir`."""
    setup_logging()
    cfg = [
        ("detection/scrfd_10g_bnkps.onnx", export_scrfd, {}),
        ("recognition/arcface_r100.onnx", export_arcface, {}),
        ("antispoof/minifasnet_v2.onnx", export_minifasnet, {}),
        ("temporal/lstm_v1.onnx", export_temporal, {}),
    ]
    for rel, fn, extra in cfg:
        out = out_dir / rel
        weights = Path("models") / rel.replace(".onnx", ".pth")
        fn(weights, out, opset=opset, **extra)
        logger.info("✓ {}", out)


def main() -> int:
    app()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
