"""Run all four ONNX exporters to populate onnx_models/."""
from __future__ import annotations

from pathlib import Path

from vision.core.logging import setup_logging
from vision.onnx.exporters.export_arcface import export_arcface
from vision.onnx.exporters.export_minifasnet import export_minifasnet
from vision.onnx.exporters.export_scrfd import export_scrfd
from vision.onnx.exporters.export_temporal import export_temporal


def main() -> int:
    setup_logging()
    out_dir = Path("onnx_models")
    cfg = [
        ("detection/scrfd_10g_bnkps.onnx", export_scrfd),
        ("recognition/arcface_r100.onnx", export_arcface),
        ("antispoof/minifasnet_v2.onnx", export_minifasnet),
        ("temporal/lstm_v1.onnx", export_temporal),
    ]
    for rel, fn in cfg:
        out = out_dir / rel
        weights = Path("models") / rel.replace(".onnx", ".pth")
        out.parent.mkdir(parents=True, exist_ok=True)
        weights.parent.mkdir(parents=True, exist_ok=True)
        fn(weights, out, opset=12)
        print(f"[ok] {out}  ({out.stat().st_size / 1024:.1f} KiB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
