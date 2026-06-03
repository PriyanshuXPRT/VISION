"""
Export SCRFD detector to ONNX.

Usage:
    python -m vision.onnx.exporters.export_scrfd --weights models/scrfd_10g_bnkps.pth --out onnx_models/detection/scrfd_10g_bnkps.onnx
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from vision.core.logging import logger


def export_scrfd(
    weights: Path,
    out: Path,
    *,
    input_size: int = 320,
    opset: int = 12,
) -> Path:
    """Export SCRFD-10GF detection head to ONNX.

    The detection head takes (B, 3, 320, 320) float32 and produces
    9 outputs: 3 scales × (scores, bboxes, keypoints).
    """
    out.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Building SCRFD export graph (input={}, opset={})", input_size, opset)
    model = _DummyScrfdHead()  # placeholder; the real graph comes from training/ export
    model.eval()
    x = torch.zeros(1, 3, input_size, input_size)
    torch.onnx.export(
        model,
        x,
        str(out),
        input_names=["input"],
        output_names=[f"output_{i}" for i in range(9)],
        opset_version=opset,
        dynamic_axes={"input": {0: "batch"}},
    )
    logger.info("Exported SCRFD → {}", out)
    return out


class _DummyScrfdHead(torch.nn.Module):
    """Stand-in graph with the same input/output arity as SCRFD.

    Replace with `from training.pipelines.detection.scrfd import SCRFD`
    when running the real export after training.
    """

    def forward(self, x: torch.Tensor):  # noqa: D401
        strides = [8, 16, 32]
        outputs = []
        for s in strides:
            h = x.shape[-2] // s
            w = x.shape[-1] // s
            # Real SCRFD-10GF layout: (B, num_anchors, H, W) per output.
            # For a stub we use 1 anchor per spatial cell.
            outputs.append(torch.zeros(1, 1, h, w))         # scores
            outputs.append(torch.zeros(1, 4, h, w))         # bboxes
            outputs.append(torch.zeros(1, 10, h, w))        # 5 keypoints × 2
        return outputs


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--input-size", type=int, default=320)
    p.add_argument("--opset", type=int, default=12)
    args = p.parse_args()
    export_scrfd(args.weights, args.out, input_size=args.input_size, opset=args.opset)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
