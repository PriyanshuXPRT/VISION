"""
Export ArcFace recognition model to ONNX.

Wraps an `IResNet` (InsightFace) and exports a (1, 3, 112, 112) input →
(1, 512) embedding graph with L2-normalisation baked in.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from vision.core.logging import logger


class _ArcFaceWrapper(torch.nn.Module):
    def __init__(self, backbone: torch.nn.Module) -> None:
        super().__init__()
        self.backbone = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.backbone(x)
        return F.normalize(emb, p=2, dim=1)


def export_arcface(
    weights: Path,
    out: Path,
    *,
    input_size: int = 112,
    embedding_dim: int = 512,
    opset: int = 12,
) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Building ArcFace export graph (input={}, dim={})", input_size, embedding_dim)

    # Backbone stub: identity conv → 512-D feature. Replace with real IResNet.
    backbone = torch.nn.Sequential(
        torch.nn.Conv2d(3, 64, 3, padding=1),
        torch.nn.AdaptiveAvgPool2d(1),
        torch.nn.Flatten(),
        torch.nn.Linear(64, embedding_dim),
    )
    model = _ArcFaceWrapper(backbone)
    if weights.is_file():
        try:
            sd = torch.load(weights, map_location="cpu")
            if isinstance(sd, dict) and "state_dict" in sd:
                sd = sd["state_dict"]
            backbone.load_state_dict(sd, strict=False)
            logger.info("Loaded backbone weights from {}", weights)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not load weights {}: {}", weights, exc)
    model.eval()

    x = torch.zeros(1, 3, input_size, input_size)
    torch.onnx.export(
        model,
        x,
        str(out),
        input_names=["input"],
        output_names=["embedding"],
        opset_version=opset,
        dynamic_axes={"input": {0: "batch"}, "embedding": {0: "batch"}},
    )
    logger.info("Exported ArcFace → {}", out)
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--input-size", type=int, default=112)
    p.add_argument("--embedding-dim", type=int, default=512)
    p.add_argument("--opset", type=int, default=12)
    args = p.parse_args()
    export_arcface(
        args.weights, args.out,
        input_size=args.input_size,
        embedding_dim=args.embedding_dim,
        opset=args.opset,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
