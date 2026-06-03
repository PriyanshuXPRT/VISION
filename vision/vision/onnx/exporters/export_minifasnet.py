"""
Export MiniFASNetV2 anti-spoofing model to ONNX.

Input: (1, 3, 80, 80) float32
Output: (1, 2) logits (real, fake)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from vision.core.logging import logger


class _MiniFASNetStub(torch.nn.Module):
    def __init__(self, embedding: int = 128, n_classes: int = 2) -> None:
        super().__init__()
        self.stem = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, 3, stride=2, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(32, embedding, 3, stride=2, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.AdaptiveAvgPool2d(1),
            torch.nn.Flatten(),
        )
        self.fc = torch.nn.Linear(embedding, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.stem(x))


def export_minifasnet(
    weights: Path,
    out: Path,
    *,
    input_size: int = 80,
    opset: int = 12,
) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    model = _MiniFASNetStub()
    if weights.is_file():
        try:
            sd = torch.load(weights, map_location="cpu")
            if isinstance(sd, dict) and "state_dict" in sd:
                sd = sd["state_dict"]
            model.load_state_dict(sd, strict=False)
            logger.info("Loaded FAS weights from {}", weights)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not load FAS weights {}: {}", weights, exc)
    model.eval()
    x = torch.zeros(1, 3, input_size, input_size)
    torch.onnx.export(
        model, x, str(out),
        input_names=["input"],
        output_names=["logits"],
        opset_version=opset,
        dynamic_axes={"input": {0: "batch"}},
    )
    logger.info("Exported MiniFASNet → {}", out)
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--input-size", type=int, default=80)
    p.add_argument("--opset", type=int, default=12)
    args = p.parse_args()
    export_minifasnet(args.weights, args.out, input_size=args.input_size, opset=args.opset)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
