"""
Export the temporal LSTM/Transformer liveness classifier to ONNX.

Input:  (1, seq_len, 8) float32 feature sequence
Output: (1, 1) float32 live probability (post-sigmoid)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn

from vision.config import settings
from vision.core.logging import logger


class _TemporalLSTM(nn.Module):
    def __init__(self, input_dim: int = 8, hidden_dim: int = 64, num_layers: int = 2) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, _ = self.lstm(x)
        last = h[:, -1, :]
        return torch.sigmoid(self.head(last))


def export_temporal(
    weights: Path,
    out: Path,
    *,
    seq_len: int | None = None,
    input_dim: int = 8,
    opset: int = 12,
) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    seq_len = seq_len or settings.temporal_seq_len
    model = _TemporalLSTM(input_dim=input_dim)
    if weights.is_file():
        try:
            sd = torch.load(weights, map_location="cpu")
            if isinstance(sd, dict) and "state_dict" in sd:
                sd = sd["state_dict"]
            model.load_state_dict(sd, strict=False)
            logger.info("Loaded temporal weights from {}", weights)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not load temporal weights {}: {}", weights, exc)
    model.eval()
    x = torch.zeros(1, seq_len, input_dim)
    torch.onnx.export(
        model, x, str(out),
        input_names=["features"],
        output_names=["live_prob"],
        opset_version=opset,
        dynamic_axes={"features": {0: "batch", 1: "seq"}},
    )
    logger.info("Exported temporal model → {}", out)
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seq-len", type=int, default=settings.temporal_seq_len)
    p.add_argument("--input-dim", type=int, default=8)
    p.add_argument("--opset", type=int, default=12)
    args = p.parse_args()
    export_temporal(
        args.weights, args.out,
        seq_len=args.seq_len, input_dim=args.input_dim, opset=args.opset,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
