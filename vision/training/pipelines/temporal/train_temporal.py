"""
Train the temporal LSTM liveness classifier.

Inputs:
  - features.npy  shape (N, seq_len, 8)   — pre-extracted sliding-window features
  - labels.npy    shape (N,)              — 1 for live, 0 for spoof

This training script is meant to run *after* the AI pipeline has been used to
extract features from labelled authentication sessions. It outputs an ONNX-
exportable LSTM checkpoint.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


class TemporalLSTM(nn.Module):
    def __init__(self, input_dim: int = 8, hidden_dim: int = 64, num_layers: int = 2) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=0.2)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, _ = self.lstm(x)
        return torch.sigmoid(self.head(h[:, -1, :]))


def train(args: argparse.Namespace) -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X = np.load(args.features)
    y = np.load(args.labels).astype(np.float32)
    print(f"features={X.shape} labels={y.shape} positive={int(y.sum())}/{len(y)}")
    Xt = torch.from_numpy(X).float()
    yt = torch.from_numpy(y).float().unsqueeze(1)
    ds = TensorDataset(Xt, yt)
    n_val = max(1, int(0.1 * len(ds)))
    val_ds, train_ds = torch.utils.data.random_split(ds, [n_val, len(ds) - n_val])
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    model = TemporalLSTM(input_dim=X.shape[-1], hidden_dim=args.hidden_dim).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    bce = nn.BCELoss()

    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    best_acc = 0.0
    for epoch in range(args.epochs):
        model.train()
        for xb, yb in tqdm(train_loader, desc=f"e{epoch+1}/{args.epochs}"):
            xb = xb.to(device); yb = yb.to(device)
            optim.zero_grad(set_to_none=True)
            yhat = model(xb)
            loss = bce(yhat, yb)
            loss.backward()
            optim.step()
        # eval
        model.eval(); correct = total = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device); yb = yb.to(device)
                yhat = (model(xb) >= 0.5).float()
                correct += (yhat == yb).sum().item(); total += yb.numel()
        acc = correct / max(1, total)
        print(f"epoch {epoch+1} val_acc={acc:.4f}")
        if acc > best_acc:
            best_acc = acc
            torch.save({"state_dict": model.state_dict(), "epoch": epoch + 1, "val_acc": acc}, out / "best.ckpt")
            print(f"  ↳ saved {out/'best.ckpt'}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--features", type=Path, required=True)
    p.add_argument("--labels", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-3)
    args = p.parse_args()
    return train(args)


if __name__ == "__main__":
    raise SystemExit(main())
