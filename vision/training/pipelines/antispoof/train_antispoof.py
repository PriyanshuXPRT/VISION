"""
Anti-spoofing training script (MiniFASNetV2 + auxiliary crops).

Binary classification: real vs fake, with an auxiliary head that predicts
attack kind (print / screen / replay) when labels are available.
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from training.datasets.augmentation.face_augment import build_train_transform

ATTACK_KINDS = ["real", "print", "mobile_screen", "laptop_screen", "replay_video", "mask", "unknown"]


# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
class AntiSpoofDataset(Dataset):
    def __init__(self, root: Path, transform) -> None:
        self.root = root
        self.samples: list[tuple[Path, int, int]] = []
        for kind in ATTACK_KINDS:
            d = root / kind
            if not d.is_dir():
                continue
            for ident in sorted(p for p in d.iterdir() if p.is_dir()):
                for img in ident.iterdir():
                    if img.suffix.lower() in {".jpg", ".png", ".jpeg"}:
                        label = 0 if kind == "real" else 1
                        kind_idx = ATTACK_KINDS.index(kind)
                        self.samples.append((img, label, kind_idx))
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label, kind = self.samples[idx]
        img = np.array(Image.open(path).convert("RGB"))
        if self.transform is not None:
            img = self.transform(image=img)["image"]
        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        img = (img - 0.5) / 0.5
        return img, label, kind


# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------
class MiniFASNetV2(nn.Module):
    """Compact 80×80 FAS model. Two scale variants live behind the same class."""

    def __init__(self, num_classes: int = 2, num_kinds: int = len(ATTACK_KINDS), scale: float = 2.7) -> None:
        super().__init__()
        self.scale = scale
        c1, c2, c3, c4 = 32, 64, 128, 256
        self.stem = nn.Sequential(
            nn.Conv2d(3, c1, 3, stride=2, padding=1), nn.BatchNorm2d(c1), nn.PReLU(c1),
            nn.Conv2d(c1, c2, 3, stride=2, padding=1), nn.BatchNorm2d(c2), nn.PReLU(c2),
        )
        self.body = nn.Sequential(
            nn.Conv2d(c2, c3, 3, stride=2, padding=1), nn.BatchNorm2d(c3), nn.PReLU(c3),
            nn.Conv2d(c3, c4, 3, stride=2, padding=1), nn.BatchNorm2d(c4), nn.PReLU(c4),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.dropout = nn.Dropout(0.3)
        self.fc_live = nn.Linear(c4, num_classes)
        self.fc_kind = nn.Linear(c4, num_kinds)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.stem(x)
        x = self.body(x)
        x = self.dropout(x)
        return self.fc_live(x), self.fc_kind(x)


# -----------------------------------------------------------------------------
# Train
# -----------------------------------------------------------------------------
def train(args: argparse.Namespace) -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = Path(args.root)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    train_ds = AntiSpoofDataset(root, transform=build_train_transform(80))
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )
    print(f"samples={len(train_ds)} distribution={Counter([s[1] for s in train_ds.samples])}")

    model = MiniFASNetV2().to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs * len(train_loader))
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    best_loss = float("inf")
    for epoch in range(args.epochs):
        model.train()
        losses = []
        t0 = time.time()
        for imgs, labels, kinds in tqdm(train_loader, desc=f"e{epoch+1}/{args.epochs}"):
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            kinds = kinds.to(device, non_blocking=True)
            optim.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                logits_live, logits_kind = model(imgs)
                loss = F.cross_entropy(logits_live, labels) + 0.1 * F.cross_entropy(logits_kind, kinds)
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
            sched.step()
            losses.append(loss.item())
        mean = float(np.mean(losses))
        print(f"epoch {epoch+1} loss={mean:.4f} time={time.time()-t0:.1f}s")
        if mean < best_loss:
            best_loss = mean
            torch.save({"state_dict": model.state_dict(), "epoch": epoch + 1, "loss": mean}, out / "best.ckpt")
            print(f"  ↳ saved {out/'best.ckpt'}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--num-workers", type=int, default=8)
    args = p.parse_args()
    return train(args)


if __name__ == "__main__":
    raise SystemExit(main())
