"""
ArcFace training script (recognition backbone).

Model: IResNet-100 (or ResNet-50) with ArcFace/SubCenter ArcFace head.
Loss:  ArcFace (margin=0.5, scale=30).
Data:  aligned 112x112 RGB crops, organised as `<root>/<identity>/*.jpg`.
Output: `models/checkpoints/recognition/<run_name>/best.ckpt` (state_dict).

Usage:
    python -m training.pipelines.recognition.train_recognition \
        --root datasets/face_recognition_processed/casia_webface \
        --backbone r100 --loss arcface --margin 0.5 --scale 30 \
        --batch-size 256 --epochs 26 --lr 0.1 --output models/checkpoints/recognition/arcface_r100_ms1m
"""
from __future__ import annotations

import argparse
import math
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from training.datasets.augmentation.face_augment import build_train_transform, build_eval_transform


# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
class FaceDataset(Dataset):
    def __init__(self, root: Path, identities: list[str], transform) -> None:
        self.root = root
        self.identities = identities
        self.id_to_idx = {i: k for k, i in enumerate(identities)}
        self.samples: list[tuple[Path, int]] = []
        for i in identities:
            for p in (root / i).iterdir():
                if p.suffix.lower() in {".jpg", ".png", ".jpeg"}:
                    self.samples.append((p, self.id_to_idx[i]))
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        img = np.array(img)
        if self.transform is not None:
            img = self.transform(image=img)["image"]
        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        # normalise to [-1, 1] for IResNet
        img = (img - 0.5) / 0.5
        return img, label


# -----------------------------------------------------------------------------
# Backbone (simple ResNet-100 style for portability)
# -----------------------------------------------------------------------------
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inp: int, out: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(inp, out, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out)
        self.conv2 = nn.Conv2d(out, out * self.expansion, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out * self.expansion)
        self.shortcut = nn.Sequential()
        if stride != 1 or inp != out * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inp, out * self.expansion, 1, stride, bias=False),
                nn.BatchNorm2d(out * self.expansion),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)


class ResNetBackbone(nn.Module):
    def __init__(self, layers: list[int], embedding_dim: int = 512, in_channels: int = 3) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.in_planes = 64
        self.layer1 = self._make_layer(64, layers[0], stride=1)
        self.layer2 = self._make_layer(128, layers[1], stride=2)
        self.layer3 = self._make_layer(256, layers[2], stride=2)
        self.layer4 = self._make_layer(512, layers[3], stride=2)
        self.embed = nn.Linear(512 * BasicBlock.expansion, embedding_dim)
        self._init_weights()

    def _make_layer(self, planes: int, blocks: int, stride: int) -> nn.Sequential:
        layers = [BasicBlock(self.in_planes, planes, stride=stride)]
        self.in_planes = planes * BasicBlock.expansion
        for _ in range(1, blocks):
            layers.append(BasicBlock(self.in_planes, planes))
        return nn.Sequential(*layers)

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        return self.embed(x)


def build_backbone(name: str, embedding_dim: int) -> ResNetBackbone:
    if name == "r50":
        layers = [3, 4, 14, 3]
    elif name == "r100":
        layers = [3, 13, 30, 3]
    else:
        raise ValueError(name)
    return ResNetBackbone(layers=layers, embedding_dim=embedding_dim)


# -----------------------------------------------------------------------------
# ArcFace head
# -----------------------------------------------------------------------------
class ArcFaceHead(nn.Module):
    def __init__(self, embedding_dim: int, num_classes: int, *, margin: float, scale: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)
        self.margin = margin
        self.scale = scale
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        x = F.normalize(embeddings)
        w = F.normalize(self.weight)
        cos_logits = F.linear(x, w)
        sin_logits = torch.sqrt(1.0 - cos_logits.pow(2).clamp(0, 1))
        phi = cos_logits * self.cos_m - sin_logits * self.sin_m
        phi = torch.where(cos_logits > self.th, phi, cos_logits - self.mm)
        one_hot = F.one_hot(labels, num_classes=cos_logits.size(1)).float()
        out = one_hot * phi + (1.0 - one_hot) * cos_logits
        return out * self.scale


# -----------------------------------------------------------------------------
# Training loop
# -----------------------------------------------------------------------------
def train(args: argparse.Namespace) -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = Path(args.root)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    identities = sorted([p.name for p in root.iterdir() if p.is_dir()])
    num_classes = len(identities)
    print(f"Identities: {num_classes}")

    train_ds = FaceDataset(root, identities, transform=build_train_transform())
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    backbone = build_backbone(args.backbone, args.embedding_dim).to(device)
    head = ArcFaceHead(args.embedding_dim, num_classes, margin=args.margin, scale=args.scale).to(device)
    optim = torch.optim.SGD(
        list(backbone.parameters()) + list(head.parameters()),
        lr=args.lr,
        momentum=0.9,
        weight_decay=5e-4,
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs * len(train_loader))
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    best_loss = float("inf")
    for epoch in range(args.epochs):
        backbone.train()
        head.train()
        losses = []
        t0 = time.time()
        for imgs, labels in tqdm(train_loader, desc=f"e{epoch+1}/{args.epochs}"):
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optim.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                emb = backbone(imgs)
                logits = head(emb, labels)
                loss = F.cross_entropy(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
            sched.step()
            losses.append(loss.item())
        mean = float(np.mean(losses))
        print(f"epoch {epoch+1} loss={mean:.4f} time={time.time()-t0:.1f}s")
        if mean < best_loss:
            best_loss = mean
            torch.save(
                {
                    "backbone": backbone.state_dict(),
                    "head": head.state_dict(),
                    "identities": identities,
                    "epoch": epoch + 1,
                    "loss": mean,
                },
                out / "best.ckpt",
            )
            print(f"  ↳ saved {out/'best.ckpt'}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--backbone", choices=["r50", "r100"], default="r100")
    p.add_argument("--embedding-dim", type=int, default=512)
    p.add_argument("--loss", choices=["arcface"], default="arcface")
    p.add_argument("--margin", type=float, default=0.5)
    p.add_argument("--scale", type=float, default=30.0)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--num-workers", type=int, default=8)
    args = p.parse_args()
    return train(args)


if __name__ == "__main__":
    raise SystemExit(main())
