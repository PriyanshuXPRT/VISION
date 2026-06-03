# V.I.S.I.O.N. — Architecture

> **V**erification and **I**ntelligent **S**poofing-resistant **I**dentity
> **O**ptimization **N**etwork

This document is the single source of truth for the system architecture.
It is intentionally implementation-agnostic and complements the
`README.md` and per-module docstrings.

---

## 1. Goals

| # | Goal | Target |
|---|---|---|
| 1 | Face detection (mAP) | ≥ 95% |
| 2 | Face recognition TAR @ FAR=1e-3 | ≥ 98% |
| 3 | Liveness detection ACER | ≤ 5% |
| 4 | End-to-end auth latency | 2–3 s |
| 5 | Offline operation | 100% |
| 6 | User scale per device | ≥ 1000 |

## 2. High-Level Diagram

```
┌──────────────────────────────┐         ┌─────────────────────────────┐
│       Android Client         │         │     Hybrid Backend          │
│  Kotlin · Compose · CameraX  │  TLS    │  FastAPI · Postgres · S3    │
│  ONNX Runtime Mobile · Room  │ ◀─────▶ │  JWT · Encrypted sync       │
└──────────────┬───────────────┘         └─────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                Local On-Device AI Pipeline (mirror of /vision)       │
│                                                                      │
│  Camera → SCRFD detect → Quality → MiniFASNet antispoof             │
│       → ML Kit Face Mesh → Blink (EAR) → Eye motion                 │
│       → Head pose (PnP) → Temporal LSTM/Transformer                  │
│       → ArcFace embed → 1:N search (BruteForce/HNSW) → Decision    │
└─────────────────────────────────────────────────────────────────────┘
```

The Android app and the Python AI core share the **same logical pipeline**
and the same ONNX models. The Android side is built on top of the same
ONNX files plus a Kotlin port of the per-frame logic.

## 3. Layered Architecture

| Layer | Responsibility | Tech |
|---|---|---|
| **Capture** | Camera frames, lifecycle | CameraX |
| **ML** | Detection, recognition, liveness | ONNX Runtime Mobile (CPU+NNAPI) |
| **Domain** | Use cases, business rules | Pure Kotlin |
| **Data** | Templates, users, logs, settings | Room (SQLite) |
| **Presentation** | UI, ViewModels, state | Jetpack Compose |
| **DI** | Wiring | Hilt |
| **Sync (optional)** | Hybrid cloud sync | Ktor client |

The Python package follows the same layering under `vision/`:

* `vision.ai.*`         — pure inference
* `vision.registration` — enrollment
* `vision.identification` — 1:N search
* `vision.authentication` — orchestrator
* `vision.database`     — SQLite
* `vision.onnx`         — export & quantisation

## 4. Authentication State Machine

```
Idle ──(frame)──▶ Detecting ──(face)──▶ Liveness ──(pass)──▶ Identifying
                       │                    │                    │
                       ▼                    ▼                    ▼
                    NoFace             SpoofDetected         NoMatch / LowSim
                       │                    │                    │
                       ▼                    ▼                    ▼
                    Idle (back)        Reject (audit)         Reject (audit)
```

A *passive* liveness gate runs on every frame; an *active* gate (temporal
LSTM) needs `seq_len` frames before it produces a final score. The final
decision requires **all three** to be green.

## 5. Data Flow

```
enroll.py
  frames ──▶ detect ──▶ align ──▶ embed ──▶ aggregate ──▶ template (sqlite BLOB)
                                                                    │
                                                                    ▼
authenticate.py                                                  search
  frames ──▶ detect ──▶ antispoof ──▶ landmarks ──▶ features ──▶ 1:N ──▶ decision
                                  └─── temporal LSTM ──┘
```

## 6. Threat Model

See [`../security/SECURITY.md`](../security/SECURITY.md).

| Threat | Defence |
|---|---|
| Printed photo | FAS texture + specular analysis |
| Mobile / tablet screen | FAS per scale + temporal motion |
| Laptop / monitor screen | FAS per scale + temporal motion |
| Replay video | Temporal LSTM (motion continuity) |
| Face swap | Multi-modal fusion + adversarial training |
| Deepfake (basic) | Heuristics + future temporal Transformer |
| Mask | FAS + per-frame geometry (yaw/pitch consistency) |
| Database exfiltration | AES-256 at rest, optional remote wipe |
| Network MITM | TLS 1.3 + pinned certs (when sync enabled) |
| Replay of auth attempt | Server-side nonce, device session token |

## 7. Performance Engineering

* **NNAPI** is used automatically on Android 10+ devices for the
  MiniFASNet and SCRFD ONNX graphs (int8 quantised where available).
* **Brute-force 1:N** is O(N·D). For N=1000 and D=512, that's ~2M
  multiply-adds per query — well under 10 ms on a Pixel 6. Beyond that,
  swap in the FAISS-backed HNSW index.
* **Temporal LSTM** uses a fixed `seq_len=30` window; the Android side
  uses a pure-Kotlin `HeuristicEngine` so the device works even without
  the ONNX LSTM model.
* **Memory** is held under ~120 MB at peak (model + index + frame buffer).

## 8. Extensibility Points

* New attack kinds → extend `vision.core.types.SpoofKind` and add a
  classifier head in `train_antispoof.py`.
* New liveness signal → add to `FEATURE_DIM` in `temporal_engine.py`
  and bump the `seq_len`/model accordingly.
* Cloud sync → plug in additional `backend/app/api/v1` routes; the
  Android `VisionApi` is a thin Ktor client and only needs new DTOs.
* iOS port → mirror the `core/ml/*` modules in Swift, swap CameraX for
  AVFoundation and ORT-Mobile for the iOS ORT package.
