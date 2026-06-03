# V.I.S.I.O.N.
## Verification and Intelligent Spoofing-resistant Identity Optimization Network

Production-grade, offline-first AI facial authentication platform.

> **Status:** MVP build · Android (Kotlin/Compose) + Python AI core + FastAPI hybrid backend

---

## 1. Mission

A 100% offline-capable, RGB-camera-only biometric authentication system that:

* Identifies registered users via 1:N face recognition
* Detects and rejects presentation attacks (print, screen, replay, deepfake)
* Runs on commodity Android phones in 2–3 seconds
* Scales to ~1000 users per device
* Is open-source at the model layer and auditable at every stage

---

## 2. High-Level Architecture

```
                          ┌─────────────────────────────┐
                          │       Android Client         │
                          │  Kotlin · Compose · CameraX  │
                          │  ONNX Runtime Mobile         │
                          └──────────────┬──────────────┘
                                         │ (optional, E2EE)
                                         ▼
                          ┌─────────────────────────────┐
                          │      Hybrid Backend          │
                          │  FastAPI · Postgres · Redis  │
                          │  Sync · Audit · Admin        │
                          └─────────────────────────────┘

   ┌──────────────────────────────────────────────────────────────┐
   │                     Python AI Pipeline                       │
   │                                                              │
   │  Camera → SCRFD detect → Quality → MiniFASNet antispoof     │
   │       → MediaPipe landmarks → Blink (EAR) → Eye motion     │
   │       → Head pose (PnP) → Temporal LSTM/Transformer         │
   │       → ArcFace embed → 1:N search (FAISS/HNSW) → Decision │
   └──────────────────────────────────────────────────────────────┘
```

Full architecture: [`documentation/architecture/ARCHITECTURE.md`](documentation/architecture/ARCHITECTURE.md)

---

## 3. Repository Layout

```
vision/
├── vision/                # Python AI core (inference)
├── training/              # Training pipelines (face rec, antispoof, temporal)
├── models/                # Pretrained checkpoints + registry
├── onnx_models/           # Exported ONNX (FP32, FP16, INT8)
├── deployment/            # Android/iOS/edge deployment guides
├── evaluation/            # Benchmarks + protocols
├── tests/                 # Unit, integration, security, perf
├── documentation/         # Architecture, API, DB, security, user, dev
├── database/              # SQLite schema + migrations + seeds
├── scripts/               # Setup, maintenance, benchmarks
├── ci/                    # GitHub Actions + Docker
├── mlops/                 # Tracking, versioning
├── backend/               # FastAPI hybrid sync server
└── android-app/           # Kotlin / Jetpack Compose app
```

---

## 4. Quickstart

### 4.1 Python AI core (development host)

```bash
python -m venv .venv && source .venv/bin/activate     # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
pip install -e .

# Run demo
python -m vision.core.demo
```

### 4.2 Android app

```bash
cd android-app
./gradlew :app:installDebug
```

### 4.3 Hybrid backend (optional)

```bash
cd backend
docker compose up -d
```

---

## 5. Module Index

| Module | Path | Purpose |
|---|---|---|
| Face detection | `vision/vision/ai/detection/face_detector.py` | SCRFD-10GF ONNX |
| Face recognition | `vision/vision/ai/recognition/face_recognizer.py` | ArcFace 512-D |
| Anti-spoofing | `vision/vision/ai/antispoof/anti_spoof.py` | MiniFASNetV2 |
| Landmarks | `vision/vision/ai/landmarks/landmark_engine.py` | MediaPipe 468 |
| Blink | `vision/vision/ai/liveness/blink/blink_detector.py` | EAR |
| Eye tracking | `vision/vision/ai/liveness/eye/eye_tracking.py` | Pupil/saccade |
| Head pose | `vision/vision/ai/liveness/headpose/head_pose.py` | solvePnP |
| Temporal | `vision/vision/ai/liveness/temporal/temporal_engine.py` | LSTM/Transformer |
| Registration | `vision/vision/registration/` | 3-sec video enrollment |
| Identification | `vision/vision/identification/` | 1:N search |
| Authentication | `vision/vision/authentication/` | Full pipeline orchestrator |
| Database | `vision/vision/database/` | SQLite + repos |
| ONNX | `vision/vision/onnx/` | Export + quantize |
| Training | `vision/training/` | Train rec/antispoof/temporal |
| Evaluation | `vision/evaluation/` | Benchmarks |
| Backend | `vision/backend/` | FastAPI hybrid sync |
| Android | `vision/android-app/` | Kotlin Compose app |

---

## 6. Security Posture

V.I.S.I.O.N. combines **passive** (per-frame MiniFASNet) and **active** (blink + eye motion + head pose + temporal sequence) liveness. Multi-signal temporal fusion is the last line of defence against:

* Printed photographs
* Mobile / tablet / laptop screen replays
* Video replay attacks
* Basic face-swap and deepfake attacks

Threat model and benchmarks: [`documentation/security/SECURITY.md`](documentation/security/SECURITY.md)

---

## 7. Performance Targets

| Metric | Target | Status |
|---|---|---|
| Face detection (mAP) | ≥ 95 % | TBD |
| Face recognition (TAR@FAR=1e-3) | ≥ 98 % | TBD |
| Liveness (ACER) | ≤ 5 % | TBD |
| End-to-end auth latency | 2–3 s | TBD |
| Offline operation | 100 % | ✓ |
| Users per device | 1000+ | ✓ |

---

## 8. License

Apache-2.0. See `LICENSE`.
