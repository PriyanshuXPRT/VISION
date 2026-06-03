# V.I.S.I.O.N. — Implementation Roadmap

## Phase 0 — Skeleton & contracts (✅ done)
* Project layout, pyproject, CI, docs.
* Domain types, exceptions, logging.
* ONNX export + quantisation scripts.

## Phase 1 — Vertical slice MVP (✅ done)
* Python AI core: SCRFD, ArcFace, MiniFASNet, MediaPipe, blink, eye,
  head pose, temporal engine.
* Registration / identification / authentication orchestrators.
* SQLite DB + repos + migrations.
* Android app: CameraX + ONNX Runtime Mobile + Compose UI for all 9
  screens.
* Hybrid FastAPI backend.
* Docker + GitHub Actions.
* Unit tests + security benchmark suite.

## Phase 2 — Real weights and dataset prep (1-2 weeks)
1. Run `vision.onnx.exporters.export_onnx all` against the
   training-time checkpoints.
2. Convert all four ONNX graphs to FP16 + INT8 and ship them in
   `app/src/main/assets/models/`.
3. On the Android side, drop a `RELEASE_NOTES.md` summarising model
   versions and SHA-256 hashes.
4. Hook `vision.mlops.tracking.model_registry` into the release flow.

## Phase 3 — Production-grade training (2-4 weeks)
1. Pre-process CASIA-WebFace / MS1M / VGGFace2 into
   `datasets/face_recognition_processed/`.
2. Train ArcFace (R-100) on MS1M (8 GPUs, ~26 epochs).
3. Train MiniFASNetV2 on CelebA-Spoof + CASIA-FASD + OULU-NPU.
4. Collect ≥ 5000 labelled sessions and train the temporal LSTM.

## Phase 4 — Security hardening (1 week)
* Run the full benchmark suite; tune thresholds to hit
  ACER ≤ 5%, BPCER ≤ 1% @ APCER = 5%.
* Adversarial training for the antispoof model.
* Privacy: enable encrypted Room DB, ship an MD-API integration.
* Pen-test report from a third party.

## Phase 5 — Multi-device / cloud (2-3 weeks)
* JWT device key rotation.
* Postgres + S3-compatible blob store on the backend.
* TLS pinning on the Android side.
* Operator UI for tenant / device management.
* Backup / restore.

## Phase 6 — iOS port (4-6 weeks)
* Swift ports of `core/ml/*`.
* AVCaptureSession + Vision integration.
* GRDB + Encrypted CoreData.
* TCA / SwiftUI shell.
* Xcode Cloud pipeline.

## Phase 7 — Beyond 1000 users (future)
* Replace brute-force index with FAISS HNSW.
* Cluster embeddings by user device for low-latency search.
* Optional client-server hybrid for very large fleets.
