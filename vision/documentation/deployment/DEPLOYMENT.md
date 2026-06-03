# V.I.S.I.O.N. — Deployment Guide

## 1. Prerequisites

| Component | Minimum | Recommended |
|---|---|---|
| Android | 8.0 (API 26) | 13+ (API 33+) |
| RAM | 2 GB | 4 GB+ |
| Storage | 500 MB free | 1 GB+ |
| Camera | 5 MP, AF | 12 MP, AF + HDR |
| Internet (sync) | Optional | TLS 1.3 |

## 2. Android deployment (offline)

```bash
cd vision/android-app
./gradlew :app:assembleRelease
adb install -r app/build/outputs/apk/release/app-release.apk
```

The first launch:
1. Copies the model bundle from `assets/models/` to internal storage.
2. Verifies each model's SHA-256 against the registry in
   `models/registry/registry.json`.
3. Initialises Room DB and applies migrations.

## 3. Android deployment (hybrid)

1. Set `VISION_BACKEND_ENABLED=true` in `Settings → Hybrid sync`.
2. Enter the backend URL and device key.
3. The app will authenticate against `/v1/auth/devices` and store the
   JWT in `EncryptedSharedPreferences`.
4. Sync runs on demand; pull returns users + templates since cursor,
   push uploads new templates and recent logs.

## 4. Backend deployment

### Docker (recommended)
```bash
cd vision/backend
docker compose up -d
# API: http://localhost:8000
# Swagger UI: http://localhost:8000/docs
```

### Bare metal
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r backend/requirements.txt
export VISION_BACKEND_DATABASE_URL=postgresql+psycopg2://...
uvicorn backend.app.main:app --host 0.0.0.0 --port 8000
```

## 5. Model updates

1. Drop the new ONNX file into `onnx_models/<component>/<name>.onnx`.
2. Bump the version in the sidecar `<name>.onnx.version`.
3. Run `python -m mlops.tracking.model_registry build` to refresh the
   manifest.
4. Tag the commit: `git tag model-v1.0.1 && git push --tags`.
5. The CI `model-release.yml` workflow validates and publishes a
   GitHub release with the model bundle.

On the Android side, the `ModelStatusScreen` surfaces the current
version and SHA-256 next to the expected one.

## 6. iOS migration (future)

The architecture is iOS-ready by design. Required work:

1. Mirror `core/ml/*` in Swift (4-5 days):
   * `OnnxEngine` → `ORTInferenceSession`
   * `FaceDetector`, `FaceRecognizer`, `AntiSpoof` — same ONNX files
   * `BlinkDetector`, `HeadPoseEstimator`, `TemporalEngine` — direct
     Kotlin→Swift ports
2. Replace `CameraX` with `AVCaptureSession` + `Vision`.
3. Replace `Room` with `GRDB` (same SQL schema).
4. Keep `Ktor` (or `URLSession`) for sync.
5. Wrap with a SwiftUI / TCA shell.

The Python AI core already runs the same pipeline against the same
ONNX files; it can act as the test harness during the iOS port.

## 7. Cloud sync rollout checklist

* [ ] Generate a tenant key (`openssl rand -hex 32`) and put it in
      `VISION_BACKEND_SECRET_KEY`.
* [ ] Issue per-device keys via `/v1/auth/devices` (operator-driven).
* [ ] Enable TLS on the backend (Caddy / nginx).
* [ ] Pin the server cert SHA-256 in the Android app
      (`NetworkSecurityConfig.xml`).
* [ ] Run a load test (`locust` or `k6`) for ≥ 1000 RPS.
* [ ] Configure backup of the `vision_pgdata` volume.
