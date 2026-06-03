# V.I.S.I.O.N. — API Reference

## Python — top-level functions

### `vision.core.build_pipeline(db=None, *, prefer_index="faiss") -> VisionPipeline`
Construct and wire every engine, repository and service.

```python
from vision.core import build_pipeline
pipeline = build_pipeline()
pipeline.registration.enroll(name="Alice", video_source=0)
result = pipeline.authenticator.authenticate(camera_frames)
```

### `vision.registration.RegistrationService.enroll`
Enroll a new user. `video_source` may be a path or a camera index. The
service captures 3 s, runs SCRFD + alignment + ArcFace, and persists a
single L2-normalised template per user.

### `vision.authentication.Authenticator.authenticate`
Run the full pipeline on a `(ts_ms, frame_rgb)` iterator. Returns an
[`AuthenticationResult`](../../vision/core/types.py) with the final
decision, scores, and per-signal liveness breakdown.

## Python — domain types

| Class | Description |
|---|---|
| `BoundingBox` | (x1, y1, x2, y2) face box |
| `Landmarks` | 468 (x, y, z) MediaPipe-style points |
| `AlignedFace` | Cropped + warped 112×112 RGB face |
| `FrameAnalysis` | Per-frame pipeline output |
| `LivenessReport` | Aggregated liveness verdict |
| `IdentificationResult` | 1:N identification outcome |
| `AuthenticationResult` | Top-level decision + audit info |

## Python — settings

All runtime config is driven by `vision.config.settings` (Pydantic
Settings). Override via `.env` or environment variables prefixed with
`VISION_` (e.g. `VISION_RECOGNITION_THRESHOLD=0.5`).

## Python — error hierarchy

```
VisionError
├── ConfigError
├── ModelNotFoundError / ModelLoadError / InferenceError
├── NoFaceDetectedError / MultipleFacesError / FaceQualityError
├── LivenessError / SpoofDetectedError
├── DatabaseError / UserNotFoundError / DuplicateUserError
└── CameraError / AssetError
```

## Hybrid backend — HTTP API (v1)

Base URL: `https://<host>/v1`

| Method | Path | Auth | Purpose |
|---|---|---|---|
| GET | `/healthz` | — | Liveness |
| GET | `/version` | — | Build info |
| POST | `/auth/devices` | — | Register a device, returns JWT |
| GET | `/auth/devices/me` | Bearer | Return current device |
| POST | `/sync/pull` | Bearer | Pull users + templates since cursor |
| POST | `/sync/push` | Bearer | Push templates + logs |
| GET | `/logs/recent` | Bearer | Recent auth logs |

All payloads are JSON; see `backend/app/schemas/__init__.py` for the
shape of each request/response.

## Android — public Kotlin API

### `OnnxEngine.fromAsset(context, "models/arcface_r100.onnx")`
Loads an ONNX model from `assets/models/`. Sessions are cached.

### `FaceDetector(context)`
SCRFD-10GF detector. `detect(rgb, w, h)` returns a `List<Detection>`.

### `FaceRecognizer(context)`
ArcFace R-100 embedder. `generateEmbedding(aligned112)` returns a
`FloatArray(512)`.

### `AntiSpoof(context)`
MiniFASNetV2 ensemble. `predict(rgb, w, h, bbox)` returns a
`LivenessPrediction`.

### `AuthenticateUseCase`
End-to-end use case. `authenticate(bitmap, ts)` returns
`AuthenticationResult?` (`null` until temporal buffer fills).

## Android — Gradle

```bash
./gradlew :app:installDebug          # install to connected device
./gradlew :app:assembleRelease       # build signed release
./gradlew :app:testDebugUnitTest     # JVM unit tests
```
