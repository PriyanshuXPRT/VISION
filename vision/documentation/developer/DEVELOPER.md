# V.I.S.I.O.N. — Developer Guide

## 1. Repo layout

```
vision/
├── vision/                # Python AI core (see `pyproject.toml`)
│   ├── ai/                # SCRFD, ArcFace, FAS, landmarks, liveness
│   ├── registration/      # Enrollment pipeline
│   ├── identification/    # 1:N search
│   ├── authentication/    # Pipeline orchestrator
│   ├── database/          # SQLite + repos
│   ├── onnx/              # Export + quantisation
│   ├── core/              # Pipeline factory + types
│   └── config/            # Pydantic settings
├── training/              # PyTorch training scripts
│   ├── datasets/          # Data prep + augmentation
│   ├── pipelines/         # recognition / antispoof / temporal
│   ├── evaluation/        # Metrics + cross-validation
│   └── configs/           # Hydra-style YAMLs
├── models/                # Checkpoints
├── onnx_models/           # Exported ONNX (FP32/FP16/INT8)
├── deployment/            # Per-platform guides
├── evaluation/            # Benchmarks
├── tests/                 # Pytest
├── documentation/         # Architecture / API / DB / Security / ...
├── database/              # SQLite DBs (gitignored)
├── scripts/               # setup/maintenance
├── ci/                    # GitHub Actions + Docker
├── mlops/                 # Tracker + registry
├── backend/               # FastAPI hybrid sync
└── android-app/           # Kotlin/Compose client
```

## 2. Local dev setup

```bash
# 1. Python ≥ 3.10
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .

# 2. Lint / test
ruff check vision training
mypy vision training
pytest -q

# 3. Pull pretrained weights (cached on first run)
python -m vision.core.demo
```

## 3. Running the Android app

```bash
cd android-app
./gradlew :app:installDebug
# Drop your ONNX models in app/src/main/assets/models/ before this if you
# have already exported them.
```

Useful gradle tasks:

| Task | Purpose |
|---|---|
| `:app:assembleDebug` | Build a debug APK |
| `:app:testDebugUnitTest` | Run JVM unit tests |
| `:app:lintDebug` | Lint |
| `:app:dependencies` | Show resolved dependencies |
| `:app:installDebug` | Install to connected device |

## 4. Adding a new liveness signal

1. Add the feature extraction to `vision.ai.liveness.<new>.py`.
2. Update `FEATURE_DIM` in `temporal_engine.py` and re-train the LSTM
   with `training.pipelines.temporal.train_temporal`.
3. Re-export the LSTM to ONNX
   (`vision.onnx.exporters.export_temporal`).
4. Mirror the signal in `OnnxEngine` consumer in the Android
   `TemporalEngine.kt` (heuristic fallback is automatic).
5. Add a unit test in `tests/unit/test_temporal.py`.

## 5. Adding a new screen

1. Create a package `vision.android-app.app.ui.screens.<name>`.
2. Add `@HiltViewModel`-annotated `…ViewModel.kt`.
3. Add the screen Composable, gather state via
   `collectAsState()`.
4. Add a route in `ui/navigation/VisionNavGraph.kt`.
5. Add a destination to `Routes`.

## 6. Releasing

1. `git tag v0.2.0 && git push --tags`
2. CI builds, signs, and uploads the APK to the GitHub release.
3. Update the model registry on the next training cycle and cut a
   `model-v*` tag.

## 7. Coding standards

* **Python** — Ruff + Black, type hints on public APIs, 100 col,
  Google-style docstrings. Tests with pytest; target ≥ 80% coverage
  on `vision/`.
* **Kotlin** — official style, Compose-first, explicit Coroutines,
  no global state outside Hilt. Tests: JUnit4 + Truth + Compose UI test.

## 8. Debugging tips

* Set `VISION_LOG_LEVEL=DEBUG` for verbose logs.
* The Android `adb logcat | grep vision` filters for our app.
* `python -m vision.core.demo` runs an end-to-end smoke test
  with synthetic frames.
* `python -m tests.security.benchmark_liveness --root datasets/security
  --out evaluation/reports/security.json` is the standard
  security regression.
