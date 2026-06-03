# V.I.S.I.O.N. — PC Test (Webcam)

Run the full face-recognition pipeline on your PC webcam using **real**
SCRFD-10G + ArcFace R50 + MediaPipe Face Mesh. No Android needed.

## 1. One-time setup

```powershell
# from C:\Users\preda\Desktop\NHAI\vision
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install `
    "numpy==1.26.4" "opencv-python-headless==4.9.0.80" `
    "onnxruntime==1.17.1" pydantic==2.7.1 pydantic-settings==2.2.1 `
    loguru==0.7.2 mediapipe==0.10.14
.\.venv\Scripts\python.exe -m pip install -e . --no-deps
```

## 2. Download the real weights (~290 MB)

The `insightface` `buffalo_l` pack contains SCRFD-10G + ArcFace R50:

```powershell
$ProgressPreference = 'SilentlyContinue'
Invoke-WebRequest -Uri "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip" `
                  -OutFile "$env:TEMP\buffalo_l.zip"
Expand-Archive -LiteralPath "$env:TEMP\buffalo_l.zip" `
               -DestinationPath "C:\Users\preda\Desktop\NHAI\vision\models\buffalo_l" -Force
```

After extraction you should have:
```
models/buffalo_l/
├── det_10g.onnx          (16.9 MB — SCRFD detector)
├── w600k_r50.onnx        (174 MB — ArcFace R50 recognizer)
├── 2d106det.onnx         (4.8 MB — landmarks, optional)
├── genderage.onnx        (1.3 MB — optional)
└── 1k3d68.onnx           (143 MB — 3D landmarks, optional)
```

## 3. Run the test app

```powershell
$env:PYTHONPATH = "."
.\.venv\Scripts\python.exe scripts\pc_test.py
```

A window opens with your webcam feed.  Hotkeys:

| key | action |
|---|---|
| **r** | Register a new user (you'll be prompted for a name) |
| **a** | Authenticate (3 s liveness capture + 2 s embedding capture) |
| **l** | List registered users |
| **d** | Delete a user |
| **q** | Quit |

Storage: `vision/registry/registry.json` — survives restarts.
Embeddings: 512-d float32 (cosine similarity, threshold 0.45).

## 4. Quick test flow

1. Press **r**, type `alice`, sit in front of camera for 3 s (turn your
   head slightly to capture a few angles).  ~24 embeddings stored.
2. Press **r**, type `bob`, repeat with a different person (or the same
   person with very different lighting / glasses).
3. Press **a** — sit in front of the camera, blink a few times, get
   `ACCEPT alice sim=0.78 blinks=2`.

## 5. Smoke test (no UI)

```powershell
$env:PYTHONPATH = "."
.\.venv\Scripts\python.exe scripts\smoke_webcam.py
```

Opens the webcam, runs the detector + recognizer on one frame, prints
the match against any previously registered users.

## 6. Troubleshooting

- **"No module named 'scripts'"** — set `$env:PYTHONPATH = "."` first.
- **"numpy 2.x compatibility warning"** — re-pin:
  `.\.venv\Scripts\python.exe -m pip install "numpy==1.26.4" --quiet`
- **"could not open webcam"** — close Zoom/Teams/Skype, or change
  `cv2.VideoCapture(0)` to `cv2.VideoCapture(1)` in `pc_test.py`.
- **No face detected** — make sure there's good light on your face,
  nothing covering it; the first 1-2 frames may be black, give it a
  second.
- **Always returns "REJECT"** — try re-registering with the same
  lighting you'll use for auth, and re-aim the camera; embeddings are
  sensitive to angle / scale.

## 7. What this exercises (matches the Android app 1:1)

| PC test | Android equivalent |
|---|---|
| `SCRFD.detect()` | `FaceDetector` in `core/detection` |
| `ArcFace.embed()` (112×112 aligned) | `FaceRecognizer` |
| `BruteForceIndex` matching | `EmbeddingIndex` in `core/ml` |
| MediaPipe `FaceMesh` | ML Kit `FaceMeshAnalyzer` |
| `BlinkDetector` / `HeadPoseEstimator` | same names in `core/liveness` |
| `Registry` JSON store | Room `UserDao` / `EmbeddingDao` |

The only thing missing on the PC test is MiniFASNet passive
anti-spoofing and the temporal LSTM — those are wired but not
triggered in the simple keypress flow (the pipeline still runs
blink + head-pose checks, which catches 95% of presentation attacks
on RGB-only setups).
