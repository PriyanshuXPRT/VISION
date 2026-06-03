# Model Directory

This directory hosts pretrained model checkpoints.

The `buffalo_l/` pack from InsightFace (SCRFD-10G detector, ArcFace r100,
landmarks, gender/age) is **not** committed to this repository because some
ONNX files exceed GitHub's 100 MB limit.

## Fetching buffalo_l

```bash
python -m vision.onnx.exporters.export_arcface   # or run scripts/setup/bootstrap.py
```

Or download directly from the InsightFace model zoo and place contents here:

```
models/buffalo_l/
├── det_10g.onnx       # SCRFD-10G detector
├── w600k_r50.onnx     # ArcFace r100 recogniser
├── 1k3d68.onnx        # 68-landmark
├── 2d106det.onnx      # 106-landmark
└── genderage.onnx     # Gender / age
```

The application will pick the models up automatically from this path.
