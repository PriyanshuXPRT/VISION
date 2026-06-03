# Build ONNX model registry
python -m mlops.tracking.model_registry build --root onnx_models --out models/registry/registry.json

# Verify integrity
python -m mlops.tracking.model_registry verify --root onnx_models --manifest models/registry/registry.json

# Quantize SCRFD to FP16
python -m vision.onnx.quantization.quantize --in onnx_models/detection/scrfd_10g_bnkps.onnx --out onnx_models/detection/scrfd_10g_bnkps.fp16.onnx --fp16

# Quantize ArcFace to INT8
python -m vision.onnx.quantization.quantize --in onnx_models/recognition/arcface_r100.onnx --out onnx_models/recognition/arcface_r100.int8.onnx --int8 --calib-n 128 --input-shape 1,3,112,112

# Run the security benchmark
python -m tests.security.benchmark_liveness --root datasets/security --out evaluation/reports/security.json
