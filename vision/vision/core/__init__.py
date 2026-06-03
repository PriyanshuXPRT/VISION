"""Core package — pipeline, demo, types, logging, exceptions."""
from vision.core.demo import main as demo_main
from vision.core.pipeline import VisionPipeline, build_pipeline

__all__ = ["VisionPipeline", "build_pipeline", "demo_main"]
