"""Blink detection subsystem."""
from vision.ai.liveness.blink.blink_detector import (
    BlinkDetector,
    BlinkEvent,
    BlinkResult,
    eye_aspect_ratio,
    has_natural_blink_pattern,
)

__all__ = [
    "BlinkDetector",
    "BlinkEvent",
    "BlinkResult",
    "eye_aspect_ratio",
    "has_natural_blink_pattern",
]
