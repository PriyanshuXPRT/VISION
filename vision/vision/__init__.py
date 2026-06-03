"""Top-level V.I.S.I.O.N. Python package."""
from vision.core import build_pipeline
from vision.core.types import (
    AuthenticationResult,
    Decision,
    IdentificationResult,
    LivenessReport,
    SpoofKind,
)

__all__ = [
    "build_pipeline",
    "AuthenticationResult",
    "Decision",
    "IdentificationResult",
    "LivenessReport",
    "SpoofKind",
]

__version__ = "0.1.0"
