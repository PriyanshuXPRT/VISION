"""
Centralised configuration for V.I.S.I.O.N.
All paths, thresholds, and feature flags flow from here.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
VISION_ROOT: Path = PROJECT_ROOT / "vision"


# -----------------------------------------------------------------------------
# Settings
# -----------------------------------------------------------------------------
class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="VISION_",
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ---- Environment ----
    env: Literal["development", "staging", "production"] = "development"
    log_level: str = "INFO"
    log_dir: Path = Field(default=PROJECT_ROOT / "logs")

    # ---- Models ----
    models_dir: Path = Field(default=PROJECT_ROOT / "models")
    onnx_dir: Path = Field(default=PROJECT_ROOT / "onnx_models")
    download_pretrained: bool = True

    # SCRFD
    scrfd_model: str = "detection/scrfd_10g_bnkps.onnx"
    scrfd_input_size: int = 320
    scrfd_conf_thresh: float = 0.5
    scrfd_nms_thresh: float = 0.4

    # ArcFace
    arcface_model: str = "recognition/arcface_r100.onnx"
    arcface_input_size: int = 112
    embedding_dim: int = 512

    # MiniFASNet
    fas_model: str = "antispoof/minifasnet_v2.onnx"
    fas_input_size: int = 80
    fas_thresh: float = 0.85

    # MediaPipe
    mp_model_complexity: int = 1
    mp_max_faces: int = 1
    mp_refine_landmarks: bool = True

    # Temporal
    temporal_model: str = "temporal/lstm_v1.onnx"
    temporal_seq_len: int = 30
    temporal_stride: int = 2

    # ---- Recognition ----
    recognition_threshold: float = 0.45
    recognition_top_k: int = 5
    faiss_index_type: Literal["Flat", "IVFFlat", "HNSW"] = "Flat"
    faiss_nlist: int = 64

    # ---- Liveness ----
    liveness_threshold: float = 0.85
    blink_min: int = 1
    blink_max: int = 5
    blink_ear_thresh: float = 0.21
    head_yaw_range: float = 25.0
    head_pitch_range: float = 20.0
    eye_motion_thresh: float = 0.3

    # ---- Database ----
    db_path: Path = Field(default=PROJECT_ROOT / "database" / "vision.db")
    db_backup_dir: Path = Field(default=PROJECT_ROOT / "database" / "backups")

    # ---- Hybrid backend ----
    backend_enabled: bool = False
    backend_url: str = "https://api.vision.local"
    backend_device_key: str = ""
    backend_tenant_id: str = ""

    # ---- Android model CDN ----
    android_model_base_url: str = ""

    # ---- Telemetry ----
    telemetry_enabled: bool = False

    # ---- Derived paths ----
    @property
    def scrfd_path(self) -> Path:
        return self.onnx_dir / self.scrfd_model

    @property
    def arcface_path(self) -> Path:
        return self.onnx_dir / self.arcface_model

    @property
    def fas_path(self) -> Path:
        return self.onnx_dir / self.fas_model

    @property
    def temporal_path(self) -> Path:
        return self.onnx_dir / self.temporal_model

    def ensure_dirs(self) -> None:
        for d in (self.models_dir, self.onnx_dir, self.log_dir, self.db_backup_dir):
            d.mkdir(parents=True, exist_ok=True)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    s = Settings()
    s.ensure_dirs()
    return s


settings = get_settings()
