"""Pydantic schemas."""
from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


# ---- Auth ----
class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int


class DeviceRegistration(BaseModel):
    device_id: str
    device_name: str
    platform: str
    pubkey: Optional[bytes] = None


# ---- Users ----
class UserCreate(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    phone: str = ""
    email: str = ""
    notes: str = ""


class UserOut(BaseModel):
    user_id: int
    name: str
    phone: str
    email: str
    notes: str
    is_active: bool
    created_at: datetime
    updated_at: datetime


# ---- Templates ----
class FaceTemplateOut(BaseModel):
    template_id: int
    user_id: int
    quality_score: float
    source: str
    is_primary: bool
    created_at: datetime


# ---- Sync ----
class FaceEmbeddingIn(BaseModel):
    user_id: int
    embedding: list[float] = Field(min_length=128, max_length=2048)
    quality_score: float = 0.0
    source: str = "self"
    is_primary: bool = True
    created_at: datetime


class SyncPullRequest(BaseModel):
    since: Optional[datetime] = None
    include_templates: bool = True


class SyncPullResponse(BaseModel):
    server_time: datetime
    users: list[UserOut]
    templates: list[FaceEmbeddingIn]
    cursor: str


class SyncPushRequest(BaseModel):
    templates: list[FaceEmbeddingIn]
    logs: list[dict] = []


class SyncPushResponse(BaseModel):
    accepted: int
    rejected: int
    server_time: datetime


# ---- Logs ----
class AuthLogOut(BaseModel):
    log_id: int
    user_id: Optional[int]
    timestamp: datetime
    liveness_score: float
    similarity_score: float
    result: str
    reason: str
    device_id: Optional[str]
    frames: int
    latency_ms: int
    spoof_kind: str
