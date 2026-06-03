"""Sync + log services."""
from __future__ import annotations

from datetime import datetime
from typing import Optional

import numpy as np
from sqlalchemy.orm import Session

from backend.app.db.models import AuthLog, Device, FaceTemplate, User
from backend.app.schemas import AuthLogOut, FaceEmbeddingIn, UserOut


def upsert_user(db: Session, *, tenant: str, u: UserOut) -> User:
    row = db.get(User, u.user_id) if u.user_id else None
    if row is None:
        row = User(tenant_id=tenant, name=u.name, phone=u.phone, email=u.email, notes=u.notes, is_active=u.is_active)
    else:
        row.name, row.phone, row.email, row.notes = u.name, u.phone, u.email, u.notes
        row.is_active = u.is_active
    db.add(row); db.flush()
    return row


def add_template(db: Session, user: User, t: FaceEmbeddingIn) -> FaceTemplate:
    blob = np.asarray(t.embedding, dtype=np.float32).tobytes()
    row = FaceTemplate(
        user_id=user.user_id,
        embedding=blob,
        quality_score=t.quality_score,
        source=t.source,
        is_primary=t.is_primary,
        created_at=t.created_at,
    )
    db.add(row); db.flush()
    return row


def add_log(db: Session, log: AuthLogOut) -> AuthLog:
    row = AuthLog(
        user_id=log.user_id,
        timestamp=log.timestamp,
        liveness_score=log.liveness_score,
        similarity_score=log.similarity_score,
        result=log.result,
        reason=log.reason,
        device_id=log.device_id,
        frames=log.frames,
        latency_ms=log.latency_ms,
        spoof_kind=log.spoof_kind,
    )
    db.add(row); db.flush()
    return row


def register_device(db: Session, *, tenant: str, d: Device) -> Device:
    row = db.get(Device, d.device_id)
    if row is None:
        row = Device(
            device_id=d.device_id, tenant_id=tenant, device_name=d.device_name,
            platform=d.platform, pubkey=d.pubkey,
        )
    else:
        row.device_name, row.platform, row.pubkey = d.device_name, d.platform, d.pubkey
    db.add(row); db.flush()
    return row


def touch_device(db: Session, device_id: str) -> None:
    row = db.get(Device, device_id)
    if row:
        row.last_seen_at = datetime.utcnow()
        db.add(row)
