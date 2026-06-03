"""Sync endpoints — pull/push users, templates, logs."""
from __future__ import annotations

from datetime import datetime
from typing import Annotated, Optional

import numpy as np
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.core.deps import current_principal
from backend.app.db import AuthLog, FaceTemplate, User, get_db
from backend.app.schemas import (
    AuthLogOut,
    FaceEmbeddingIn,
    SyncPullRequest,
    SyncPullResponse,
    SyncPushRequest,
    SyncPushResponse,
    UserOut,
)
from backend.app.services import add_log, add_template, touch_device, upsert_user

router = APIRouter()


@router.post("/sync/pull", response_model=SyncPullResponse)
def pull(
    req: SyncPullRequest,
    db: Annotated[Session, Depends(get_db)],
    principal: Annotated[dict, Depends(current_principal)],
) -> SyncPullResponse:
    device_id = principal.get("sub", "")
    if device_id:
        touch_device(db, device_id)

    q = select(User)
    if req.since:
        q = q.where(User.updated_at >= req.since)
    users = [UserOut.model_validate(u) for u in db.scalars(q).all()]
    templates: list[FaceEmbeddingIn] = []
    if req.include_templates:
        tq = select(FaceTemplate)
        if req.since:
            tq = tq.where(FaceTemplate.created_at >= req.since)
        for t in db.scalars(tq).all():
            emb = np.frombuffer(t.embedding, dtype=np.float32).tolist()
            templates.append(
                FaceEmbeddingIn(
                    user_id=t.user_id,
                    embedding=emb,
                    quality_score=t.quality_score,
                    source=t.source,
                    is_primary=t.is_primary,
                    created_at=t.created_at,
                )
            )
    return SyncPullResponse(server_time=datetime.utcnow(), users=users, templates=templates, cursor=req.since.isoformat() if req.since else "")


@router.post("/sync/push", response_model=SyncPushResponse)
def push(
    req: SyncPushRequest,
    db: Annotated[Session, Depends(get_db)],
    principal: Annotated[dict, Depends(current_principal)],
) -> SyncPushResponse:
    accepted = rejected = 0
    device_id = principal.get("sub", "")
    try:
        for t in req.templates:
            user = db.get(User, t.user_id)
            if user is None:
                rejected += 1
                continue
            add_template(db, user, t)
            accepted += 1
        for log_dict in req.logs:
            try:
                log = AuthLogOut(**log_dict)
                add_log(db, log)
                accepted += 1
            except Exception:
                rejected += 1
        if device_id:
            touch_device(db, device_id)
        db.commit()
    except Exception as exc:  # noqa: BLE001
        db.rollback()
        raise HTTPException(500, f"push failed: {exc}") from exc
    return SyncPushResponse(accepted=accepted, rejected=rejected, server_time=datetime.utcnow())


@router.get("/logs/recent", response_model=list[AuthLogOut])
def recent_logs(
    db: Annotated[Session, Depends(get_db)],
    principal: Annotated[dict, Depends(current_principal)],
    limit: int = 50,
    user_id: Optional[int] = None,
) -> list[AuthLogOut]:
    q = select(AuthLog).order_by(AuthLog.timestamp.desc()).limit(limit)
    if user_id is not None:
        q = q.where(AuthLog.user_id == user_id)
    return [AuthLogOut.model_validate(r) for r in db.scalars(q).all()]
