"""Authentication / device registration endpoints."""
from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from backend.app.core import make_token, random_device_key
from backend.app.core.deps import current_principal
from backend.app.db import Device, get_db
from backend.app.schemas import DeviceRegistration, TokenResponse
from backend.app.services import register_device

router = APIRouter()


@router.post("/devices", response_model=TokenResponse)
def register(d: DeviceRegistration, db: Annotated[Session, Depends(get_db)]) -> TokenResponse:
    if not d.device_id.strip():
        raise HTTPException(400, "device_id is required")
    db_d = Device(
        device_id=d.device_id, device_name=d.device_name, platform=d.platform, pubkey=d.pubkey,
    )
    register_device(db, tenant="default", d=db_d)
    db.commit()
    return TokenResponse(access_token=make_token(d.device_id))


@router.get("/devices/me")
def me(principal: Annotated[dict, Depends(current_principal)]) -> dict:
    return {"device_id": principal.get("sub")}
