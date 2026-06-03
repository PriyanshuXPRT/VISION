"""Health router."""
from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter

router = APIRouter()


@router.get("/healthz")
def healthz() -> dict:
    return {"status": "ok", "ts": datetime.now(tz=timezone.utc).isoformat()}


@router.get("/version")
def version() -> dict:
    return {"name": "V.I.S.I.O.N. Backend", "version": "0.1.0"}
