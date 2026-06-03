"""API v1 router."""
from fastapi import APIRouter

from backend.app.api.v1 import auth, health, sync

api_router = APIRouter()
api_router.include_router(health.router, tags=["health"])
api_router.include_router(auth.router, prefix="/auth", tags=["auth"])
api_router.include_router(sync.router, prefix="", tags=["sync"])
