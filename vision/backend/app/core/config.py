"""Core (settings, security, deps) for the FastAPI app."""
from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class BackendSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="VISION_BACKEND_", env_file=".env", extra="ignore")
    name: str = "V.I.S.I.O.N. Hybrid Backend"
    debug: bool = False
    secret_key: str = "change-me"
    access_token_minutes: int = 60
    database_url: str = "sqlite:///./vision_backend.db"
    cors_origins: list[str] = ["*"]


@lru_cache
def get_settings() -> BackendSettings:
    return BackendSettings()


settings = get_settings()
