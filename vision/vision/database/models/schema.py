"""Domain ORM models for SQLite (using raw SQL via aiosqlite for portability)."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


# -----------------------------------------------------------------------------
# Users
# -----------------------------------------------------------------------------
@dataclass(slots=True)
class User:
    user_id: int
    name: str
    phone: str
    email: str
    created_at: datetime
    updated_at: datetime
    is_active: bool = True
    notes: str = ""


# -----------------------------------------------------------------------------
# Face templates
# -----------------------------------------------------------------------------
@dataclass(slots=True)
class FaceTemplate:
    template_id: int
    user_id: int
    embedding: bytes            # float32[512] packed
    quality_score: float
    created_at: datetime
    source: str = "self"         # "self" | "admin"
    is_primary: bool = True


# -----------------------------------------------------------------------------
# Auth log
# -----------------------------------------------------------------------------
@dataclass(slots=True)
class AuthLog:
    log_id: int
    user_id: Optional[int]
    timestamp: datetime
    liveness_score: float
    similarity_score: float
    result: str                  # "accept" | "reject" | "inconclusive"
    reason: str = ""
    device_id: Optional[str] = None
    frames: int = 0
    latency_ms: int = 0
    spoof_kind: str = "none"


# -----------------------------------------------------------------------------
# Devices
# -----------------------------------------------------------------------------
@dataclass(slots=True)
class Device:
    device_id: str
    device_name: str
    platform: str
    registered_at: datetime
    last_seen_at: Optional[datetime] = None
    is_revoked: bool = False
    pubkey: Optional[bytes] = None


# -----------------------------------------------------------------------------
# Schema
# -----------------------------------------------------------------------------
SCHEMA_SQL: list[str] = [
    """
    CREATE TABLE IF NOT EXISTS users (
        user_id     INTEGER PRIMARY KEY AUTOINCREMENT,
        name        TEXT    NOT NULL,
        phone       TEXT    NOT NULL DEFAULT '',
        email       TEXT    NOT NULL DEFAULT '',
        notes       TEXT    NOT NULL DEFAULT '',
        is_active   INTEGER NOT NULL DEFAULT 1,
        created_at  TEXT    NOT NULL,
        updated_at  TEXT    NOT NULL
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS face_templates (
        template_id   INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id       INTEGER NOT NULL,
        embedding     BLOB    NOT NULL,
        quality_score REAL    NOT NULL DEFAULT 0.0,
        source        TEXT    NOT NULL DEFAULT 'self',
        is_primary    INTEGER NOT NULL DEFAULT 1,
        created_at    TEXT    NOT NULL,
        FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
    );
    """,
    "CREATE INDEX IF NOT EXISTS ix_face_templates_user ON face_templates(user_id);",
    """
    CREATE TABLE IF NOT EXISTS auth_logs (
        log_id           INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id          INTEGER,
        timestamp        TEXT    NOT NULL,
        liveness_score   REAL    NOT NULL,
        similarity_score REAL    NOT NULL,
        result           TEXT    NOT NULL,
        reason           TEXT    NOT NULL DEFAULT '',
        device_id        TEXT,
        frames           INTEGER NOT NULL DEFAULT 0,
        latency_ms       INTEGER NOT NULL DEFAULT 0,
        spoof_kind       TEXT    NOT NULL DEFAULT 'none',
        FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE SET NULL
    );
    """,
    "CREATE INDEX IF NOT EXISTS ix_auth_logs_user_ts ON auth_logs(user_id, timestamp DESC);",
    "CREATE INDEX IF NOT EXISTS ix_auth_logs_ts ON auth_logs(timestamp DESC);",
    """
    CREATE TABLE IF NOT EXISTS devices (
        device_id     TEXT    PRIMARY KEY,
        device_name   TEXT    NOT NULL,
        platform      TEXT    NOT NULL,
        pubkey        BLOB,
        registered_at TEXT    NOT NULL,
        last_seen_at  TEXT,
        is_revoked    INTEGER NOT NULL DEFAULT 0
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS settings (
        key   TEXT PRIMARY KEY,
        value TEXT NOT NULL
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS schema_migrations (
        version    INTEGER PRIMARY KEY,
        applied_at TEXT NOT NULL
    );
    """,
]
