"""Auth and device repositories."""
from __future__ import annotations

import sqlite3
from datetime import datetime
from typing import Optional

from vision.database.connection import Database
from vision.database.models import AuthLog, Device
from vision.database.repositories.user_repository import _authlog_from_row, _device_from_row, _now


# -----------------------------------------------------------------------------
# AuthLogRepository
# -----------------------------------------------------------------------------
class AuthLogRepository:
    def __init__(self, db: Database) -> None:
        self.db = db

    def add(
        self,
        *,
        user_id: Optional[int],
        liveness_score: float,
        similarity_score: float,
        result: str,
        reason: str = "",
        device_id: Optional[str] = None,
        frames: int = 0,
        latency_ms: int = 0,
        spoof_kind: str = "none",
        timestamp: Optional[datetime] = None,
    ) -> AuthLog:
        ts = (timestamp or datetime.utcnow()).isoformat(timespec="seconds")
        with self.db.transaction() as conn:
            cur = conn.execute(
                """INSERT INTO auth_logs
                    (user_id, timestamp, liveness_score, similarity_score, result, reason,
                     device_id, frames, latency_ms, spoof_kind)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    user_id, ts, float(liveness_score), float(similarity_score),
                    result, reason, device_id, int(frames), int(latency_ms), spoof_kind,
                ),
            )
            lid = int(cur.lastrowid)
        with self.db.cursor() as cur:
            cur.execute("SELECT * FROM auth_logs WHERE log_id = ?", (lid,))
            return _authlog_from_row(cur.fetchone())

    def list_recent(self, *, limit: int = 100, user_id: Optional[int] = None) -> list[AuthLog]:
        with self.db.cursor() as cur:
            if user_id is None:
                cur.execute("SELECT * FROM auth_logs ORDER BY timestamp DESC LIMIT ?", (limit,))
            else:
                cur.execute(
                    "SELECT * FROM auth_logs WHERE user_id = ? ORDER BY timestamp DESC LIMIT ?",
                    (user_id, limit),
                )
            return [_authlog_from_row(r) for r in cur.fetchall()]

    def stats_since(self, since: datetime) -> dict[str, int]:
        with self.db.cursor() as cur:
            cur.execute(
                """SELECT result, COUNT(*) AS n FROM auth_logs WHERE timestamp >= ? GROUP BY result""",
                (since.isoformat(timespec="seconds"),),
            )
            return {r["result"]: int(r["n"]) for r in cur.fetchall()}

    def purge_older_than(self, days: int) -> int:
        with self.db.transaction() as conn:
            cur = conn.execute(
                "DELETE FROM auth_logs WHERE timestamp < datetime('now', ?)",
                (f"-{int(days)} days",),
            )
            return cur.rowcount


# -----------------------------------------------------------------------------
# DeviceRepository
# -----------------------------------------------------------------------------
class DeviceRepository:
    def __init__(self, db: Database) -> None:
        self.db = db

    def register(
        self,
        device_id: str,
        device_name: str,
        platform: str,
        pubkey: Optional[bytes] = None,
    ) -> Device:
        with self.db.transaction() as conn:
            conn.execute(
                """INSERT INTO devices(device_id, device_name, platform, pubkey, registered_at)
                   VALUES (?, ?, ?, ?, ?)
                   ON CONFLICT(device_id) DO UPDATE SET
                       device_name=excluded.device_name,
                       platform=excluded.platform,
                       pubkey=COALESCE(excluded.pubkey, devices.pubkey)""",
                (device_id, device_name, platform, pubkey, _now()),
            )
        return self.get(device_id)

    def get(self, device_id: str) -> Device:
        with self.db.cursor() as cur:
            cur.execute("SELECT * FROM devices WHERE device_id = ?", (device_id,))
            row = cur.fetchone()
        if not row:
            raise KeyError(device_id)
        return _device_from_row(row)

    def list(self) -> list[Device]:
        with self.db.cursor() as cur:
            cur.execute("SELECT * FROM devices ORDER BY registered_at DESC")
            return [_device_from_row(r) for r in cur.fetchall()]

    def touch(self, device_id: str) -> None:
        with self.db.transaction() as conn:
            conn.execute(
                "UPDATE devices SET last_seen_at = ? WHERE device_id = ?", (_now(), device_id)
            )

    def revoke(self, device_id: str, revoked: bool = True) -> None:
        with self.db.transaction() as conn:
            conn.execute(
                "UPDATE devices SET is_revoked = ? WHERE device_id = ?", (int(revoked), device_id)
            )
