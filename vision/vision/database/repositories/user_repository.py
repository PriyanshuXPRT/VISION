"""Repository layer — one class per aggregate."""
from __future__ import annotations

import sqlite3
import struct
from datetime import datetime
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from vision.core.exceptions import DatabaseError, DuplicateUserError, UserNotFoundError
from vision.core.logging import logger
from vision.database.connection import Database
from vision.database.models import AuthLog, Device, FaceTemplate, User

_EMBEDDING_DIM = 512


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _now() -> str:
    return datetime.utcnow().isoformat(timespec="seconds")


def _pack_embedding(emb: NDArray[np.float32]) -> bytes:
    arr = np.ascontiguousarray(emb.astype(np.float32))
    if arr.shape != (_EMBEDDING_DIM,):
        raise ValueError(f"Embedding must be ({_EMBEDDING_DIM},), got {arr.shape}")
    return arr.tobytes()


def _unpack_embedding(blob: bytes) -> NDArray[np.float32]:
    return np.frombuffer(blob, dtype=np.float32).copy()


def _user_from_row(row: sqlite3.Row) -> User:
    return User(
        user_id=int(row["user_id"]),
        name=row["name"],
        phone=row["phone"],
        email=row["email"],
        notes=row["notes"],
        is_active=bool(row["is_active"]),
        created_at=datetime.fromisoformat(row["created_at"]),
        updated_at=datetime.fromisoformat(row["updated_at"]),
    )


def _template_from_row(row: sqlite3.Row) -> FaceTemplate:
    return FaceTemplate(
        template_id=int(row["template_id"]),
        user_id=int(row["user_id"]),
        embedding=_unpack_embedding(row["embedding"]),
        quality_score=float(row["quality_score"]),
        source=row["source"],
        is_primary=bool(row["is_primary"]),
        created_at=datetime.fromisoformat(row["created_at"]),
    )


def _authlog_from_row(row: sqlite3.Row) -> AuthLog:
    return AuthLog(
        log_id=int(row["log_id"]),
        user_id=int(row["user_id"]) if row["user_id"] is not None else None,
        timestamp=datetime.fromisoformat(row["timestamp"]),
        liveness_score=float(row["liveness_score"]),
        similarity_score=float(row["similarity_score"]),
        result=row["result"],
        reason=row["reason"],
        device_id=row["device_id"],
        frames=int(row["frames"]),
        latency_ms=int(row["latency_ms"]),
        spoof_kind=row["spoof_kind"],
    )


def _device_from_row(row: sqlite3.Row) -> Device:
    return Device(
        device_id=row["device_id"],
        device_name=row["device_name"],
        platform=row["platform"],
        pubkey=row["pubkey"],
        registered_at=datetime.fromisoformat(row["registered_at"]),
        last_seen_at=datetime.fromisoformat(row["last_seen_at"]) if row["last_seen_at"] else None,
        is_revoked=bool(row["is_revoked"]),
    )


# -----------------------------------------------------------------------------
# UserRepository
# -----------------------------------------------------------------------------
class UserRepository:
    def __init__(self, db: Database) -> None:
        self.db = db

    def create(self, *, name: str, phone: str = "", email: str = "", notes: str = "") -> User:
        if not name.strip():
            raise DuplicateUserError("Name cannot be empty")
        with self.db.transaction() as conn:
            try:
                cur = conn.execute(
                    """INSERT INTO users(name, phone, email, notes, is_active, created_at, updated_at)
                       VALUES (?, ?, ?, ?, 1, ?, ?)""",
                    (name.strip(), phone.strip(), email.strip(), notes.strip(), _now(), _now()),
                )
            except sqlite3.IntegrityError as exc:
                raise DatabaseError(str(exc)) from exc
            uid = int(cur.lastrowid)
        return self.get(uid)

    def get(self, user_id: int) -> User:
        with self.db.cursor() as cur:
            cur.execute("SELECT * FROM users WHERE user_id = ?", (user_id,))
            row = cur.fetchone()
        if not row:
            raise UserNotFoundError(f"user_id={user_id}")
        return _user_from_row(row)

    def get_by_phone_or_email(self, *, phone: str = "", email: str = "") -> Optional[User]:
        with self.db.cursor() as cur:
            if phone:
                cur.execute("SELECT * FROM users WHERE phone = ?", (phone,))
            elif email:
                cur.execute("SELECT * FROM users WHERE email = ?", (email,))
            else:
                return None
            row = cur.fetchone()
        return _user_from_row(row) if row else None

    def list(self, *, include_inactive: bool = False) -> list[User]:
        sql = "SELECT * FROM users"
        if not include_inactive:
            sql += " WHERE is_active = 1"
        sql += " ORDER BY created_at DESC"
        with self.db.cursor() as cur:
            cur.execute(sql)
            return [_user_from_row(r) for r in cur.fetchall()]

    def update(
        self,
        user_id: int,
        *,
        name: Optional[str] = None,
        phone: Optional[str] = None,
        email: Optional[str] = None,
        notes: Optional[str] = None,
        is_active: Optional[bool] = None,
    ) -> User:
        fields: list[str] = []
        params: list = []
        if name is not None:
            fields.append("name = ?"); params.append(name.strip())
        if phone is not None:
            fields.append("phone = ?"); params.append(phone.strip())
        if email is not None:
            fields.append("email = ?"); params.append(email.strip())
        if notes is not None:
            fields.append("notes = ?"); params.append(notes.strip())
        if is_active is not None:
            fields.append("is_active = ?"); params.append(int(is_active))
        if not fields:
            return self.get(user_id)
        fields.append("updated_at = ?"); params.append(_now())
        params.append(user_id)
        with self.db.transaction() as conn:
            conn.execute(f"UPDATE users SET {', '.join(fields)} WHERE user_id = ?", tuple(params))
        return self.get(user_id)

    def delete(self, user_id: int) -> None:
        with self.db.transaction() as conn:
            conn.execute("DELETE FROM users WHERE user_id = ?", (user_id,))


# -----------------------------------------------------------------------------
# FaceTemplateRepository
# -----------------------------------------------------------------------------
class FaceTemplateRepository:
    def __init__(self, db: Database) -> None:
        self.db = db

    def add(
        self,
        user_id: int,
        embedding: NDArray[np.float32],
        quality_score: float,
        *,
        source: str = "self",
        is_primary: bool = True,
    ) -> FaceTemplate:
        blob = _pack_embedding(embedding)
        with self.db.transaction() as conn:
            if is_primary:
                conn.execute("UPDATE face_templates SET is_primary = 0 WHERE user_id = ?", (user_id,))
            cur = conn.execute(
                """INSERT INTO face_templates(user_id, embedding, quality_score, source, is_primary, created_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (user_id, blob, float(quality_score), source, int(is_primary), _now()),
            )
            tid = int(cur.lastrowid)
        return self.get(tid)

    def get(self, template_id: int) -> FaceTemplate:
        with self.db.cursor() as cur:
            cur.execute("SELECT * FROM face_templates WHERE template_id = ?", (template_id,))
            row = cur.fetchone()
        if not row:
            raise DatabaseError(f"template_id={template_id} not found")
        return _template_from_row(row)

    def list_for_user(self, user_id: int) -> list[FaceTemplate]:
        with self.db.cursor() as cur:
            cur.execute(
                "SELECT * FROM face_templates WHERE user_id = ? ORDER BY is_primary DESC, created_at DESC",
                (user_id,),
            )
            return [_template_from_row(r) for r in cur.fetchall()]

    def list_all(self) -> list[FaceTemplate]:
        with self.db.cursor() as cur:
            cur.execute("SELECT * FROM face_templates ORDER BY user_id, is_primary DESC")
            return [_template_from_row(r) for r in cur.fetchall()]

    def delete(self, template_id: int) -> None:
        with self.db.transaction() as conn:
            conn.execute("DELETE FROM face_templates WHERE template_id = ?", (template_id,))

    def delete_for_user(self, user_id: int) -> None:
        with self.db.transaction() as conn:
            conn.execute("DELETE FROM face_templates WHERE user_id = ?", (user_id,))
