"""Password / token helpers."""
from __future__ import annotations

import hashlib
import hmac
import os
import secrets
from datetime import datetime, timedelta, timezone
from typing import Optional

import jwt

from backend.app.core.config import settings


def hash_password(pw: str, *, iterations: int = 120_000) -> str:
    salt = os.urandom(16)
    dk = hashlib.pbkdf2_hmac("sha256", pw.encode(), salt, iterations)
    return f"pbkdf2_sha256${iterations}${salt.hex()}${dk.hex()}"


def verify_password(pw: str, hashed: str) -> bool:
    try:
        algo, iters, salt, dk = hashed.split("$")
        if algo != "pbkdf2_sha256":
            return False
        dk2 = hashlib.pbkdf2_hmac("sha256", pw.encode(), bytes.fromhex(salt), int(iters))
        return hmac.compare_digest(dk, dk2.hex())
    except Exception:  # noqa: BLE001
        return False


def make_token(subject: str, *, expires_min: Optional[int] = None) -> str:
    exp = datetime.now(tz=timezone.utc) + timedelta(minutes=expires_min or settings.access_token_minutes)
    return jwt.encode({"sub": subject, "exp": exp}, settings.secret_key, algorithm="HS256")


def decode_token(token: str) -> dict:
    return jwt.decode(token, settings.secret_key, algorithms=["HS256"])


def random_device_key() -> str:
    return secrets.token_urlsafe(32)
