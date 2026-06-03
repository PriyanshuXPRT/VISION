"""SQLite connection wrapper with sane defaults.

Uses synchronous sqlite3 (offline-first; no external DB server).
Async wrappers can layer on top in the Android/HTTP layers.
"""
from __future__ import annotations

import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from vision.config import settings
from vision.core.exceptions import DatabaseError
from vision.core.logging import logger
from vision.database.models import SCHEMA_SQL


def _connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(
        str(path),
        detect_types=sqlite3.PARSE_DECLTYPES,
        check_same_thread=False,
        isolation_level=None,   # autocommit; we manage txns explicitly
    )
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.execute("PRAGMA journal_mode = WAL;")
    conn.execute("PRAGMA synchronous = NORMAL;")
    conn.execute("PRAGMA temp_store = MEMORY;")
    return conn


class Database:
    """Thread-safe SQLite handle.

    The class is intentionally small; the heavy lifting is in repositories.
    """

    def __init__(self, path: Path | None = None) -> None:
        self.path = path or settings.db_path
        self._lock = threading.RLock()
        self._conn: sqlite3.Connection | None = None
        self.connect()
        self.migrate()

    # ---------------------------------------------------------------- lifecycle
    def connect(self) -> None:
        with self._lock:
            if self._conn is None:
                self._conn = _connect(self.path)
                logger.info("Database connected at {}", self.path)

    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            if self._conn is None:
                raise DatabaseError("Database is not connected")
            try:
                self._conn.execute("BEGIN IMMEDIATE;")
                yield self._conn
                self._conn.execute("COMMIT;")
            except Exception:
                self._conn.execute("ROLLBACK;")
                raise

    @contextmanager
    def cursor(self) -> Iterator[sqlite3.Cursor]:
        with self._lock:
            if self._conn is None:
                raise DatabaseError("Database is not connected")
            cur = self._conn.cursor()
            try:
                yield cur
            finally:
                cur.close()

    def execute(self, sql: str, params: tuple = ()) -> sqlite3.Cursor:
        with self.cursor() as cur:
            return cur.execute(sql, params)

    def executemany(self, sql: str, seq: list[tuple]) -> sqlite3.Cursor:
        with self.cursor() as cur:
            return cur.executemany(sql, seq)

    def migrate(self) -> None:
        """Apply the schema and record the migration."""
        with self._lock:
            if self._conn is None:
                raise DatabaseError("Database is not connected")
            cur = self._conn.cursor()
            try:
                for ddl in SCHEMA_SQL:
                    cur.execute(ddl)
                cur.execute("SELECT version FROM schema_migrations ORDER BY version DESC LIMIT 1;")
                row = cur.fetchone()
                last = int(row["version"]) if row else 0
                if last < 1:
                    cur.execute(
                        "INSERT INTO schema_migrations(version, applied_at) VALUES (?, datetime('now'));",
                        (1,),
                    )
                logger.info("Schema migration at version 1 (was {})", last)
            except Exception as exc:  # noqa: BLE001
                raise DatabaseError(f"Schema migration failed: {exc}") from exc
            finally:
                cur.close()

    def backup_to(self, dst: Path) -> None:
        """Atomic backup via SQLite backup API."""
        dst.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            if self._conn is None:
                raise DatabaseError("Database is not connected")
            backup = _connect(dst)
            try:
                self._conn.backup(backup)
            finally:
                backup.close()
        logger.info("Backed up DB to {}", dst)
