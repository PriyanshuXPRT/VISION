"""Migration scripts (one file per version). They are run forward-only."""
from __future__ import annotations

from pathlib import Path

# Each migration is a SQL string applied in order.
MIGRATIONS: dict[int, str] = {
    1: """
        -- baseline: all tables defined in SCHEMA_SQL
        CREATE TABLE IF NOT EXISTS users (
            user_id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            phone TEXT NOT NULL DEFAULT '',
            email TEXT NOT NULL DEFAULT '',
            notes TEXT NOT NULL DEFAULT '',
            is_active INTEGER NOT NULL DEFAULT 1,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        """,
    2: """
        -- add role + external_id columns
        ALTER TABLE users ADD COLUMN role TEXT NOT NULL DEFAULT 'user';
        ALTER TABLE users ADD COLUMN external_id TEXT;
        CREATE INDEX IF NOT EXISTS ix_users_external_id ON users(external_id);
        """,
    3: """
        -- add session_token + lock columns
        ALTER TABLE devices ADD COLUMN session_token TEXT;
        ALTER TABLE devices ADD COLUMN locked_until TEXT;
        """,
    4: """
        -- audit chain table for tamper detection on auth_logs
        CREATE TABLE IF NOT EXISTS audit_chain (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            log_id     INTEGER NOT NULL,
            prev_hash  TEXT NOT NULL,
            row_hash   TEXT NOT NULL,
            ts         TEXT NOT NULL,
            FOREIGN KEY (log_id) REFERENCES auth_logs(log_id) ON DELETE CASCADE
        );
        CREATE INDEX IF NOT EXISTS ix_audit_chain_log ON audit_chain(log_id);
        """,
}


def list_migrations_dir() -> Path:
    """Helper for external tooling (returns the migrations directory)."""
    return Path(__file__).resolve().parent
