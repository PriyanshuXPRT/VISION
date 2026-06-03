# V.I.S.I.O.N. — Database Schema

The on-device database is **SQLite** (WAL mode) accessed through
`vision.database.Database` on Python and Room (`VisionDatabase`) on
Android. The two schemas mirror each other.

## Tables

### `users`
| Column | Type | Notes |
|---|---|---|
| `user_id` | INTEGER PK AUTOINCREMENT | |
| `name` | TEXT NOT NULL | |
| `phone` | TEXT DEFAULT '' | |
| `email` | TEXT DEFAULT '' | |
| `notes` | TEXT DEFAULT '' | |
| `is_active` | INTEGER NOT NULL DEFAULT 1 | |
| `created_at` | TEXT NOT NULL | ISO-8601 UTC |
| `updated_at` | TEXT NOT NULL | ISO-8601 UTC |

Indexes: PK only.

### `face_templates`
| Column | Type | Notes |
|---|---|---|
| `template_id` | INTEGER PK AUTOINCREMENT | |
| `user_id` | INTEGER NOT NULL | FK → users(user_id), ON DELETE CASCADE |
| `embedding` | BLOB NOT NULL | `float32[512]` packed |
| `quality_score` | REAL NOT NULL | 0..1 |
| `source` | TEXT NOT NULL DEFAULT 'self' | "self" or "admin" |
| `is_primary` | INTEGER NOT NULL DEFAULT 1 | |
| `created_at` | TEXT NOT NULL | ISO-8601 UTC |

Indexes: `ix_face_templates_user` (user_id).

### `auth_logs`
| Column | Type | Notes |
|---|---|---|
| `log_id` | INTEGER PK AUTOINCREMENT | |
| `user_id` | INTEGER | FK → users(user_id), ON DELETE SET NULL |
| `timestamp` | TEXT NOT NULL | ISO-8601 UTC |
| `liveness_score` | REAL NOT NULL | 0..1 |
| `similarity_score` | REAL NOT NULL | 0..1 |
| `result` | TEXT NOT NULL | accept / reject / inconclusive |
| `reason` | TEXT NOT NULL DEFAULT '' | |
| `device_id` | TEXT | |
| `frames` | INTEGER NOT NULL DEFAULT 0 | |
| `latency_ms` | INTEGER NOT NULL DEFAULT 0 | |
| `spoof_kind` | TEXT NOT NULL DEFAULT 'none' | |

Indexes: `ix_auth_logs_user_ts` (user_id, timestamp DESC),
`ix_auth_logs_ts` (timestamp DESC).

### `devices`
| Column | Type | Notes |
|---|---|---|
| `device_id` | TEXT PK | |
| `device_name` | TEXT NOT NULL | |
| `platform` | TEXT NOT NULL | "android" / "ios" |
| `pubkey` | BLOB | optional E2EE pubkey |
| `registered_at` | TEXT NOT NULL | |
| `last_seen_at` | TEXT | nullable |
| `is_revoked` | INTEGER NOT NULL DEFAULT 0 | |

### `settings`
| Column | Type | Notes |
|---|---|---|
| `key` | TEXT PK | |
| `value` | TEXT NOT NULL | |

### `schema_migrations`
| Column | Type | Notes |
|---|---|---|
| `version` | INTEGER PK | |
| `applied_at` | TEXT NOT NULL | |

## Migrations

Forward-only. See `vision/database/migrations/versions.py`.

* **v1** — baseline tables above
* **v2** — `users.role`, `users.external_id`
* **v3** — `devices.session_token`, `devices.locked_until`
* **v4** — `audit_chain` (tamper-evident log hash chain)

## Backup

* Python: `vision.database.Database.backup_to(dst)` (uses SQLite
  backup API — atomic, consistent).
* Android: copy `vision.db` while the app is paused; Room's
  `createFromAsset` / `exportSchema` are enabled.

## Encryption

The DB is **not** encrypted at rest by default. To enable:

* Android: build the Room DB with `SupportFactory(getPassphrase())` and
  pipe a passphrase from `EncryptedSharedPreferences` or Android Keystore.
* Python: use `sqlcipher3` in place of `sqlite3`.

## Cloud shadow schema (backend)

Identical structure under `backend.app.db.models`. Multi-tenant via the
`tenants.tenant_id` foreign keys.
