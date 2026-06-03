"""Maintenance helpers — DB cleanup, log rotation, model downloads."""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta

from vision.config import settings
from vision.core.logging import logger, setup_logging
from vision.database import AuthLogRepository, Database


def purge_logs(days: int) -> int:
    db = Database()
    try:
        repo = AuthLogRepository(db)
        cutoff = datetime.utcnow() - timedelta(days=days)
        n = repo.purge_older_than(cutoff)
        logger.info("Purged {} auth logs older than {} days", n, days)
        return n
    finally:
        db.close()


def backup_db(dst: Path) -> None:
    db = Database()
    try:
        db.backup_to(dst)
    finally:
        db.close()


def main() -> int:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("purge-logs").add_argument("--days", type=int, default=90)
    b = sub.add_parser("backup-db")
    b.add_argument("--dst", type=Path, required=True)
    args = p.parse_args()
    setup_logging()
    if args.cmd == "purge-logs":
        purge_logs(args.days)
    elif args.cmd == "backup-db":
        backup_db(args.dst)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
