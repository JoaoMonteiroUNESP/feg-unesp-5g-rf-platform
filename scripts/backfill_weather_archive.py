"""Populate valid historical weather for measurements already in SQLite."""

from __future__ import annotations

import argparse
import json

from app.db import SessionLocal, init_db
from app.weather_archive import backfill_archive_weather


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", help="Restrict to one campaign_id")
    parser.add_argument("--dry-run", action="store_true", help="Fetch and validate, then rollback")
    args = parser.parse_args()

    init_db()
    session = SessionLocal()
    try:
        result = backfill_archive_weather(
            session,
            campaign_id=args.campaign,
            dry_run=args.dry_run,
        )
    finally:
        session.close()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
