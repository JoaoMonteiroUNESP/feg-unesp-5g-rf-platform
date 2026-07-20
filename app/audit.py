"""
Auditing helpers. Every ingestion creates an IngestionRun row; every external
API call creates an ApiCallLog row tied to that run. Nothing in the pipeline
mutates the database without a run_id.
"""
from __future__ import annotations
import json
import logging
from datetime import datetime

from sqlalchemy.orm import Session

from app.db import IngestionRun, ApiCallLog
from app.config import settings

log = logging.getLogger(__name__)


def open_run(session: Session, *, filename: str, file_sha256: str,
             file_size_bytes: int, log_variant: str, delimiter: str,
             columns_detected: list[str], columns_missing: list[str],
             accuracy_threshold_m: float | None = None,
             campaign_id: str | None = None) -> IngestionRun:
    run = IngestionRun(
        filename=filename,
        file_sha256=file_sha256,
        file_size_bytes=file_size_bytes,
        log_variant=log_variant,
        delimiter=delimiter,
        columns_detected=json.dumps(columns_detected, ensure_ascii=False),
        columns_missing=json.dumps(columns_missing, ensure_ascii=False),
        accuracy_threshold_m=accuracy_threshold_m or settings.gps_acc_low_m,
        started_at=datetime.utcnow(),
        campaign_id=(campaign_id or None),
    )
    session.add(run)
    session.flush()        # populate run.id
    return run


def find_existing_run_by_sha(session: Session, sha256: str) -> IngestionRun | None:
    """Return the most recent run with the given SHA-256, or None."""
    return (session.query(IngestionRun)
                   .filter(IngestionRun.file_sha256 == sha256)
                   .order_by(IngestionRun.id.desc())
                   .first())


def close_run(session: Session, run: IngestionRun, *,
              rows_raw: int, rows_dropped_essential: int,
              rows_dropped_gps: int, rows_valid: int,
              notes: str | None = None) -> None:
    run.rows_raw = rows_raw
    run.rows_dropped_essential = rows_dropped_essential
    run.rows_dropped_gps = rows_dropped_gps
    run.rows_valid = rows_valid
    run.ended_at = datetime.utcnow()
    if notes:
        run.notes = notes
    session.commit()


def log_api_call(session: Session, *, run_id: int, api_name: str,
                 lat: float | None, lon: float | None,
                 status: str, http_code: int | None = None,
                 error_message: str | None = None,
                 latency_ms: float | None = None) -> None:
    rec = ApiCallLog(
        run_id=run_id, api_name=api_name, lat=lat, lon=lon,
        status=status, http_code=http_code,
        error_message=(error_message or "")[:2000],
        latency_ms=latency_ms,
    )
    session.add(rec)
