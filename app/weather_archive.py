"""Reproducible historical-weather enrichment by measurement campaign.

The analytical fallback is the campaign median from Open-Meteo Archive. Point
hour values are also stored for audit, but are not silently substituted for a
field observation. One API request is made per campaign/date group.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections import defaultdict
from datetime import datetime

import httpx
import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from app.config import settings
from app.db import ApiCallLog, Measurement


HOURLY_FIELDS = {
    "temperature_2m": "temperature_c_archive",
    "relative_humidity_2m": "humidity_archive",
    "cloud_cover": "cloud_cover_pct_archive",
}


def archive_frame(payload: dict) -> pd.DataFrame:
    """Validate an Open-Meteo hourly payload and return a timestamp index."""
    hourly = payload.get("hourly") or {}
    times = hourly.get("time") or []
    if not times:
        return pd.DataFrame(columns=list(HOURLY_FIELDS.values()))
    frame = pd.DataFrame(index=pd.to_datetime(times, errors="coerce"))
    for source, target in HOURLY_FIELDS.items():
        values = hourly.get(source)
        if not isinstance(values, list) or len(values) != len(frame):
            frame[target] = np.nan
        else:
            frame[target] = pd.to_numeric(values, errors="coerce")
    return frame.loc[~frame.index.isna()].sort_index()


def nearest_archive_row(
    frame: pd.DataFrame,
    timestamp: datetime,
    tolerance: str = "90min",
) -> tuple[pd.Timestamp, pd.Series] | None:
    """Return the closest archive hour inside a bounded tolerance."""
    if frame.empty or timestamp is None:
        return None
    target = pd.Timestamp(timestamp)
    position = frame.index.get_indexer(
        [target], method="nearest", tolerance=pd.Timedelta(tolerance)
    )[0]
    if position < 0:
        return None
    return frame.index[position], frame.iloc[position]


def _group_key(measurement: Measurement) -> str | None:
    if measurement.campaign_id:
        return str(measurement.campaign_id)
    if measurement.timestamp_log:
        return f"date:{measurement.timestamp_log.date().isoformat()}"
    return None


def _query_id(params: dict) -> str:
    canonical = json.dumps(params, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def backfill_archive_weather(
    session: Session,
    *,
    campaign_id: str | None = None,
    dry_run: bool = False,
    client: httpx.Client | None = None,
) -> dict:
    """Fetch and persist campaign-level historical weather.

    Manual values are never overwritten. Archive point values and campaign
    medians are written to dedicated columns so derivations can apply the
    declared manual > archive > missing policy.
    """
    query = session.query(Measurement)
    if campaign_id:
        query = query.filter(Measurement.campaign_id == campaign_id)
    measurements = query.order_by(Measurement.timestamp_log).all()

    groups: dict[str, list[Measurement]] = defaultdict(list)
    skipped_without_time = 0
    for measurement in measurements:
        key = _group_key(measurement)
        if key is None or measurement.timestamp_log is None:
            skipped_without_time += 1
            continue
        groups[key].append(measurement)

    own_client = client is None
    http = client or httpx.Client(
        timeout=settings.http_timeout_s,
        headers={"User-Agent": settings.http_user_agent, "Accept": "application/json"},
    )
    summaries = []
    try:
        for key, rows in groups.items():
            lat = float(np.median([row.latitude for row in rows]))
            lon = float(np.median([row.longitude for row in rows]))
            dates = [row.timestamp_log.date() for row in rows]
            params = {
                "latitude": round(lat, 6),
                "longitude": round(lon, 6),
                "start_date": min(dates).isoformat(),
                "end_date": max(dates).isoformat(),
                "hourly": ",".join(HOURLY_FIELDS),
                "timezone": "America/Sao_Paulo",
            }
            query_id = _query_id(params)
            started = time.perf_counter()
            try:
                response = http.get(settings.open_meteo_archive_url, params=params)
                response.raise_for_status()
                hourly = archive_frame(response.json())
                matched = []
                for row in rows:
                    nearest = nearest_archive_row(hourly, row.timestamp_log)
                    if nearest is None:
                        continue
                    archive_hour, values = nearest
                    row.temperature_c_archive = _finite_or_none(
                        values.get("temperature_c_archive")
                    )
                    row.humidity_archive = _finite_or_none(values.get("humidity_archive"))
                    row.cloud_cover_pct_archive = _finite_or_none(
                        values.get("cloud_cover_pct_archive")
                    )
                    row.weather_archive_hour = archive_hour.isoformat()
                    row.weather_archive_query_id = query_id
                    matched.append(row)

                medians = {
                    "temperature": _median_or_none(
                        [row.temperature_c_archive for row in matched]
                    ),
                    "humidity": _median_or_none([row.humidity_archive for row in matched]),
                    "cloud": _median_or_none(
                        [row.cloud_cover_pct_archive for row in matched]
                    ),
                }
                complete = all(value is not None for value in medians.values())
                for row in rows:
                    row.temperature_c_archive_campaign_median = medians["temperature"]
                    row.humidity_archive_campaign_median = medians["humidity"]
                    row.cloud_cover_pct_archive_campaign_median = medians["cloud"]
                    row.weather_source_v5 = (
                        "manual_database"
                        if _manual_complete(row)
                        else "archive_campaign_median" if complete else "missing"
                    )
                    row.weather_missing_v5 = 0 if (_manual_complete(row) or complete) else 1
                session.add(ApiCallLog(
                    run_id=rows[0].run_id,
                    api_name="open_meteo_archive",
                    lat=lat,
                    lon=lon,
                    status="ok" if complete else "missing_field",
                    http_code=response.status_code,
                    latency_ms=(time.perf_counter() - started) * 1000,
                ))
                summaries.append({
                    "group": key,
                    "rows": len(rows),
                    "matched_hours": len(matched),
                    "query_id": query_id,
                    "campaign_medians": medians,
                    "status": "ok" if complete else "missing_field",
                })
            except (httpx.HTTPError, ValueError, TypeError) as exc:
                session.add(ApiCallLog(
                    run_id=rows[0].run_id,
                    api_name="open_meteo_archive",
                    lat=lat,
                    lon=lon,
                    status="failed",
                    error_message=str(exc)[:500],
                    latency_ms=(time.perf_counter() - started) * 1000,
                ))
                summaries.append({
                    "group": key,
                    "rows": len(rows),
                    "matched_hours": 0,
                    "query_id": query_id,
                    "status": "failed",
                    "error": str(exc),
                })
        if dry_run:
            session.rollback()
        else:
            session.commit()
    finally:
        if own_client:
            http.close()

    return {
        "dry_run": dry_run,
        "groups_processed": len(summaries),
        "rows_considered": len(measurements),
        "rows_skipped_without_timestamp": skipped_without_time,
        "groups": summaries,
    }


def _finite_or_none(value) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _median_or_none(values) -> float | None:
    valid = [_finite_or_none(value) for value in values]
    valid = [value for value in valid if value is not None]
    return float(np.median(valid)) if valid else None


def _manual_complete(row: Measurement) -> bool:
    return all(
        value is not None
        for value in (
            row.temperature_c_manual,
            row.humidity_manual,
            row.cloud_cover_pct_manual,
        )
    )
