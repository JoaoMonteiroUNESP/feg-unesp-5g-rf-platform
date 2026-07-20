"""Tests for reproducible Open-Meteo Archive enrichment."""

from __future__ import annotations

from datetime import datetime

import httpx

from app.db import IngestionRun, Measurement, SessionLocal
from app.weather_archive import archive_frame, backfill_archive_weather, nearest_archive_row


PAYLOAD = {
    "hourly": {
        "time": ["2026-07-10T09:00", "2026-07-10T10:00", "2026-07-10T11:00"],
        "temperature_2m": [17.0, 18.0, 19.0],
        "relative_humidity_2m": [85, 83, 80],
        "cloud_cover": [30, 20, 10],
    }
}


def test_archive_payload_and_nearest_hour():
    frame = archive_frame(PAYLOAD)
    nearest = nearest_archive_row(frame, datetime(2026, 7, 10, 10, 24))
    assert nearest is not None
    timestamp, row = nearest
    assert timestamp.hour == 10
    assert row["temperature_c_archive"] == 18.0


def test_backfill_persists_campaign_medians(fresh_db):
    session = SessionLocal()
    run = IngestionRun(
        filename="synthetic.txt",
        file_sha256="weather-test",
        campaign_id="campaign-a",
    )
    session.add(run)
    session.flush()
    for hour in (9, 10, 11):
        session.add(Measurement(
            run_id=run.id,
            campaign_id="campaign-a",
            timestamp_log=datetime(2026, 7, 10, hour, 5),
            latitude=-23.2,
            longitude=-45.8,
            rsrp_dbm=-90,
        ))
    session.commit()

    def handler(request: httpx.Request) -> httpx.Response:
        assert "archive-api.open-meteo.com" in str(request.url)
        return httpx.Response(200, json=PAYLOAD)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    try:
        result = backfill_archive_weather(session, client=client)
    finally:
        client.close()
    assert result["groups_processed"] == 1
    assert result["groups"][0]["status"] == "ok"

    rows = session.query(Measurement).order_by(Measurement.timestamp_log).all()
    assert [row.temperature_c_archive for row in rows] == [17.0, 18.0, 19.0]
    assert all(row.temperature_c_archive_campaign_median == 18.0 for row in rows)
    assert all(row.weather_source_v5 == "archive_campaign_median" for row in rows)
    session.close()
