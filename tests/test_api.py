"""Integration tests for the FastAPI surface."""
from __future__ import annotations




def test_health(api_client):
    r = api_client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "version" in body


def test_dashboard_html_renders(api_client):
    r = api_client.get("/")
    assert r.status_code == 200
    body = r.text
    # Spot-check key tabs and ML controls.
    for token in ("INGESTÃO", "ESTATÍSTICA", "ML · REGRESSÃO",
                  "CALIBRAÇÃO", "envSel", "regFeatures", "clsTarget",
                  "BASELINE", "choro_rsrp"):
        assert token in body, f"missing UI token: {token}"


def test_upload_filename_is_sanitized(api_client, fresh_db, clean_calibration):
    payload = (
        "Timestamp\tLatitude\tLongitude\tAccuracy\tLevel\n"
        "2026.05.04_09.00.00\t-23.2095\t-45.8765\t8\t-90\n"
    ).encode("utf-8")
    response = api_client.post(
        "/api/upload",
        params={"enrich": "false"},
        files={"file": ("../../unsafe name.txt", payload, "text/plain")},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "success"
    assert ".." not in body["raw_file_persisted"]
    assert "unsafe_name.txt" in body["raw_file_persisted"]


def test_upload_rejects_unsupported_suffix(api_client):
    response = api_client.post(
        "/api/upload",
        files={"file": ("payload.exe", b"not-a-log", "application/octet-stream")},
    )
    assert response.status_code == 415


def test_calibration_status_uncalibrated(api_client, clean_calibration):
    r = api_client.get("/api/sectors/calibration")
    assert r.status_code == 200
    body = r.json()
    assert body["calibrated"] is False


def test_post_calibration_rejects_few_points(api_client):
    r = api_client.post("/api/sectors/calibration", json={"control_points": []})
    assert r.status_code == 400


def test_post_calibration_accepts_synthetic(api_client, clean_calibration):
    cps = [
        {"name": "CP_SW", "x_local": 0,   "y_local": 0,   "lat": -23.21000, "lon": -45.87800},
        {"name": "CP_NE", "x_local": 675, "y_local": 415, "lat": -23.20627, "lon": -45.87139},
        {"name": "CP_NW", "x_local": 0,   "y_local": 415, "lat": -23.20627, "lon": -45.87800},
        {"name": "CP_SE", "x_local": 675, "y_local": 0,   "lat": -23.21000, "lon": -45.87139},
    ]
    r = api_client.post("/api/sectors/calibration",
                        json={"control_points": cps, "notes": "pytest"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["saved"] is True
    assert body["n_points"] == 4
    # RMS will be larger here than the perfectly-linear fixture (cps were built
    # with rounded ellipsoidal-degree distances), but should still be metres.
    assert body["rms_m"] < 100.0

    # GET should now report calibrated=True
    r2 = api_client.get("/api/sectors/calibration")
    assert r2.json()["calibrated"] is True


def test_sectors_geojson_endpoint(api_client):
    r = api_client.get("/api/sectors")
    assert r.status_code == 200
    fc = r.json()
    assert fc["type"] == "FeatureCollection"
    assert fc["properties"]["n_sectors"] == 21


def test_classify_endpoint_uncalibrated_or_calibrated(api_client):
    r = api_client.get("/api/sectors/classify",
                       params={"lat": -23.21, "lon": -45.878})
    assert r.status_code == 200
    body = r.json()
    assert "sector_code" in body  # may be None if uncalibrated
    assert "calibrated" in body


def test_aggregates_endpoint_validates_agg_param(api_client):
    r = api_client.get("/api/sectors/aggregates",
                       params={"metric": "rsrp_dbm", "agg": "bogus"})
    assert r.status_code == 422


def test_statistics_endpoint_no_data(api_client, fresh_db):
    r = api_client.get("/api/statistics",
                       params={"factor": "network_tech", "response": "rsrp_dbm"})
    assert r.status_code == 200
    body = r.json()
    assert "error" in body or body.get("n_total") == 0


def test_audit_runs_listing(api_client, fresh_db):
    r = api_client.get("/api/audit/runs")
    assert r.status_code == 200
    assert r.json() == []
