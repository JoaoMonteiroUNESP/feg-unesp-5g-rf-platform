"""
Tests for the UX/safety fixes:
* /api/summary header badge data
* /api/enrich (re-enrich already-ingested measurements; only_missing semantics)
* Calibration.looks_synthetic() heuristic
* /api/sectors/calibration GET surfaces the synthetic warning
"""
from __future__ import annotations



from app.sectors.calibration import (
    Calibration, ControlPoint, fit_affine, save_calibration,
)


# --- /api/summary ----------------------------------------------------------
def test_summary_endpoint_empty(api_client, fresh_db, clean_calibration):
    body = api_client.get("/api/summary").json()
    assert body == {
        "n_measurements": 0,
        "n_measurements_raw": 0,
        "n_measurements_analytical": 0,
        "n_duplicates_removed": 0,
        "n_runs": 0,
        "n_campaigns": 0,
        "n_classified": 0,
        "n_enriched": 0,
        "n_weather_valid": 0,
        "weather_sources": {
            "manual_database": 0,
            "manual_notebook": 0,
            "archive_campaign_median": 0,
            "missing": 0,
        },
        "calibration": "uncalibrated",
    }


def test_summary_endpoint_after_upload(api_client, fresh_db, clean_calibration):
    # Minimal log (1 row) just to populate counts.
    header = (
        "Timestamp\tLatitude\tLongitude\tAccuracy\tAltitude\tSpeed\t"
        "Operatorname\tOperator\tCGI\tCellID\tLAC\tNetworkTech\tNetworkMode\t"
        "Level\tQual\tSNR\tCQI\tLTERSSI\tARFCN\tBAND\tBANDWIDTH\t"
        "DL_bitrate\tUL_bitrate\tDistance\tEVENT\tEVENTDETAILS\t"
        "PINGAVG\tPINGMIN\tPINGMAX\tPINGSTDEV\tPINGLOSS\t"
        "TESTDOWNLINK\tTESTUPLINK\tTESTDOWNLINKMAX\tTESTUPLINKMAX\t"
        "CSI_RSRP\tCSI_RSRQ\tCSI_SNR"
    )
    row = (
        "2026.05.04_09.00.00\t-23.20950\t-45.87650\t8\t580\t1.2\t"
        "VIVO\tVIVO\t724.06.123.4567\t1234567\t6\t5G\t5G NSA\t"
        "-90\t-12\t8\t10\t-65\t1850\tn78\t100\t"
        "12000\t1500\t120\tPERIODIC\t-\t"
        "32\t28\t40\t3\t0\t"
        "5000\t800\t8000\t1200\t"
        "-95\t-13\t9"
    )
    payload = (header + "\n" + row + "\n").encode("utf-8")
    r = api_client.post("/api/upload",
                        params={"enrich": "false", "campaign_id": "manha"},
                        files={"file": ("a.txt", payload, "text/plain")})
    assert r.json()["status"] == "success"
    body = api_client.get("/api/summary").json()
    assert body["n_measurements"] == 1
    assert body["n_measurements_analytical"] == 1
    assert body["n_runs"] == 1
    assert body["n_campaigns"] == 1
    # No enrich was requested → 0 enriched.
    assert body["n_enriched"] == 0


# --- looks_synthetic heuristic ---------------------------------------------
def test_looks_synthetic_flags_pytest_notes(synthetic_calibration):
    cal = synthetic_calibration
    # Fixture sets notes="pytest fixture"
    assert cal.looks_synthetic() is True


def test_looks_synthetic_flags_subcm_rms():
    # Build a calibration with realistic notes but absurdly tight RMS.
    cps = [
        ControlPoint("a", 0,   0,   -23.21,  -45.878),
        ControlPoint("b", 100, 0,   -23.21,  -45.877),
        ControlPoint("c", 0,   100, -23.209, -45.878),
        ControlPoint("d", 100, 100, -23.209, -45.877),
    ]
    cal = fit_affine(cps, notes="campo 2026-05-04, RTK")
    # These coords are linearly consistent → sub-mm RMS regardless of notes.
    assert cal.rms_m < 0.01
    assert cal.looks_synthetic() is True


def test_looks_synthetic_false_for_realistic_rms():
    # Hand-craft a Calibration with realistic-looking RMS and benign notes.
    cal = Calibration(
        p_lon=[1.0, 0.0, 0.0], p_lat=[0.0, 1.0, 0.0],
        rms_m=2.3, n_points=5,
        residuals_m=[], fitted_at="", project_version="",
        control_points=[], notes="campo 2026-05-04 RTK 6 pontos",
    )
    assert cal.looks_synthetic() is False


# --- /api/sectors/calibration warns when synthetic ------------------------
def test_calibration_status_warns_when_synthetic(api_client, clean_calibration,
                                                  synthetic_calibration):
    save_calibration(synthetic_calibration)
    body = api_client.get("/api/sectors/calibration").json()
    assert body["calibrated"] is True
    assert body["looks_synthetic"] is True
    assert "warning" in body
    assert "sintética" in body["warning"].lower() or \
           "sintetica" in body["warning"].lower()


# --- /api/enrich -----------------------------------------------------------
def test_enrich_existing_no_data(api_client, fresh_db):
    r = api_client.post("/api/enrich")
    assert r.status_code == 200
    assert "error" in r.json()


def test_enrich_existing_only_missing_skips_already_ok(api_client, fresh_db,
                                                       monkeypatch):
    """Insert a measurement that's already 'ok' on every status; enrich with
    only_missing=true must skip it without making any HTTP calls."""
    from app.db import SessionLocal, Measurement, IngestionRun
    s = SessionLocal()
    run = IngestionRun(filename="x", file_sha256="x" * 64,
                       file_size_bytes=10, log_variant="gnettrack_full",
                       delimiter="tab",
                       columns_detected="[]", columns_missing="[]")
    s.add(run); s.flush()
    m = Measurement(
        run_id=run.id, latitude=-23.21, longitude=-45.878, rsrp_dbm=-90,
        temperature_status="ok", building_status="ok", ndvi_status="ok",
    )
    s.add(m); s.commit(); s.close()

    # Fail loudly if any HTTP call is made (it shouldn't be).
    def _boom(*a, **k):
        raise AssertionError("enrich_point should not have been called for "
                             "an already-ok measurement")
    monkeypatch.setattr("app.api.enrich_point", _boom)

    body = api_client.post("/api/enrich",
                           params={"only_missing": "true"}).json()
    assert body["n_skipped_already_ok"] == 1
    assert body["n_enriched_this_call"] == 0
