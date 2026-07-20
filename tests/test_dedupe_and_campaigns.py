"""
Tests for SHA-256 dedupe + campaign_id propagation + temporal coverage report.
These exercise the upload pipeline end-to-end via TestClient against the
isolated test DB defined in conftest.
"""
from __future__ import annotations




# --- A minimal G-NetTrack 'full' log usable by the upload route ------------
HEADER = (
    "Timestamp\tLatitude\tLongitude\tAccuracy\tAltitude\tSpeed\t"
    "Operatorname\tOperator\tCGI\tCellID\tLAC\tNetworkTech\tNetworkMode\t"
    "Level\tQual\tSNR\tCQI\tLTERSSI\tARFCN\tBAND\tBANDWIDTH\t"
    "DL_bitrate\tUL_bitrate\tDistance\tEVENT\tEVENTDETAILS\t"
    "PINGAVG\tPINGMIN\tPINGMAX\tPINGSTDEV\tPINGLOSS\t"
    "TESTDOWNLINK\tTESTUPLINK\tTESTDOWNLINKMAX\tTESTUPLINKMAX\t"
    "CSI_RSRP\tCSI_RSRQ\tCSI_SNR"
)
ROW_OK = (
    "{ts}\t-23.20950\t-45.87650\t8\t580\t1.2\t"
    "VIVO\tVIVO\t724.06.123.4567\t1234567\t6\t5G\t5G NSA\t"
    "-90\t-12\t8\t10\t-65\t1850\tn78\t100\t"
    "12000\t1500\t120\tPERIODIC\t-\t"
    "32\t28\t40\t3\t0\t"
    "5000\t800\t8000\t1200\t"
    "-95\t-13\t9"
)


def _log_bytes(timestamps: list[str]) -> bytes:
    rows = [ROW_OK.format(ts=ts) for ts in timestamps]
    return ("\r\n".join([HEADER, *rows]) + "\r\n").encode("utf-8")


def _upload(client, payload: bytes, **params):
    return client.post(
        "/api/upload",
        params={"enrich": "false", **params},
        files={"file": ("sample.txt", payload, "text/plain")},
    )


def test_first_upload_succeeds(api_client, fresh_db):
    r = _upload(api_client, _log_bytes(["2026.05.04_09.00.00"]),
                campaign_id="manha-test")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "success"
    assert body["campaign_id"] == "manha-test"
    assert body["rows_inserted"] == 1


def test_dedupe_returns_already_ingested(api_client, fresh_db):
    payload = _log_bytes(["2026.05.04_09.00.00"])
    r1 = _upload(api_client, payload, campaign_id="c1")
    assert r1.json()["status"] == "success"

    r2 = _upload(api_client, payload, campaign_id="c1")
    body = r2.json()
    assert body["status"] == "already_ingested"
    assert body["run_id"] == r1.json()["run_id"]
    # Existing run's campaign_id is preserved (not overwritten by repeat upload).
    assert body["campaign_id"] == "c1"


def test_dedupe_force_reingests(api_client, fresh_db):
    payload = _log_bytes(["2026.05.04_09.00.00"])
    r1 = _upload(api_client, payload, campaign_id="c1")
    r2 = _upload(api_client, payload, campaign_id="c1", force="true")
    assert r2.json()["status"] == "success"
    assert r2.json()["run_id"] != r1.json()["run_id"]


def test_campaign_id_is_persisted_to_measurements(api_client, fresh_db):
    _upload(api_client, _log_bytes(["2026.05.04_09.00.00",
                                    "2026.05.04_09.00.10"]),
            campaign_id="manha-2026-05-04")
    pts = api_client.get("/api/points").json()
    assert pts and all(p["campaign_id"] == "manha-2026-05-04" for p in pts)


def test_campaign_id_blank_becomes_null(api_client, fresh_db):
    _upload(api_client, _log_bytes(["2026.05.04_09.00.00"]), campaign_id="")
    pts = api_client.get("/api/points").json()
    assert pts and pts[0]["campaign_id"] is None


def test_campaigns_endpoint_lists_distinct_campaigns(api_client, fresh_db):
    _upload(api_client, _log_bytes(["2026.05.04_09.00.00"]),
            campaign_id="manha")
    _upload(api_client, _log_bytes(["2026.05.04_18.00.00"]),
            campaign_id="tarde")
    body = api_client.get("/api/campaigns").json()
    ids = {c["campaign_id"] for c in body["campaigns"]}
    assert {"manha", "tarde"} <= ids
    # Each campaign must report at least one run + one measurement.
    for c in body["campaigns"]:
        if c["campaign_id"] in {"manha", "tarde"}:
            assert c["n_runs"] >= 1
            assert c["n_measurements"] >= 1


def test_temporal_coverage_flags_single_hour(api_client, fresh_db):
    # Two uploads, both at 09:xx → unclassified bucket should report
    # n_distinct_hours == 1.
    _upload(api_client, _log_bytes(["2026.05.04_09.00.00",
                                    "2026.05.04_09.05.00"]),
            campaign_id="manha")
    body = api_client.get("/api/sectors/temporal_coverage").json()
    assert "sectors" in body
    s = body["sectors"][0]
    assert s["n_measurements"] >= 2
    assert s["n_distinct_hours"] == 1
    assert s["hours"] == [9]


def test_statistics_filter_by_campaign(api_client, fresh_db):
    _upload(api_client, _log_bytes(["2026.05.04_09.00.00"]), campaign_id="A")
    _upload(api_client, _log_bytes(["2026.05.04_18.00.00"]), campaign_id="B")
    r = api_client.get("/api/statistics",
                       params={"factor": "network_tech", "response": "rsrp_dbm",
                               "campaign_id": "A"})
    body = r.json()
    # n_total scoped to campaign A only.
    if "n_total" in body:
        assert body["n_total"] == 1
