"""Tests for the G-NetTrack log parser."""
from __future__ import annotations


import pandas as pd

from app.parser import parse_log_bytes, parse_log_text


def _full_log() -> str:
    """Minimal but variant-recognisable G-NetTrack 'full' log (3 rows)."""
    header = (
        "Timestamp\tLatitude\tLongitude\tAccuracy\tAltitude\tSpeed\t"
        "Operatorname\tOperator\tCGI\tCellID\tLAC\tNetworkTech\tNetworkMode\t"
        "Level\tQual\tSNR\tCQI\tLTERSSI\tARFCN\tBAND\tBANDWIDTH\t"
        "DL_bitrate\tUL_bitrate\tDistance\tEVENT\tEVENTDETAILS\t"
        "PINGAVG\tPINGMIN\tPINGMAX\tPINGSTDEV\tPINGLOSS\t"
        "TESTDOWNLINK\tTESTUPLINK\tTESTDOWNLINKMAX\tTESTUPLINKMAX\t"
        "CSI_RSRP\tCSI_RSRQ\tCSI_SNR"
    )
    rows = [
        # ok row
        "2026.05.03_08.34.46\t-23.20950\t-45.87650\t8\t580\t1.2\t"
        "VIVO\tVIVO\t724.06.123.4567\t1234567\t6\t5G\t5G NSA\t"
        "-90\t-12\t8\t10\t-65\t1850\tn78\t100\t"
        "12000\t1500\t120\tPERIODIC\t-\t"
        "32\t28\t40\t3\t0\t"
        "5000\t800\t8000\t1200\t"
        "-95\t-13\t9",
        # missing GPS accuracy → still pass; threshold default 50m
        "2026.05.03_08.34.50\t-23.20951\t-45.87651\t12\t581\t1.0\t"
        "VIVO\tVIVO\t724.06.123.4567\t1234567\t6\t4G\tLTE\t"
        "-100\t-15\t5\t8\t-72\t3500\tB7\t20\t"
        "8000\t1000\t-1\tPERIODIC\t-\t"
        "40\t35\t60\t5\t0\t"
        "3000\t500\t6000\t900\t"
        "-\t-\t-",
        # discarded row: GPS accuracy 200m (above threshold)
        "2026.05.03_08.34.55\t-23.20952\t-45.87649\t200\t580\t0.5\t"
        "VIVO\tVIVO\t724.06.123.4567\t1234567\t6\t4G\tLTE\t"
        "-105\t-16\t4\t6\t-75\t3500\tB7\t20\t"
        "5000\t800\t300\tPERIODIC\t-\t"
        "50\t45\t70\t6\t0\t"
        "2000\t300\t4000\t600\t"
        "-\t-\t-",
    ]
    return header + "\n" + "\n".join(rows) + "\n"


def test_parser_detects_full_variant_and_drops_low_gps():
    pr = parse_log_bytes(_full_log().encode("utf-8"), filename="sample.txt")
    assert pr.variant == "gnettrack_full"
    assert pr.delimiter == "tab"
    assert pr.rows_raw == 3
    assert pr.rows_dropped_gps == 1
    assert len(pr.df) == 2
    assert "rsrp_dbm" in pr.df.columns
    assert pr.file_sha256 and len(pr.file_sha256) == 64
    assert pr.file_size_bytes > 0


def test_parser_handles_null_markers_and_status():
    pr = parse_log_text(_full_log())
    df = pr.df
    # CSI is '-' on the 4G row → must surface as NaN with status missing_field.
    csi_status_col = "csi_rsrp_dbm_status"
    assert csi_status_col in df.columns
    rows_4g = df[df["network_tech"] == "4G"]
    assert all(rows_4g[csi_status_col] == "missing_field")
    assert all(pd.isna(rows_4g["csi_rsrp_dbm"]))


def test_parser_distance_minus_one_sentinel_becomes_nan_or_haversine():
    """Distance=-1 is a G-NetTrack sentinel; it must NOT be stored as -1."""
    pr = parse_log_text(_full_log())
    assert "distance_to_serving_m" in pr.df.columns
    assert not (pr.df["distance_to_serving_m"] == -1).any()


def test_parser_signal_rating_thresholds():
    pr = parse_log_text(_full_log())
    df = pr.df
    # Row 1 has RSRP=-90 → 'Bom'; row 2 has RSRP=-100 → 'Bom'.
    assert set(df["signal_rating"].unique()) <= {
        "Excelente", "Bom", "Satisfatório", "Ruim", "Péssimo", "Nulo",
    }


def test_parser_empty_input_returns_structured_result():
    pr = parse_log_text("")
    assert pr.df.empty
    assert pr.warnings  # must report something useful
