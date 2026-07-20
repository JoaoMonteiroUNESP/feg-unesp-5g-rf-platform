"""Tests for PCA, k-means, DBSCAN and grouped validation."""

from __future__ import annotations

import numpy as np

from app.analysis_data import deduplicate_measurements
from app.ml import regression_train
from app.unsupervised import unsupervised_analysis


def _with_route_metadata(df):
    work = df.copy()
    n = len(work)
    work["id"] = np.arange(1, n + 1)
    work["latitude"] = -22.803 + np.linspace(0, 0.002, n)
    work["longitude"] = -45.191 + np.linspace(0, 0.002, n)
    work["timestamp_log"] = np.datetime64("2026-07-01") + np.arange(n).astype(
        "timedelta64[s]"
    )
    work["campaign_id"] = [f"campaign-{i // 20:02d}" for i in range(n)]
    work["sector_code_effective"] = work["sector_code"]
    return work


def test_exact_duplicates_are_removed_without_changing_source(synthetic_df):
    work = _with_route_metadata(synthetic_df)
    duplicated = work.iloc[[0]].copy()
    duplicated["id"] = 9999
    duplicated["campaign_id"] = "duplicate-ingestion"
    combined = work._append(duplicated, ignore_index=True)

    clean, n_dropped, key = deduplicate_measurements(combined)

    assert len(combined) == len(work) + 1
    assert len(clean) == len(work)
    assert n_dropped == 1
    assert "campaign_id" not in key and "run_id" not in key


def test_duplicate_state_coalesces_complementary_qos(synthetic_df):
    work = _with_route_metadata(synthetic_df.iloc[:1].copy())
    first = work.iloc[[0]].copy()
    first["id"] = 1
    first["run_id"] = 10
    first["campaign_id"] = "original"
    first["data_connection_type"] = "M"
    first["test_dl_max_kbps"] = np.nan
    first["test_ul_max_kbps"] = 8000.0
    first["test_ul_max_status"] = "ok"

    second = first.copy()
    second["id"] = 2
    second["run_id"] = 11
    second["campaign_id"] = "same-log-reingested"
    second["test_dl_max_kbps"] = 45464.0
    second["test_dl_max_status"] = "ok"
    second["test_ul_max_kbps"] = np.nan
    second["test_ul_max_status"] = "missing"

    combined = first._append(second, ignore_index=True)
    original = combined.copy(deep=True)
    clean, n_dropped, _ = deduplicate_measurements(combined)

    assert n_dropped == 1
    assert len(clean) == 1
    assert clean.loc[0, "test_dl_max_kbps"] == 45464.0
    assert clean.loc[0, "test_dl_max_status"] == "ok"
    assert clean.loc[0, "test_ul_max_kbps"] == 8000.0
    assert clean.loc[0, "test_ul_max_status"] == "ok"
    assert combined.equals(original), "the source DataFrame must remain unchanged"


def test_wifi_invalid_qos_is_not_used_as_coalesce_source(synthetic_df):
    work = _with_route_metadata(synthetic_df.iloc[:1].copy())
    mobile = work.iloc[[0]].copy()
    mobile["id"] = 1
    mobile["data_connection_type"] = "M"
    mobile["test_dl_max_kbps"] = 12000.0
    mobile["test_dl_max_status"] = "ok"

    wifi = mobile.copy()
    wifi["id"] = 2
    wifi["data_connection_type"] = "WIFI"
    wifi["test_dl_max_kbps"] = 99999.0
    wifi["test_dl_max_status"] = "wifi_invalid"

    clean, _, _ = deduplicate_measurements(
        wifi._append(mobile, ignore_index=True)
    )

    assert len(clean) == 1
    assert clean.loc[0, "data_connection_type"] == "M"
    assert clean.loc[0, "test_dl_max_kbps"] == 12000.0
    assert clean.loc[0, "test_dl_max_status"] == "ok"


def test_unsupervised_returns_all_promised_methods(synthetic_df):
    work = _with_route_metadata(synthetic_df)
    result = unsupervised_analysis(
        work,
        features=[
            "rsrp_dbm",
            "sinr_db",
            "altitude_m",
            "temperature_c",
            "humidity",
            "building_count",
            "tree_density_ndvi",
        ],
    )

    assert "error" not in result, result
    assert result["pca"]["n_components_for_90_pct"] >= 2
    assert 2 <= result["kmeans"]["selected_k"] <= 6
    assert result["kmeans"]["profiles"]
    assert result["dbscan"]["candidates"]
    assert result["audit"]["rows_complete_case"] == len(work)
    assert "no value imputation" in result["audit"]["missingness_rule"]


def test_grouped_regression_keeps_campaigns_intact(synthetic_df):
    work = _with_route_metadata(synthetic_df)
    result = regression_train(
        work,
        features=["distance_to_serving_m", "altitude_m", "environment_class"],
        target="rsrp_dbm",
        group_by="campaign_id",
    )

    assert "error" not in result, result
    assert result["group_by"] == "campaign_id"
    assert result["n_groups"] == 10
    assert result["cv_scheme"].startswith("GroupKFold")
    assert "LinearRegression" in {model["name"] for model in result["models"]}
