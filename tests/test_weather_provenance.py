"""Regression tests for the v5 historical-weather provenance policy."""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.derivations import add_effective_columns


def test_historical_weather_never_falls_back_to_legacy_current_fields():
    frame = pd.DataFrame(
        {
            "temperature_c": [20.1],
            "humidity": [72.0],
            "cloud_cover_pct": [93.0],
            "temperature_c_manual": [np.nan],
            "humidity_manual": [np.nan],
            "cloud_cover_pct_manual": [np.nan],
        }
    )
    result = add_effective_columns(frame)
    assert result["temperature_c_eff"].isna().all()
    assert result["humidity_eff"].isna().all()
    assert result["cloud_cover_pct_eff"].isna().all()
    assert result.loc[0, "weather_source_eff"] == "missing"
    assert bool(result.loc[0, "weather_missing_eff"])


def test_manual_notebook_has_precedence_over_campaign_archive():
    frame = pd.DataFrame(
        {
            "temperature_c_manual": [18.0],
            "humidity_manual": [83.0],
            "cloud_cover_pct_manual": [20.0],
            "temperature_c_archive": [17.4],
            "humidity_archive": [71.0],
            "cloud_cover_pct_archive": [62.0],
            "temperature_c_archive_campaign_median": [17.8],
            "humidity_archive_campaign_median": [69.0],
            "cloud_cover_pct_archive_campaign_median": [55.0],
            "manual_weather_provenance_v5": ["manual_notebook_user_declaration"],
        }
    )
    result = add_effective_columns(frame)
    assert result.loc[0, "temperature_c_eff"] == 18.0
    assert result.loc[0, "humidity_eff"] == 83.0
    assert result.loc[0, "cloud_cover_pct_eff"] == 20.0
    assert result.loc[0, "weather_source_eff"] == "manual_notebook"
    assert not bool(result.loc[0, "weather_missing_eff"])


def test_archive_fallback_uses_campaign_summary_not_point_hour():
    frame = pd.DataFrame(
        {
            "temperature_c_manual": [np.nan],
            "humidity_manual": [np.nan],
            "cloud_cover_pct_manual": [np.nan],
            "temperature_c_archive": [12.0],
            "humidity_archive": [90.0],
            "cloud_cover_pct_archive": [100.0],
            "temperature_c_archive_campaign_median": [16.5],
            "humidity_archive_campaign_median": [75.0],
            "cloud_cover_pct_archive_campaign_median": [60.0],
        }
    )
    result = add_effective_columns(frame)
    assert result.loc[0, "temperature_c_eff"] == 16.5
    assert result.loc[0, "humidity_eff"] == 75.0
    assert result.loc[0, "cloud_cover_pct_eff"] == 60.0
    assert result.loc[0, "weather_source_eff"] == "archive_campaign_median"


def test_partial_manual_override_is_reported_as_mixed_source():
    frame = pd.DataFrame({
        "temperature_c_manual": [18.0],
        "humidity_manual": [None],
        "cloud_cover_pct_manual": [None],
        "temperature_c_archive_campaign_median": [17.5],
        "humidity_archive_campaign_median": [70.0],
        "cloud_cover_pct_archive_campaign_median": [40.0],
    })
    result = add_effective_columns(frame)
    assert result.loc[0, "temperature_c_eff"] == 18.0
    assert result.loc[0, "humidity_eff"] == 70.0
    assert result.loc[0, "weather_source_eff"] == "mixed_manual_archive"
