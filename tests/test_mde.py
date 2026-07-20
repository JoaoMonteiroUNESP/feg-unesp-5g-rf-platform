"""Tests for the post-hoc Minimum Detectable Effect helper."""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.stats import minimum_detectable_effect


def test_mde_reports_one_row_per_eligible_group(synthetic_df):
    out = minimum_detectable_effect(
        synthetic_df,
        factor_col="environment_class",
        response_col="rsrp_dbm",
    )
    assert "rows" in out
    levels = {row["level"] for row in out["rows"]}
    assert levels == {"aberto", "arborizado", "edificado"}
    for row in out["rows"]:
        assert 0.0 < row["min_detectable_d"] < 1.0


def test_mde_floor_increases_for_small_groups():
    rng = np.random.default_rng(0)
    big = pd.DataFrame({
        "f": ["a"] * 60 + ["b"] * 60,
        "y": rng.normal(0, 1, 120),
    })
    small = pd.DataFrame({
        "f": ["a"] * 12 + ["b"] * 12,
        "y": rng.normal(0, 1, 24),
    })
    big_d = minimum_detectable_effect(big, "f", "y")["rows"][0]["min_detectable_d"]
    small_d = minimum_detectable_effect(small, "f", "y")["rows"][0]["min_detectable_d"]
    assert small_d > big_d * 1.3, (small_d, big_d)


def test_mde_returns_error_when_one_group():
    frame = pd.DataFrame({"f": ["a"] * 10, "y": list(range(10))})
    out = minimum_detectable_effect(frame, "f", "y")
    assert "error" in out
