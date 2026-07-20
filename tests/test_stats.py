"""Tests for the scientific-statistics layer."""
from __future__ import annotations

import pandas as pd

from app.stats import (
    anova_by_factor, anova_robust, pairwise_hedges_g,
    pearson_with_fdr, summary_by_sector, summary_by_environment,
    stratified_summary,
)


def test_anova_by_factor_detects_real_effect(synthetic_df):
    out = anova_by_factor(synthetic_df, "environment_class", "rsrp_dbm")
    assert "F" in out and out["p"] < 0.001
    assert out["n_groups"] == 3
    assert any(t["reject"] for t in out["tukey_hsd"])


def test_anova_by_factor_rejects_one_group():
    df = pd.DataFrame({"f": ["a"] * 10, "y": list(range(10))})
    out = anova_by_factor(df, "f", "y")
    assert "error" in out


def test_anova_robust_reports_three_tests(synthetic_df):
    out = anova_robust(synthetic_df, "environment_class", "rsrp_dbm")
    tests = out["tests"]
    assert {"anova_classic", "anova_welch", "kruskal_wallis"} <= tests.keys()
    # Strong synthetic effect → all three must reject H0.
    assert tests["anova_classic"]["p"] < 0.001
    assert tests["anova_welch"]["p"] < 0.001
    assert tests["kruskal_wallis"]["p"] < 0.001
    assert "omega_squared" in out["effect_size"]
    assert "recommendation" in out and isinstance(out["recommendation"], str)


def test_anova_robust_returns_error_one_group():
    df = pd.DataFrame({"f": ["a"] * 10, "y": list(range(10))})
    out = anova_robust(df, "f", "y")
    assert "error" in out


def test_hedges_g_pairs_and_directions(synthetic_df):
    out = pairwise_hedges_g(synthetic_df, "environment_class", "rsrp_dbm")
    assert len(out["pairs"]) == 3
    # Edificado vs aberto must be the largest |g| (biggest mean gap).
    g_by_pair = {(p["group1"], p["group2"]): p["g"] for p in out["pairs"]}
    abs_g = {k: abs(v) for k, v in g_by_pair.items()}
    biggest = max(abs_g, key=abs_g.get)
    assert "edificado" in biggest and "aberto" in biggest


def test_pearson_fdr_basic_shape(synthetic_df):
    out = pearson_with_fdr(synthetic_df,
                           ["rsrp_dbm", "sinr_db", "ping_avg_ms",
                            "distance_to_serving_m", "altitude_m"])
    n = len(out["cols"])
    assert len(out["r"]) == n and len(out["r"][0]) == n
    assert out["method"].startswith("Benjamini")


def test_pearson_fdr_pairwise_is_used():
    """A column with only NaNs must not poison the rest."""
    df = pd.DataFrame({"a": [1.0, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                       "b": [10.0, 9, 8, 7, 6, 5, 4, 3, 2, 1],
                       "c": [float("nan")] * 10})
    out = pearson_with_fdr(df, ["a", "b", "c"])
    assert out["r"][0][1] is not None
    assert out["r"][0][2] is None


def test_summary_by_sector(synthetic_df):
    out = summary_by_sector(synthetic_df)
    assert out["n_sectors_with_data"] == 4
    assert all("metrics" in s and "rsrp_dbm" in s["metrics"]
               for s in out["sectors"])


def test_summary_by_sector_missing_column():
    out = summary_by_sector(pd.DataFrame({"x": [1, 2, 3]}))
    assert "error" in out


def test_summary_by_environment(synthetic_df):
    out = summary_by_environment(synthetic_df)
    envs = [e["environment_class"] for e in out["environments"]]
    assert set(envs) == {"aberto", "arborizado", "edificado"}


def test_stratified_summary(synthetic_df):
    out = stratified_summary(synthetic_df)
    # Must have one entry per network_tech.
    assert set(out.keys()) == {"4G", "5G"}
