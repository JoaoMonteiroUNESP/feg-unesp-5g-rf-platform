"""Tests for the ML pipelines (regression + classification)."""
from __future__ import annotations

import pandas as pd
import pytest

from app.ml import (
    LEAKY,
    regression_train, classification_train,
)


pytestmark = pytest.mark.slow


def test_regression_includes_baselines_and_models(synthetic_df):
    out = regression_train(synthetic_df, target="rsrp_dbm")
    names = [m["name"] for m in out["models"]]
    assert "Baseline_Mean" in names and "Baseline_Median" in names
    assert {"RandomForest", "XGBoost", "SVR"} <= set(names)
    # Real models must beat baseline R².
    base = next(m for m in out["models"] if m["name"] == "Baseline_Mean")
    rf   = next(m for m in out["models"] if m["name"] == "RandomForest")
    assert rf["summary"]["R2"]["mean"] > base["summary"]["R2"]["mean"]


def test_regression_blocks_leaky_features(synthetic_df):
    out = regression_train(synthetic_df, target="rsrp_dbm")
    assert all(c not in out["features_used"] for c in LEAKY)
    assert "rsrp_dbm" not in out["features_used"]


def test_regression_target_excluded_from_features(synthetic_df):
    """If `environment_class` is the target, it must not be one-hot in X."""
    out = regression_train(synthetic_df, target="ping_avg_ms",
                           features=["distance_to_serving_m", "ping_avg_ms"])
    assert "ping_avg_ms" not in out["features_used"]


def test_regression_insufficient_n_returns_error():
    df = pd.DataFrame({"distance_to_serving_m": list(range(10)),
                       "rsrp_dbm": list(range(10))})
    out = regression_train(df, target="rsrp_dbm",
                           features=["distance_to_serving_m"])
    assert "error" in out


def test_classification_clean_target_environment_class(synthetic_df):
    out = classification_train(synthetic_df, target="environment_class")
    assert "environment_class" not in out["features_used"]
    names = [m["name"] for m in out["models"]]
    assert "Baseline_MostFrequent" in names
    assert {"RandomForest", "XGBoost", "SVC"} <= set(names)


def test_classification_returns_confusion_matrix(synthetic_df):
    # Use environment_class as target — synthetic_df has 3 balanced classes;
    # signal_rating can have a too-small 'Excelente' bucket depending on RNG.
    out = classification_train(synthetic_df, target="environment_class")
    assert "models" in out, out
    rf = next(m for m in out["models"] if m["name"] == "RandomForest")
    cm = rf["confusion_matrix_mean"]
    labels = rf["confusion_matrix_labels"]
    assert len(cm) == len(labels) and len(cm[0]) == len(labels)
    # Each class entry has precision/recall/f1.
    for cls in labels:
        block = rf["classification_report"][cls]
        assert {"precision", "recall", "f1"} <= block.keys()


def test_classification_refuses_when_class_too_small():
    """Each class needs ≥cv_splits samples."""
    df = pd.DataFrame({
        "distance_to_serving_m": list(range(30)),
        "signal_rating": ["A"] * 28 + ["B"] * 2,
    })
    out = classification_train(df, target="signal_rating",
                               features=["distance_to_serving_m"])
    assert "error" in out
    assert "class_counts" in out


def test_permutation_importance_present_for_real_models(synthetic_df):
    out = regression_train(synthetic_df, target="rsrp_dbm")
    rf = next(m for m in out["models"] if m["name"] == "RandomForest")
    assert rf["permutation_importance"] is not None
    # Distance must be the strongest feature in this synthetic process.
    top = next(iter(rf["permutation_importance"]))
    assert top.startswith("distance_to_serving_m")
