"""Tests for the affine calibration + sector classifier."""
from __future__ import annotations


import pytest

from app.sectors.calibration import (
    Calibration, ControlPoint, fit_affine,
    save_calibration, load_calibration,
    transform_local_to_wgs84, transform_wgs84_to_local,
)
from app.sectors.classifier import SectorClassifier
from app.sectors.loader import build_sector_geojson, load_local_sectors


def test_local_sectors_load_21():
    secs = load_local_sectors()
    assert len(secs) == 21
    assert all({"id", "xmin", "ymin", "xmax", "ymax"} <= s.keys() for s in secs)


def test_uncalibrated_geojson_has_no_geometry():
    fc = build_sector_geojson(calibration=None)
    assert fc["properties"]["calibrated"] is False
    assert fc["properties"]["n_sectors"] == 21
    assert all(f["geometry"] is None for f in fc["features"])


def test_uncalibrated_classifier_returns_none(synthetic_calibration):
    fc = build_sector_geojson(calibration=None)
    cls = SectorClassifier(fc)
    assert cls.calibrated is False
    hit = cls.classify(-23.21, -45.878)
    assert hit.sector_code is None
    assert hit.environment_class is None


def test_fit_affine_roundtrip_and_low_rms(synthetic_calibration):
    cal = synthetic_calibration
    assert cal.n_points == 5
    # Linear control points should fit exactly (sub-millimetre).
    assert cal.rms_m < 1e-3

    # Forward + inverse must round-trip.
    lat, lon = transform_local_to_wgs84(cal, 100.0, 200.0)
    x_back, y_back = transform_wgs84_to_local(cal, lat, lon)
    assert abs(x_back - 100) < 1e-3
    assert abs(y_back - 200) < 1e-3


def test_fit_affine_rejects_few_points():
    with pytest.raises(ValueError):
        fit_affine([ControlPoint("a", 0, 0, -23.0, -45.0),
                    ControlPoint("b", 1, 0, -23.0, -45.0)])


def test_save_and_load_calibration(synthetic_calibration, tmp_path):
    path = tmp_path / "cal.json"
    save_calibration(synthetic_calibration, path=path)
    loaded = load_calibration(path=path)
    assert isinstance(loaded, Calibration)
    assert loaded.rms_m == pytest.approx(synthetic_calibration.rms_m, abs=1e-12)


def test_load_calibration_returns_none_when_absent(tmp_path):
    assert load_calibration(path=tmp_path / "missing.json") is None


def test_classifier_classifies_known_interior_point(synthetic_calibration):
    fc = build_sector_geojson(calibration=synthetic_calibration)
    cls = SectorClassifier(fc)
    assert cls.calibrated is True
    assert cls.n_sectors == 21
    # FEGÃO bbox in legend is (32.5, 25, ...). Use an interior point.
    lat, lon = transform_local_to_wgs84(synthetic_calibration, 65.0, 50.0)
    hit = cls.classify(lat, lon)
    assert hit.sector_code == "S05"
    assert hit.environment_class == "edificado"


def test_classifier_outside_returns_none(synthetic_calibration):
    fc = build_sector_geojson(calibration=synthetic_calibration)
    cls = SectorClassifier(fc)
    # Far below the SW corner, well outside any sector.
    lat, lon = transform_local_to_wgs84(synthetic_calibration, -100.0, -100.0)
    assert cls.classify(lat, lon).sector_code is None


def test_classifier_handles_none_input(synthetic_calibration):
    fc = build_sector_geojson(calibration=synthetic_calibration)
    cls = SectorClassifier(fc)
    assert cls.classify(None, -45.878).sector_code is None
    assert cls.classify(-23.21, None).sector_code is None
