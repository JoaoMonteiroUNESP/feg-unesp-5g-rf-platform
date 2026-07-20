"""
Shared pytest fixtures.

The test suite isolates every test from the developer's real database by
overriding `FEG_DB_URL` to a temporary SQLite file BEFORE the app modules are
imported by the test process. The override happens here, in conftest.py, so
that pytest collection picks it up before any `from app...` import.
"""
from __future__ import annotations
import os
import tempfile
from pathlib import Path

import pytest


# --- Database isolation ----------------------------------------------------
# Must run before any `from app...` import. pytest imports conftest.py first,
# so this is the right place. We point the DB to a per-session temp file.
_TMP_DB = Path(tempfile.gettempdir()) / "feg_test_research.db"
_TMP_DB.unlink(missing_ok=True)
os.environ["FEG_DB_URL"] = f"sqlite:///{_TMP_DB.as_posix()}"
# Disable enrichment APIs by default so tests are hermetic.
os.environ.setdefault("FEG_GEE_PROJECT", "")


@pytest.fixture(scope="session", autouse=True)
def _init_db():
    """Create schema once per session against the temp DB."""
    from app.db import init_db
    init_db()
    yield
    # SQLite on Windows holds a file lock until the engine is disposed.
    try:
        from app.db import engine
        engine.dispose()
        _TMP_DB.unlink(missing_ok=True)
    except (OSError, PermissionError):
        pass            # leave the temp file; the OS will reap it


@pytest.fixture
def clean_calibration():
    """Remove any persisted calibration so tests start uncalibrated.

    Restores the developer's pre-existing calibration on teardown. If there
    was none, ALSO removes any calibration the test wrote — earlier we left
    a `notes='pytest'` calibration on disk that bled into manual runs.
    """
    from app.sectors.calibration import CALIBRATION_PATH
    backup = None
    if CALIBRATION_PATH.exists():
        backup = CALIBRATION_PATH.read_bytes()
        CALIBRATION_PATH.unlink()
    yield
    if backup is not None:
        CALIBRATION_PATH.write_bytes(backup)
    elif CALIBRATION_PATH.exists():
        # Test created a calibration; nuke it so it can't pollute dev state.
        try:
            CALIBRATION_PATH.unlink()
        except OSError:
            pass


@pytest.fixture
def fresh_db():
    """Clean every table between tests that exercise the DB."""
    from app.db import SessionLocal, Measurement, IngestionRun, ApiCallLog, NeighborCell
    s = SessionLocal()
    try:
        for cls in (NeighborCell, Measurement, ApiCallLog, IngestionRun):
            s.query(cls).delete()
        s.commit()
    finally:
        s.close()
    yield


@pytest.fixture
def synthetic_df():
    """A small (n=200) synthetic dataset for stats/ML tests."""
    import numpy as np
    import pandas as pd
    rng = np.random.default_rng(7)
    n = 200
    env = rng.choice(["edificado", "aberto", "arborizado"], size=n)
    sec = rng.choice(["S01", "S05", "S06", "S17"], size=n)
    tech = rng.choice(["4G", "5G"], size=n)
    band = rng.choice(["B3", "B7", "n78"], size=n)
    dist = rng.uniform(20, 400, size=n)
    rsrp = (-60 - 20 * np.log10(dist + 1)
            + np.where(env == "edificado", -8, np.where(env == "aberto", +2, 0))
            + np.where(tech == "5G", +3, 0)
            + rng.normal(0, 3, size=n))
    sinr = 5 + (rsrp + 100) / 5 + rng.normal(0, 2, size=n)
    ping = 30 + np.where(env == "edificado", 8, 0) + rng.gamma(2, 5, size=n)

    def rate(r):
        if r > -85:
            return "Excelente"
        if r > -95:
            return "Bom"
        if r > -105:
            return "Satisfatório"
        if r > -115:
            return "Ruim"
        if r > -125:
            return "Péssimo"
        return "Nulo"

    return pd.DataFrame({
        "environment_class": env, "sector_code": sec, "sector_name": sec,
        "network_tech": tech, "band": band,
        "distance_to_serving_m": dist,
        "speed_kmh": rng.uniform(0, 8, size=n),
        "altitude_m": rng.normal(580, 5, size=n),
        "gps_accuracy_m": rng.uniform(2, 30, size=n),
        "arfcn": rng.integers(1000, 5000, size=n),
        "n_neighbors": rng.integers(0, 8, size=n),
        "temperature_c": rng.normal(22, 2, size=n),
        "humidity": rng.normal(60, 10, size=n),
        "building_count": rng.integers(0, 30, size=n),
        "avg_building_height": rng.normal(8, 4, size=n),
        "tree_density_ndvi": rng.uniform(0, 1, size=n),
        "rsrp_dbm": rsrp, "sinr_db": sinr, "ping_avg_ms": ping,
        "signal_rating": [rate(x) for x in rsrp],
    })


@pytest.fixture
def synthetic_calibration():
    """Linear synthetic calibration around FEG-UNESP campus origin."""
    from app.sectors.calibration import ControlPoint, fit_affine
    ORIGIN_LAT, ORIGIN_LON = -23.21000, -45.87800
    DEG_PER_M_LAT = 1 / 111000.0
    DEG_PER_M_LON = 1 / (111000.0 * 0.92)
    cps = []
    for n_, x, y in [("CP_SW", 0, 0), ("CP_NE", 675, 415),
                     ("CP_NW", 0, 415), ("CP_SE", 675, 0),
                     ("CP_M", 337, 207)]:
        lat = ORIGIN_LAT + y * DEG_PER_M_LAT
        lon = ORIGIN_LON + x * DEG_PER_M_LON
        cps.append(ControlPoint(name=n_, x_local=x, y_local=y,
                                lat=lat, lon=lon))
    return fit_affine(cps, notes="pytest fixture")


@pytest.fixture
def api_client():
    """FastAPI TestClient bound to the isolated test DB."""
    from fastapi.testclient import TestClient
    from app.main import app
    return TestClient(app)
