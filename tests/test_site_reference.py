"""Safety tests for the optional spatial-reference descriptor."""

from __future__ import annotations

import numpy as np

from app.site_estimate import fit_logdist


def test_site_reference_is_disabled_in_public_profile(api_client):
    body = api_client.get("/api/site/estimate").json()
    assert body["available"] is False
    assert "desativada" in body["message"]


def test_log_distance_rejects_implausible_slope():
    distance = np.linspace(20, 500, 60)
    rsrp = -40 - 10 * 25 * np.log10(distance)
    assert fit_logdist(distance, rsrp) is None


def test_log_distance_accepts_broadly_plausible_slope():
    distance = np.linspace(20, 500, 60)
    rsrp = -40 - 10 * 3 * np.log10(distance)
    result = fit_logdist(distance, rsrp)
    assert result is not None
    assert abs(result[1] - 3.0) < 0.01
