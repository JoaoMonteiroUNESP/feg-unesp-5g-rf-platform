"""
Tests for the geographic cache bucketing in app.enrichment.

Hermetic — no real HTTP. We verify:
  * per-API bucket sizes coalesce nearby measurements onto the same key;
  * different APIs at the same lat/lon produce different keys;
  * a tiny cosine-of-latitude lookalike is honoured (longitude buckets at
    -23° latitude are larger in *degrees* than latitude buckets of the same
    metric size).
"""
from __future__ import annotations

import math


from app.enrichment import (
    BUCKET_M_PER_API, DEFAULT_BUCKET_M,
    _cache_key, _quantize_latlon,
)


# --- Quantization basics ---------------------------------------------------
def test_quantize_collapses_points_within_bucket():
    """Two points 5 m apart must hash to the same 25 m bucket."""
    lat0, lon0 = -23.21000, -45.87800
    # Move ~5 m east at this latitude
    m_per_deg_lon = 111_000.0 * math.cos(math.radians(lat0))
    lon1 = lon0 + 5.0 / m_per_deg_lon
    q0 = _quantize_latlon(lat0, lon0, 25.0)
    q1 = _quantize_latlon(lat0, lon1, 25.0)
    assert q0 == q1


def test_quantize_separates_points_across_bucket_boundary():
    lat0, lon0 = -23.21000, -45.87800
    # Move ~50 m east — crosses two 25 m cells.
    m_per_deg_lon = 111_000.0 * math.cos(math.radians(lat0))
    lon_far = lon0 + 50.0 / m_per_deg_lon
    q0 = _quantize_latlon(lat0, lon0, 25.0)
    qN = _quantize_latlon(lat0, lon_far, 25.0)
    assert q0 != qN


def test_quantize_uses_latitude_cosine_for_longitude_grid():
    """A 25 m bucket at -23° latitude must use a wider longitude step than
    the latitude step, because metre-per-degree is smaller in longitude."""
    qlat_step = 25.0 / 111_000.0
    qlon_step = 25.0 / (111_000.0 * math.cos(math.radians(-23.21)))
    assert qlon_step > qlat_step
    # _quantize must reflect this in its outputs.
    a = _quantize_latlon(-23.21000, -45.87800, 25.0)
    b = _quantize_latlon(-23.21000, -45.87800 + qlon_step / 2, 25.0)
    # Half-bucket east → still same cell.
    assert a == b


# --- Cache key behaviour ---------------------------------------------------
def test_open_meteo_collapses_whole_campus_to_one_key():
    """The 5 km bucket means weather across the entire FEG campus shares one
    cache row — we should pay for ONE call, not one per measurement."""
    # 200 measurements scattered within ~700 m × 400 m of the campus.
    keys = set()
    for d_lat in (-200, -100, 0, 100, 200):
        for d_lon in (-300, -100, 0, 100, 300):
            lat = -23.21000 + d_lat / 111_000.0
            lon = -45.87800 + d_lon / (111_000.0 * math.cos(math.radians(-23.21)))
            keys.add(_cache_key("open_meteo", lat, lon))
    assert len(keys) == 1, f"expected 1 bucket, got {len(keys)}: {keys}"


def test_ndvi_uses_finer_bucket_than_overpass():
    """NDVI has Sentinel-2 native ~10 m resolution; bucket must be finer than
    the 25 m one used for Overpass."""
    assert BUCKET_M_PER_API["gee_ndvi"] < BUCKET_M_PER_API["overpass"]


def test_different_apis_produce_different_keys_at_same_point():
    lat, lon = -23.21000, -45.87800
    keys = {api: _cache_key(api, lat, lon) for api in BUCKET_M_PER_API}
    # All four keys must be distinct (different prefix + bucket size).
    assert len(set(keys.values())) == len(keys)


def test_unknown_api_falls_back_to_default_bucket():
    k = _cache_key("brand_new_api", -23.21, -45.878)
    assert f"b{int(DEFAULT_BUCKET_M)}m" in k


def test_cache_key_is_deterministic():
    """Same inputs → identical key (no random salt, no clock dependence)."""
    a = _cache_key("overpass", -23.21000123, -45.87800456)
    b = _cache_key("overpass", -23.21000123, -45.87800456)
    assert a == b


# --- Coalescing rate ------------------------------------------------------
def test_coalescing_rate_for_clustered_measurements():
    """500 measurements within a 100 m × 100 m sector → at most a handful of
    distinct buckets per API. Empirical reality test: with 25 m bucket the
    upper bound is 25 cells (5×5)."""
    import random
    rng = random.Random(42)
    lat0, lon0 = -23.21000, -45.87800
    m_per_deg_lat = 111_000.0
    m_per_deg_lon = 111_000.0 * math.cos(math.radians(lat0))

    pts = []
    for _ in range(500):
        # uniform in ±50 m around the centre
        dlat = rng.uniform(-50, 50) / m_per_deg_lat
        dlon = rng.uniform(-50, 50) / m_per_deg_lon
        pts.append((lat0 + dlat, lon0 + dlon))

    n_overpass = len({_cache_key("overpass",       la, lo) for la, lo in pts})
    n_meteo    = len({_cache_key("open_meteo",     la, lo) for la, lo in pts})
    n_ndvi     = len({_cache_key("gee_ndvi",       la, lo) for la, lo in pts})

    # Theoretical max cells for span s and bucket b is (ceil(s/b)+1)² — the
    # +1 accounts for boundary cells when points fall near cell edges.
    # span = 100 m here.
    # Overpass: 25 m bucket → ≤ (4+1)² = 25.
    assert n_overpass <= 25, n_overpass
    # Meteo: 5 km bucket easily covers 100 m → exactly 1 bucket.
    assert n_meteo == 1
    # NDVI: 15 m bucket → ≤ (7+1)² = 64.
    assert n_ndvi <= 64, n_ndvi
    # Massively better than the naive ~500 distinct keys.
    assert n_overpass < 500
