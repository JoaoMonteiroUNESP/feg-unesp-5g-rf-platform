"""
External enrichment APIs.

Hard rules
----------
* Every field returned has both a value (or None) and a status.
  Possible statuses: 'ok', 'missing_field', 'api_failed', 'cached', 'skipped'.
* When the upstream API returns 200 but does not contain the expected key,
  the field is `(None, 'missing_field')`. NEVER substituted with a constant.
* Every call writes one row to ApiCallLog with HTTP code + latency.
* Cache is persistent (table geo_cache, see BUCKET_M_PER_API below).

Per-API geographic bucketing
----------------------------
Two adjacent measurements 5 m apart used to trigger two API calls because the
old cache key rounded coordinates to 4 decimals (~11 m) for *every* API. That
was wasted effort: weather is invariant on the kilometre scale, NDVI has 10 m
native resolution from Sentinel-2, and Overpass already queries a 50 m radius.
We now quantise lat/lon per-API so nearby measurements coalesce to the same
cache key:

    open_meteo       5000 m  →  one call per campaign (campus is ~700 m wide)
    open_meteo_elev    25 m
    overpass           25 m  (smaller than the 50 m query radius is wasteful)
    gee_ndvi           15 m  (≈ Sentinel-2 native pixel)

For 5.000 measurements clustered in the FEG campus this typically reduces the
upstream call count by 10× to 50× while preserving spatial fidelity at the
relevant scale per API.
"""
from __future__ import annotations
import asyncio
import json
import logging
import math
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import httpx
from sqlalchemy.orm import Session

from app.config import settings
from app.db import GeoCache
from app.audit import log_api_call


log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Earth Engine bootstrap (optional)
# ---------------------------------------------------------------------------
_GEE_OK = False
try:
    if settings.gee_project:
        import ee  # type: ignore
        try:
            ee.Initialize(project=settings.gee_project)
            _GEE_OK = True
        except Exception as exc:                        # noqa: BLE001
            log.warning("gee_init_failed", exc_info=exc)
except Exception:                                       # noqa: BLE001
    pass


@dataclass
class EnrichedPoint:
    temperature_c: float | None;     temperature_status: str
    humidity:      float | None;     humidity_status:    str
    cloud_cover_pct: float | None;   cloud_cover_status: str
    cloud_cover_label: str | None
    altitude_m:    float | None;     altitude_status:    str
    building_count: int | None;      building_status:    str
    avg_building_height: float | None
    distance_to_building_m: float | None
    tree_count:    int | None;       tree_status:        str
    avg_tree_height_m: float | None
    distance_to_tree_m: float | None
    tree_density_ndvi: float | None; ndvi_status: str


# ---------------------------------------------------------------------------
# Cache helpers — per-API geographic bucketing
# ---------------------------------------------------------------------------
BUCKET_M_PER_API: dict[str, float] = {
    "open_meteo":      5000.0,
    "open_meteo_elev":   25.0,
    "overpass":          25.0,
    "overpass_trees":    25.0,
    "gee_ndvi":          15.0,
}
DEFAULT_BUCKET_M = 25.0


def _quantize_latlon(lat: float, lon: float, bucket_m: float
                     ) -> tuple[float, float]:
    """Snap (lat, lon) to the centre of a bucket-sized grid cell.

    Two-stage quantisation: latitude is snapped first, then `cos(lat)` is
    derived from the *quantised* latitude. This makes the longitude grid
    stable across all points that share a latitude bucket — without it, two
    measurements 10 m apart in latitude could produce slightly different
    longitude bucket centres because cos(lat) differs by 10⁻⁵, defeating
    coalescing entirely.
    """
    m_per_deg_lat = 111_000.0
    qlat = round(lat * m_per_deg_lat / bucket_m) * bucket_m / m_per_deg_lat
    m_per_deg_lon = 111_000.0 * max(0.10, math.cos(math.radians(qlat)))
    qlon = round(lon * m_per_deg_lon / bucket_m) * bucket_m / m_per_deg_lon
    return qlat, qlon


def _cache_key(api: str, lat: float, lon: float) -> str:
    bucket_m = BUCKET_M_PER_API.get(api, DEFAULT_BUCKET_M)
    qlat, qlon = _quantize_latlon(lat, lon, bucket_m)
    # 6-decimal precision avoids float-rounding noise creating sibling keys.
    return f"{api}:{qlat:.6f}:{qlon:.6f}:b{int(bucket_m)}m"


def _cache_get(session: Session, api: str, lat: float, lon: float) -> Any | None:
    rec = session.query(GeoCache).filter_by(cache_key=_cache_key(api, lat, lon)).first()
    if rec is None:
        return None
    try:
        return json.loads(rec.payload_json)
    except Exception:                                   # noqa: BLE001
        return None


def _cache_put(session: Session, api: str, lat: float, lon: float, payload: Any) -> None:
    """UPSERT by cache_key. Bucketing makes many measurements share a key, so
    a plain INSERT (or `session.merge(rec)` with no id) violates the unique
    constraint on the second call. We look the row up by key and update it
    in place when it already exists."""
    key = _cache_key(api, lat, lon)
    rec = session.query(GeoCache).filter_by(cache_key=key).first()
    payload_str = json.dumps(payload, ensure_ascii=False)
    if rec is None:
        session.add(GeoCache(
            cache_key=key, api_name=api,
            payload_json=payload_str, fetched_at=datetime.utcnow(),
        ))
    else:
        rec.payload_json = payload_str
        rec.fetched_at = datetime.utcnow()
    session.flush()


# ---------------------------------------------------------------------------
# Per-API fetchers (atomic; never raise to caller)
# ---------------------------------------------------------------------------
def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6_371_000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return float(2 * R * math.asin(math.sqrt(a)))


def _cloud_label(pct: float | None) -> str | None:
    """Map Open-Meteo cloud_cover_percent to the plan's 3-level label."""
    if pct is None:
        return None
    if pct < 25:  return "Sem nuvens"
    if pct < 75:  return "Parcialmente nublado"
    return "Nublado"


async def fetch_open_meteo(client: httpx.AsyncClient, session: Session, run_id: int,
                           lat: float, lon: float
                           ) -> tuple[float | None, str, float | None, str,
                                      float | None, str, str | None]:
    """Returns (temp, temp_status, humidity, humidity_status,
                cloud_cover_pct, cloud_status, cloud_label)."""
    def _unpack(cur: dict) -> tuple:
        t = cur.get("temperature_2m")
        h = cur.get("relative_humidity_2m")
        c = cur.get("cloud_cover")
        return (
            t, "ok" if t is not None else "missing_field",
            h, "ok" if h is not None else "missing_field",
            c, "ok" if c is not None else "missing_field",
            _cloud_label(c),
        )

    cached = _cache_get(session, "open_meteo", lat, lon)
    if cached:
        log_api_call(session, run_id=run_id, api_name="open_meteo",
                     lat=lat, lon=lon, status="cached")
        return _unpack(cached.get("current", {}))

    t0 = time.perf_counter()
    try:
        r = await client.get(settings.open_meteo_forecast_url, params={
            "latitude": lat, "longitude": lon,
            "current": "temperature_2m,relative_humidity_2m,cloud_cover",
        }, timeout=settings.http_timeout_s)
        latency_ms = (time.perf_counter() - t0) * 1000
        if r.status_code != 200:
            log_api_call(session, run_id=run_id, api_name="open_meteo",
                         lat=lat, lon=lon, status="failed",
                         http_code=r.status_code, latency_ms=latency_ms)
            return (None, "api_failed", None, "api_failed",
                    None, "api_failed", None)
        data = r.json()
        _cache_put(session, "open_meteo", lat, lon, data)
        log_api_call(session, run_id=run_id, api_name="open_meteo",
                     lat=lat, lon=lon, status="ok",
                     http_code=r.status_code, latency_ms=latency_ms)
        return _unpack(data.get("current", {}))
    except Exception as exc:                            # noqa: BLE001
        log_api_call(session, run_id=run_id, api_name="open_meteo",
                     lat=lat, lon=lon, status="failed",
                     error_message=str(exc),
                     latency_ms=(time.perf_counter() - t0) * 1000)
        return (None, "api_failed", None, "api_failed",
                None, "api_failed", None)


async def fetch_elevation(client: httpx.AsyncClient, session: Session, run_id: int,
                          lat: float, lon: float
                          ) -> tuple[float | None, str]:
    cached = _cache_get(session, "open_meteo_elev", lat, lon)
    if cached:
        log_api_call(session, run_id=run_id, api_name="open_meteo_elev",
                     lat=lat, lon=lon, status="cached")
        elev_list = cached.get("elevation") or []
        v = elev_list[0] if elev_list else None
        return (v, "ok" if v is not None else "missing_field")

    t0 = time.perf_counter()
    try:
        r = await client.get(settings.open_meteo_elevation_url,
                             params={"latitude": lat, "longitude": lon},
                             timeout=settings.http_timeout_s)
        latency_ms = (time.perf_counter() - t0) * 1000
        if r.status_code != 200:
            log_api_call(session, run_id=run_id, api_name="open_meteo_elev",
                         lat=lat, lon=lon, status="failed",
                         http_code=r.status_code, latency_ms=latency_ms)
            return (None, "api_failed")
        data = r.json()
        _cache_put(session, "open_meteo_elev", lat, lon, data)
        log_api_call(session, run_id=run_id, api_name="open_meteo_elev",
                     lat=lat, lon=lon, status="ok",
                     http_code=r.status_code, latency_ms=latency_ms)
        v = (data.get("elevation") or [None])[0]
        return (v, "ok" if v is not None else "missing_field")
    except Exception as exc:                            # noqa: BLE001
        log_api_call(session, run_id=run_id, api_name="open_meteo_elev",
                     lat=lat, lon=lon, status="failed", error_message=str(exc),
                     latency_ms=(time.perf_counter() - t0) * 1000)
        return (None, "api_failed")


async def fetch_overpass(client: httpx.AsyncClient, session: Session, run_id: int,
                         lat: float, lon: float
                         ) -> tuple[int | None, str, float | None, float | None]:
    """Returns (building_count, status, avg_building_height, distance_to_nearest_m).

    Asks for `out center` so each way comes with a representative lat/lon we
    can use to compute the distance to the nearest building (the project's
    Quadro 2 demands this — not just the count)."""
    cached = _cache_get(session, "overpass", lat, lon)
    if cached is not None:
        log_api_call(session, run_id=run_id, api_name="overpass",
                     lat=lat, lon=lon, status="cached")
        return (cached.get("count"),
                "ok" if cached.get("count") is not None else "missing_field",
                cached.get("avg_height"),
                cached.get("distance_to_nearest_m"))

    t0 = time.perf_counter()
    q = (f'[out:json];(way(around:{settings.overpass_radius_m},{lat},{lon})'
         f'["building"];);out tags center;')
    try:
        r = await client.post(settings.overpass_url, data={"data": q},
                              timeout=settings.http_timeout_s)
        latency_ms = (time.perf_counter() - t0) * 1000
        if r.status_code != 200:
            log_api_call(session, run_id=run_id, api_name="overpass",
                         lat=lat, lon=lon, status="failed",
                         http_code=r.status_code, latency_ms=latency_ms)
            return (None, "api_failed", None, None)
        data = r.json()
        elements = data.get("elements", [])
        count = len(elements)
        heights: list[float] = []
        nearest_m: float | None = None
        for el in elements:
            tags = el.get("tags", {})
            if "height" in tags:
                try: heights.append(float(tags["height"]))
                except ValueError: pass
            elif "building:levels" in tags:
                try: heights.append(float(tags["building:levels"]) * 3.5)
                except ValueError: pass
            c = el.get("center") or {}
            if c.get("lat") is not None and c.get("lon") is not None:
                d = _haversine_m(lat, lon, float(c["lat"]), float(c["lon"]))
                if nearest_m is None or d < nearest_m:
                    nearest_m = d
        avg_h = sum(heights) / len(heights) if heights else None
        payload = {"count": count, "avg_height": avg_h,
                   "distance_to_nearest_m": nearest_m}
        _cache_put(session, "overpass", lat, lon, payload)
        log_api_call(session, run_id=run_id, api_name="overpass",
                     lat=lat, lon=lon, status="ok",
                     http_code=r.status_code, latency_ms=latency_ms)
        return (count, "ok", avg_h, nearest_m)
    except Exception as exc:                            # noqa: BLE001
        log_api_call(session, run_id=run_id, api_name="overpass",
                     lat=lat, lon=lon, status="failed", error_message=str(exc),
                     latency_ms=(time.perf_counter() - t0) * 1000)
        return (None, "api_failed", None, None)


async def fetch_overpass_trees(client: httpx.AsyncClient, session: Session,
                               run_id: int, lat: float, lon: float
                               ) -> tuple[int | None, str, float | None, float | None]:
    """Returns (tree_count, status, avg_tree_height_m, distance_to_nearest_m).

    Queries OSM for `natural=tree` nodes within the configured radius. Trees
    in OSM are point features, so distance is direct. Average height is only
    available when the node has a `height` tag."""
    cached = _cache_get(session, "overpass_trees", lat, lon)
    if cached is not None:
        log_api_call(session, run_id=run_id, api_name="overpass_trees",
                     lat=lat, lon=lon, status="cached")
        return (cached.get("count"),
                "ok" if cached.get("count") is not None else "missing_field",
                cached.get("avg_height"),
                cached.get("distance_to_nearest_m"))

    t0 = time.perf_counter()
    q = (f'[out:json];(node(around:{settings.overpass_radius_m},{lat},{lon})'
         f'["natural"="tree"];);out tags;')
    try:
        r = await client.post(settings.overpass_url, data={"data": q},
                              timeout=settings.http_timeout_s)
        latency_ms = (time.perf_counter() - t0) * 1000
        if r.status_code != 200:
            log_api_call(session, run_id=run_id, api_name="overpass_trees",
                         lat=lat, lon=lon, status="failed",
                         http_code=r.status_code, latency_ms=latency_ms)
            return (None, "api_failed", None, None)
        data = r.json()
        elements = data.get("elements", [])
        count = len(elements)
        heights: list[float] = []
        nearest_m: float | None = None
        for el in elements:
            tags = el.get("tags", {})
            h_raw = tags.get("height")
            if h_raw:
                try: heights.append(float(h_raw))
                except ValueError: pass
            t_lat, t_lon = el.get("lat"), el.get("lon")
            if t_lat is not None and t_lon is not None:
                d = _haversine_m(lat, lon, float(t_lat), float(t_lon))
                if nearest_m is None or d < nearest_m:
                    nearest_m = d
        avg_h = sum(heights) / len(heights) if heights else None
        payload = {"count": count, "avg_height": avg_h,
                   "distance_to_nearest_m": nearest_m}
        _cache_put(session, "overpass_trees", lat, lon, payload)
        log_api_call(session, run_id=run_id, api_name="overpass_trees",
                     lat=lat, lon=lon, status="ok",
                     http_code=r.status_code, latency_ms=latency_ms)
        return (count, "ok", avg_h, nearest_m)
    except Exception as exc:                            # noqa: BLE001
        log_api_call(session, run_id=run_id, api_name="overpass_trees",
                     lat=lat, lon=lon, status="failed", error_message=str(exc),
                     latency_ms=(time.perf_counter() - t0) * 1000)
        return (None, "api_failed", None, None)


def fetch_ndvi(session: Session, run_id: int, lat: float, lon: float
               ) -> tuple[float | None, str]:
    """Synchronous because the Earth Engine Python client is synchronous."""
    if not _GEE_OK:
        log_api_call(session, run_id=run_id, api_name="gee_ndvi",
                     lat=lat, lon=lon, status="skipped",
                     error_message="GEE not initialised")
        return (None, "api_failed")

    cached = _cache_get(session, "gee_ndvi", lat, lon)
    if cached is not None:
        log_api_call(session, run_id=run_id, api_name="gee_ndvi",
                     lat=lat, lon=lon, status="cached")
        v = cached.get("ndvi")
        return (v, "ok" if v is not None else "missing_field")

    t0 = time.perf_counter()
    try:
        import ee  # type: ignore
        point = ee.Geometry.Point([lon, lat])
        coll = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
                  .filterBounds(point)
                  .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", settings.gee_max_cloud_pct))
                  .sort("system:time_start", False))
        img = coll.first()
        if img is None:
            log_api_call(session, run_id=run_id, api_name="gee_ndvi",
                         lat=lat, lon=lon, status="ok",
                         latency_ms=(time.perf_counter() - t0) * 1000)
            _cache_put(session, "gee_ndvi", lat, lon, {"ndvi": None})
            return (None, "missing_field")
        ndvi = img.normalizedDifference(["B8", "B4"]).rename("NDVI")
        val = ndvi.reduceRegion(ee.Reducer.mean(), point, 10).getInfo() or {}
        v = val.get("NDVI")
        v = round(float(v), 4) if v is not None else None
        _cache_put(session, "gee_ndvi", lat, lon, {"ndvi": v})
        log_api_call(session, run_id=run_id, api_name="gee_ndvi",
                     lat=lat, lon=lon, status="ok",
                     latency_ms=(time.perf_counter() - t0) * 1000)
        return (v, "ok" if v is not None else "missing_field")
    except Exception as exc:                            # noqa: BLE001
        log_api_call(session, run_id=run_id, api_name="gee_ndvi",
                     lat=lat, lon=lon, status="failed", error_message=str(exc),
                     latency_ms=(time.perf_counter() - t0) * 1000)
        return (None, "api_failed")


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
async def enrich_point(client: httpx.AsyncClient, session: Session,
                       run_id: int, lat: float, lon: float,
                       *, need_altitude: bool) -> EnrichedPoint:
    """Run weather + Overpass (buildings + trees) + elevation in parallel.
    NDVI stays synchronous because Earth Engine's Python client is sync."""
    tasks = [
        fetch_open_meteo(client, session, run_id, lat, lon),
        fetch_overpass(client, session, run_id, lat, lon),
        fetch_overpass_trees(client, session, run_id, lat, lon),
    ]
    if need_altitude:
        tasks.append(fetch_elevation(client, session, run_id, lat, lon))
    res = await asyncio.gather(*tasks)
    (t, t_st, h, h_st, c_pct, c_st, c_label) = res[0]
    (n_b, b_st, avg_bh, dist_b) = res[1]
    (n_t, t_st_o, avg_th, dist_t) = res[2]
    if need_altitude:
        (alt, alt_st)  = res[3]
    else:
        alt, alt_st = (None, "skipped")

    ndvi, ndvi_st = fetch_ndvi(session, run_id, lat, lon)

    return EnrichedPoint(
        temperature_c=t, temperature_status=t_st,
        humidity=h,      humidity_status=h_st,
        cloud_cover_pct=c_pct, cloud_cover_status=c_st,
        cloud_cover_label=c_label,
        altitude_m=alt,  altitude_status=alt_st,
        building_count=n_b, building_status=b_st,
        avg_building_height=avg_bh,
        distance_to_building_m=dist_b,
        tree_count=n_t, tree_status=t_st_o,
        avg_tree_height_m=avg_th,
        distance_to_tree_m=dist_t,
        tree_density_ndvi=ndvi, ndvi_status=ndvi_st,
    )
