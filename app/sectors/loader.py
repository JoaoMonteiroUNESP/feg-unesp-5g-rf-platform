"""
Sector loader: local-meters JSON → WGS84 GeoJSON FeatureCollection.

The local JSON (`feg_sectors_local.json`) holds 21 axis-aligned rectangles in
campus-drawing metres (origin SW). To use them analytically (point-in-polygon
on real GPS fixes) every corner is transformed to WGS84 via the affine
calibration produced by `calibration.fit_affine`.

If no calibration is available the loader returns an empty FeatureCollection
with `properties.calibrated = False`, so the rest of the system can run in
honest 'uncalibrated' mode without faking coordinates.
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any

from app.sectors.calibration import (
    Calibration, transform_local_to_wgs84,
)
from app.sectors.legend import LEGEND, code_for


SECTORS_LOCAL_PATH = Path(__file__).with_name("feg_sectors_local.json")


# ---------------------------------------------------------------------------
# Local JSON loading
# ---------------------------------------------------------------------------
def load_local_sectors(path: Path = SECTORS_LOCAL_PATH) -> list[dict]:
    """Returns the raw list of sector dicts (xmin,ymin,xmax,ymax in metres)."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    sectors = payload.get("sectors", [])
    # Sanity: keep only well-formed entries
    clean: list[dict] = []
    for s in sectors:
        try:
            sid = int(s["id"])
            xmin = float(s["xmin"]); ymin = float(s["ymin"])
            xmax = float(s["xmax"]); ymax = float(s["ymax"])
            if xmax <= xmin or ymax <= ymin:
                continue
            clean.append({
                "id": sid,
                "xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax,
            })
        except (KeyError, TypeError, ValueError):
            continue
    clean.sort(key=lambda s: s["id"])
    return clean


# ---------------------------------------------------------------------------
# GeoJSON building
# ---------------------------------------------------------------------------
def _rect_corners_local(s: dict) -> list[tuple[float, float]]:
    """CCW order, closing ring."""
    return [
        (s["xmin"], s["ymin"]),
        (s["xmax"], s["ymin"]),
        (s["xmax"], s["ymax"]),
        (s["xmin"], s["ymax"]),
        (s["xmin"], s["ymin"]),
    ]


def _feature_uncalibrated(s: dict) -> dict:
    """Sector feature with NO geometry — analyses must skip these gracefully."""
    meta = LEGEND.get(s["id"])
    return {
        "type": "Feature",
        "geometry": None,
        "properties": {
            "sector_id": s["id"],
            "sector_code": code_for(s["id"]),
            "sector_name": meta.name if meta else f"Setor {s['id']}",
            "environment_class": meta.environment_class if meta else None,
            "to_verify": bool(meta.to_verify) if meta else True,
            "local_bbox_m": [s["xmin"], s["ymin"], s["xmax"], s["ymax"]],
            "calibrated": False,
        },
    }


def _feature_wgs84(s: dict, cal: Calibration) -> dict:
    corners_xy = _rect_corners_local(s)
    # GeoJSON expects [lon, lat]
    coords_lonlat: list[list[float]] = []
    for x, y in corners_xy:
        lat, lon = transform_local_to_wgs84(cal, x, y)
        coords_lonlat.append([lon, lat])

    meta = LEGEND.get(s["id"])
    return {
        "type": "Feature",
        "geometry": {
            "type": "Polygon",
            "coordinates": [coords_lonlat],
        },
        "properties": {
            "sector_id": s["id"],
            "sector_code": code_for(s["id"]),
            "sector_name": meta.name if meta else f"Setor {s['id']}",
            "environment_class": meta.environment_class if meta else None,
            "to_verify": bool(meta.to_verify) if meta else True,
            "local_bbox_m": [s["xmin"], s["ymin"], s["xmax"], s["ymax"]],
            "calibrated": True,
        },
    }


def build_sector_geojson(sectors: list[dict] | None = None,
                         calibration: Calibration | None = None
                         ) -> dict[str, Any]:
    """
    Build a FeatureCollection.

    * sectors=None      → loads the local JSON
    * calibration=None  → returns features with geometry=None and
                          properties.calibrated=False
    """
    secs = sectors if sectors is not None else load_local_sectors()

    if calibration is None:
        feats = [_feature_uncalibrated(s) for s in secs]
        return {
            "type": "FeatureCollection",
            "properties": {
                "calibrated": False,
                "n_sectors": len(feats),
                "rms_m": None,
            },
            "features": feats,
        }

    feats = [_feature_wgs84(s, calibration) for s in secs]
    return {
        "type": "FeatureCollection",
        "properties": {
            "calibrated": True,
            "n_sectors": len(feats),
            "rms_m": calibration.rms_m,
            "n_control_points": calibration.n_points,
            "fitted_at": calibration.fitted_at,
            "project_version": calibration.project_version,
        },
        "features": feats,
    }
