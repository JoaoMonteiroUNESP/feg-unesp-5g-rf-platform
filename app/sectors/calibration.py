"""
Affine calibration: local Cartesian (metres) → WGS84 (degrees).

Model
-----
Six parameters, fit by ordinary least squares from ≥3 control points:

    lon =  a · x_local  +  b · y_local  +  tx
    lat =  c · x_local  +  d · y_local  +  ty

The matrix [a b ; c d] absorbs scale, rotation and shear in one shot. With
4–6 well-distributed control points the residual root-mean-square (RMS),
computed in metres on the WGS84 ellipsoid, is the calibration's quality
indicator. Recommended target: RMS < 2 m for analyses by sector that has
sub-50 m sides.

Auditability
------------
The fitted calibration JSON contains:
  * the original control points (so anyone can re-fit and verify);
  * the fitted parameters;
  * the per-point residuals in metres;
  * the RMS;
  * the timestamp + project version.
"""
from __future__ import annotations
import json
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
from pyproj import Geod

from app import __version__
from app.config import DATA_DIR


CALIBRATION_PATH = DATA_DIR / "calibration.json"
_GEOD = Geod(ellps="WGS84")


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class ControlPoint:
    name:    str
    x_local: float
    y_local: float
    lat:     float
    lon:     float


@dataclass
class Calibration:
    p_lon: list[float]                      # [a, b, tx]
    p_lat: list[float]                      # [c, d, ty]
    rms_m: float
    n_points: int
    residuals_m: list[dict] = field(default_factory=list)  # per-CP residuals
    fitted_at: str = ""                     # ISO-8601
    project_version: str = ""
    control_points: list[dict] = field(default_factory=list)
    notes: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    def looks_synthetic(self) -> bool:
        """
        Heuristic flag: True if this calibration smells like a test fixture
        rather than a real field survey. Two signals:

        * RMS < 0.01 m — real GPS fixes never fit that tightly; a sub-cm RMS
          means the control points were generated from a closed-form model.
        * `notes` contains test/synthetic/pytest markers.

        The dashboard surfaces this so the student doesn't accidentally use
        a placeholder calibration to classify real measurements.
        """
        if self.rms_m is not None and self.rms_m < 0.01:
            return True
        n = (self.notes or "").lower()
        return any(tok in n for tok in ("pytest", "synthetic", "fixture",
                                        "test", "smoke"))


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------
def fit_affine(controls: Iterable[ControlPoint], notes: str = "") -> Calibration:
    cps = list(controls)
    if len(cps) < 3:
        raise ValueError(
            f"São necessários ≥3 pontos de controle não colineares; recebidos {len(cps)}."
        )

    A = np.array([[c.x_local, c.y_local, 1.0] for c in cps], dtype=float)
    lon = np.array([c.lon for c in cps], dtype=float)
    lat = np.array([c.lat for c in cps], dtype=float)

    if np.linalg.matrix_rank(A) < 3:
        raise ValueError("Pontos de controle colineares ou degenerados.")

    p_lon, *_ = np.linalg.lstsq(A, lon, rcond=None)
    p_lat, *_ = np.linalg.lstsq(A, lat, rcond=None)

    residuals: list[dict] = []
    sq = []
    for c in cps:
        pred_lon = float(p_lon @ [c.x_local, c.y_local, 1.0])
        pred_lat = float(p_lat @ [c.x_local, c.y_local, 1.0])
        _, _, dist_m = _GEOD.inv(c.lon, c.lat, pred_lon, pred_lat)
        residuals.append({
            "name": c.name,
            "x_local": c.x_local, "y_local": c.y_local,
            "lat_obs": c.lat, "lon_obs": c.lon,
            "lat_pred": pred_lat, "lon_pred": pred_lon,
            "residual_m": float(dist_m),
        })
        sq.append(dist_m ** 2)

    rms = float(np.sqrt(np.mean(sq)))

    return Calibration(
        p_lon=[float(x) for x in p_lon],
        p_lat=[float(x) for x in p_lat],
        rms_m=rms,
        n_points=len(cps),
        residuals_m=residuals,
        fitted_at=datetime.utcnow().isoformat(timespec="seconds") + "Z",
        project_version=__version__,
        control_points=[asdict(c) for c in cps],
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------
def save_calibration(cal: Calibration, path: Path = CALIBRATION_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cal.to_dict(), indent=2, ensure_ascii=False),
                    encoding="utf-8")
    return path


def load_calibration(path: Path = CALIBRATION_PATH) -> Calibration | None:
    if not path.exists():
        return None
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
        return Calibration(
            p_lon=list(d["p_lon"]), p_lat=list(d["p_lat"]),
            rms_m=float(d["rms_m"]), n_points=int(d["n_points"]),
            residuals_m=d.get("residuals_m", []),
            fitted_at=d.get("fitted_at", ""),
            project_version=d.get("project_version", ""),
            control_points=d.get("control_points", []),
            notes=d.get("notes", ""),
        )
    except Exception:                                   # noqa: BLE001
        return None


# ---------------------------------------------------------------------------
# Inverse + forward transform helpers
# ---------------------------------------------------------------------------
def transform_local_to_wgs84(cal: Calibration, x: float, y: float
                             ) -> tuple[float, float]:
    """Returns (lat, lon)."""
    a, b, tx = cal.p_lon
    c, d, ty = cal.p_lat
    lon = a * x + b * y + tx
    lat = c * x + d * y + ty
    return lat, lon


def transform_wgs84_to_local(cal: Calibration, lat: float, lon: float
                             ) -> tuple[float, float]:
    """Returns (x_local, y_local). Solves the affine system in matrix form."""
    a, b, tx = cal.p_lon
    c, d, ty = cal.p_lat
    M = np.array([[a, b], [c, d]], dtype=float)
    rhs = np.array([lon - tx, lat - ty], dtype=float)
    sol = np.linalg.solve(M, rhs)
    return float(sol[0]), float(sol[1])
