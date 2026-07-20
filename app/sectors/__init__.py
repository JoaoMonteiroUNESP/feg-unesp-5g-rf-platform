"""
FEG-UNESP sector module.

Pipeline
--------
1. `feg_sectors_local.json` — sectors in the local Cartesian system (origin SW
    corner of the campus drawing, units = metres). Static.
2. `data/calibration.json` — affine transform local→WGS84, fitted from
    user-provided control points. Persisted by the calibration tool.
3. `loader.build_sector_geojson` — applies the calibration to produce
    a GeoJSON FeatureCollection in WGS84.
4. `classifier.SectorClassifier` — point-in-polygon lookup with STRtree.

Until the user provides control points, the system runs in 'uncalibrated'
mode: every measurement is classified as `sector_code=None`, and the API
endpoints expose this status so the dashboard can flag it.
"""
from app.sectors.legend import LEGEND
from app.sectors.calibration import (
    ControlPoint, Calibration, fit_affine,
    save_calibration, load_calibration, transform_local_to_wgs84,
)
from app.sectors.loader import (
    load_local_sectors, build_sector_geojson,
)
from app.sectors.classifier import SectorClassifier
