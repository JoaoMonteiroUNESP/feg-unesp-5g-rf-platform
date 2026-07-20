"""
Point-in-polygon sector classifier.

Built once from a GeoJSON FeatureCollection (output of `loader.build_sector_geojson`)
and answered millions of times per upload via a Shapely STRtree spatial index.

Behaviour
---------
* If the FeatureCollection is *uncalibrated* (geometry=None on all features),
  every classification returns `(None, None, None)`. This is the honest
  "we don't know yet" state — never silently default to a sector.
* If calibrated, returns the first feature whose polygon `contains` the point.
  Sectors in the FEG-UNESP map do not overlap, so first-match is unambiguous.
* If a point falls in no polygon, returns `(None, None, None)`.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable

from shapely.geometry import Point, shape
from shapely.strtree import STRtree


@dataclass(frozen=True)
class SectorHit:
    sector_code: str | None
    sector_name: str | None
    environment_class: str | None


_NULL_HIT = SectorHit(None, None, None)


class SectorClassifier:
    """
    Wraps a list of (polygon, properties) pairs in an STRtree.

    Coordinate convention: polygons are in WGS84 (lon, lat). The classifier
    receives queries as (lat, lon) — the order seen in our Measurement model —
    and converts internally.
    """

    def __init__(self, geojson: dict):
        self.calibrated: bool = bool(
            geojson.get("properties", {}).get("calibrated", False)
        )
        self._geoms = []
        self._props: list[dict] = []

        for feat in geojson.get("features", []):
            geom_dict = feat.get("geometry")
            if geom_dict is None:
                continue
            try:
                g = shape(geom_dict)
            except Exception:                                        # noqa: BLE001
                continue
            if g.is_empty or not g.is_valid:
                continue
            self._geoms.append(g)
            self._props.append(feat.get("properties", {}))

        # STRtree over geometries; query returns indices in shapely 2.x
        self._tree: STRtree | None = (
            STRtree(self._geoms) if self._geoms else None
        )

    # ------------------------------------------------------------------
    @property
    def n_sectors(self) -> int:
        return len(self._geoms)

    # ------------------------------------------------------------------
    def classify(self, lat: float | None, lon: float | None) -> SectorHit:
        if not self.calibrated or self._tree is None:
            return _NULL_HIT
        if lat is None or lon is None:
            return _NULL_HIT
        try:
            pt = Point(float(lon), float(lat))
        except (TypeError, ValueError):
            return _NULL_HIT

        # shapely 2.x: query returns ndarray of indices of bbox-candidates
        idxs = self._tree.query(pt)
        for i in idxs:
            geom = self._geoms[int(i)]
            if geom.covers(pt):                # inclusive of boundary
                p = self._props[int(i)]
                return SectorHit(
                    sector_code=p.get("sector_code"),
                    sector_name=p.get("sector_name"),
                    environment_class=p.get("environment_class"),
                )
        return _NULL_HIT

    # ------------------------------------------------------------------
    def classify_batch(
        self, points: Iterable[tuple[float | None, float | None]]
    ) -> list[SectorHit]:
        return [self.classify(lat, lon) for lat, lon in points]
