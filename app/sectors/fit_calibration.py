"""
Command-line tool to fit and persist the affine calibration.

Usage
-----
1) Generate the template CSV (a header + a few example rows commented out):

       python -m app.sectors.fit_calibration --template > control_points.csv

2) Fill `control_points.csv` with at least 3 (recommended 4–6) measured points.
   Each row maps a known location on the campus drawing (x_local, y_local in
   metres, origin SW) to its real-world WGS84 coordinates (lat, lon).

3) Fit:

       python -m app.sectors.fit_calibration --csv control_points.csv \
           --notes "Survey 2026-05-04, GPS RTK, 8 CPs"

The result is written to `data/calibration.json` and a quality report is
printed to stdout (per-point residual + RMS in metres).
"""
from __future__ import annotations
import argparse
import csv
import sys
from pathlib import Path

from app.sectors.calibration import (
    CALIBRATION_PATH, ControlPoint, fit_affine, save_calibration,
)


TEMPLATE = (
    "# Control points for the FEG-UNESP calibration.\n"
    "# Required columns: name,x_local,y_local,lat,lon\n"
    "# x_local, y_local are in METRES, origin = SW corner of the campus drawing.\n"
    "# lat, lon are WGS84 decimal degrees (e.g. -23.20987,-45.87654).\n"
    "# Provide at least 3 NON-COLINEAR points (4–6 is recommended).\n"
    "name,x_local,y_local,lat,lon\n"
    "# Example rows (delete the leading '#' and replace with real measurements):\n"
    "# CP1_corner_SW_FEGAO,32.5,25.0,-23.21000,-45.87800\n"
    "# CP2_corner_NE_Biblioteca,265.0,350.0,-23.20850,-45.87650\n"
    "# CP3_creche_NW,330.0,105.0,-23.20950,-45.87600\n"
    "# CP4_moradia_E,675.0,210.0,-23.20880,-45.87420\n"
)


def _read_csv(path: Path) -> list[ControlPoint]:
    cps: list[ControlPoint] = []
    with path.open(encoding="utf-8") as f:
        # skip blank/comment lines, then dict-read
        rows = [ln for ln in f.readlines()
                if ln.strip() and not ln.lstrip().startswith("#")]
    if not rows:
        raise SystemExit("CSV vazio (apenas comentários).")
    reader = csv.DictReader(rows)
    required = {"name", "x_local", "y_local", "lat", "lon"}
    if reader.fieldnames is None or not required.issubset(reader.fieldnames):
        raise SystemExit(
            f"CSV deve ter as colunas {sorted(required)}; "
            f"encontradas {reader.fieldnames}"
        )
    for i, row in enumerate(reader, start=1):
        try:
            cps.append(ControlPoint(
                name=row["name"].strip(),
                x_local=float(row["x_local"]),
                y_local=float(row["y_local"]),
                lat=float(row["lat"]),
                lon=float(row["lon"]),
            ))
        except (KeyError, ValueError) as e:
            raise SystemExit(f"Linha {i} inválida: {e}")
    return cps


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="fit_calibration",
        description="Fit local→WGS84 affine calibration from control points.",
    )
    p.add_argument("--template", action="store_true",
                   help="Print a CSV template to stdout and exit.")
    p.add_argument("--csv", type=Path,
                   help="CSV file with control points (see --template).")
    p.add_argument("--out", type=Path, default=CALIBRATION_PATH,
                   help=f"Output JSON path (default: {CALIBRATION_PATH}).")
    p.add_argument("--notes", default="",
                   help="Free-text notes saved with the calibration.")
    p.add_argument("--max-rms", type=float, default=None,
                   help="If set, exits with code 2 when RMS exceeds this (m).")
    args = p.parse_args(argv)

    if args.template:
        sys.stdout.write(TEMPLATE)
        return 0

    if not args.csv:
        p.error("Use --template ou --csv <arquivo>.")

    cps = _read_csv(args.csv)
    cal = fit_affine(cps, notes=args.notes)
    save_calibration(cal, path=args.out)

    print(f"Calibração salva em: {args.out}")
    print(f"  pontos de controle : {cal.n_points}")
    print(f"  RMS (metros)       : {cal.rms_m:.3f}")
    print("  resíduos por ponto :")
    for r in cal.residuals_m:
        print(f"    - {r['name']:30s}  {r['residual_m']:7.3f} m")

    if args.max_rms is not None and cal.rms_m > args.max_rms:
        print(f"\nFALHA: RMS {cal.rms_m:.3f} m excede limite {args.max_rms} m.",
              file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
