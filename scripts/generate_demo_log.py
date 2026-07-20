"""Generate a deterministic, explicitly synthetic G-NetTrack-style log.

The output exists only to demonstrate ingestion and visualisation. It is not a
measurement campaign and must never be used in scientific results.
"""

from __future__ import annotations

import argparse
import csv
import math
import random
from datetime import datetime, timedelta
from pathlib import Path


DEFAULT_OUTPUT = Path("data/demo/demo_gnettrack_synthetic.tsv")
HEADER = [
    "Timestamp", "Latitude", "Longitude", "Accuracy", "Altitude", "Speed",
    "Operatorname", "Operator", "CGI", "CellID", "LAC", "NetworkTech",
    "NetworkMode", "Level", "Qual", "SNR", "CQI", "LTERSSI", "ARFCN",
    "BAND", "BANDWIDTH", "DL_bitrate", "UL_bitrate", "Distance", "EVENT",
    "EVENTDETAILS", "PINGAVG", "PINGMIN", "PINGMAX", "PINGSTDEV", "PINGLOSS",
    "TESTDOWNLINK", "TESTUPLINK", "TESTDOWNLINKMAX", "TESTUPLINKMAX",
    "CSI_RSRP", "CSI_RSRQ", "CSI_SNR", "DataConnection_Type",
]


def _row(index: int, total: int, rng: random.Random) -> list[object]:
    angle = 2 * math.pi * index / total
    timestamp = datetime(2026, 1, 15, 9, 0) + timedelta(seconds=2 * index)

    # Synthetic loop near the public campus extent; values do not reproduce a
    # real route, device, cell or observation.
    latitude = -23.2090 + 0.0017 * math.sin(angle) + rng.gauss(0, 0.00002)
    longitude = -45.8750 + 0.0022 * math.cos(angle) + rng.gauss(0, 0.00002)
    distance = 45 + 360 * (0.5 + 0.5 * math.sin(angle - 0.8))
    obstruction = 8 if math.sin(2 * angle) > 0.45 else 0
    rsrp = -70 - 18 * math.log10(1 + distance / 30) - obstruction + rng.gauss(0, 2.2)
    rsrq = -8 - max(0, (-90 - rsrp) * 0.12) + rng.gauss(0, 0.8)
    sinr = 25 - max(0, (-80 - rsrp) * 0.45) + rng.gauss(0, 1.5)
    cqi = max(1, min(15, round((sinr + 5) / 2)))

    # Active QoS tests are sparse by design, reflecting their slower cadence.
    qos = index % 12 == 0
    ping = max(12, 34 + (-90 - rsrp) * 0.8 + rng.gauss(0, 4)) if qos else "-"
    download = max(600, 8500 + (rsrp + 95) * 280 + rng.gauss(0, 500)) if qos else "-"
    upload = max(300, 2200 + (rsrp + 95) * 75 + rng.gauss(0, 220)) if qos else "-"

    return [
        timestamp.strftime("%Y.%m.%d_%H.%M.%S"),
        f"{latitude:.7f}", f"{longitude:.7f}", f"{rng.uniform(4, 12):.1f}",
        f"{545 + 4 * math.sin(angle):.1f}", f"{rng.uniform(1.0, 4.5):.1f}",
        "DEMO", "DEMO", "000.00.000.0000", "SYNTHETIC", "0", "5G", "5G NSA",
        f"{rsrp:.1f}", f"{rsrq:.1f}", f"{sinr:.1f}", cqi, f"{rsrp + 25:.1f}",
        1850, "L7", "20", "-", "-", f"{distance:.1f}", "PERIODIC", "SYNTHETIC",
        f"{ping:.1f}" if qos else "-", f"{max(8, ping - 5):.1f}" if qos else "-",
        f"{ping + 7:.1f}" if qos else "-", f"{rng.uniform(1, 5):.1f}" if qos else "-",
        "0" if qos else "-", "-", "-", f"{download:.1f}" if qos else "-",
        f"{upload:.1f}" if qos else "-", f"{rsrp - 3:.1f}", f"{rsrq - 1:.1f}",
        f"{sinr - 2:.1f}", "M",
    ]


def generate(output: Path, rows: int = 240, seed: int = 42) -> Path:
    if rows < 24:
        raise ValueError("Use pelo menos 24 linhas para a demonstração.")
    output.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
        writer.writerow(HEADER)
        writer.writerows(_row(index, rows, rng) for index in range(rows))
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--rows", type=int, default=240)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    path = generate(args.output, rows=args.rows, seed=args.seed)
    print(f"Log sintético criado em: {path.resolve()}")
    print("Uso exclusivo para demonstração; não citar como dado científico.")


if __name__ == "__main__":
    main()
