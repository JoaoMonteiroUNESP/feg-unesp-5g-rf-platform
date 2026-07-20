"""
Referência espacial ESTIMADA a partir do sinal por auto-calibração.

Metodo
------
Grid search sobre posicoes candidatas: para cada candidata, calcula a
distancia de todas as medicoes outdoor em banda L7 e ajusta por minimos
quadrados o modelo log-distancia

    RSRP = A - 10 * n * log10(d)

A posicao escolhida e a que maximiza o R² com n > 0 (fisica valida:
sinal cai com a distância). Trata-se de uma referência empírica para análise
relativa, não de uma localização oficial de infraestrutura.

Honestidade metodologica
------------------------
A posição é ESTIMADA a partir dos próprios dados de RSRP — não é a
coordenada oficial da torre e não resolve a associação LTE↔NR. Por isso:
  * o resultado e salvo em data/estimated_site.json com metodo, data e
    parametros do ajuste (auditavel);
  * a coluna derivada chama-se `distance_to_site_est_m` ("_est_" de
    estimada) e NAO entra nas features padrao de ML (seria parcialmente
    circular: a posicao foi otimizada contra o proprio RSRP).
"""
from __future__ import annotations
import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from app.config import DATA_DIR

SITE_PATH = DATA_DIR / "estimated_site.json"
MANUAL_SITE_PATH = DATA_DIR / "site_manual.json"


def load_estimated_site(path: Path = SITE_PATH) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:                                       # noqa: BLE001
        return None


def load_manual_site(path: Path = MANUAL_SITE_PATH) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:                                       # noqa: BLE001
        return None


def save_manual_site(lat: float, lon: float, notes: str = "",
                     path: Path = MANUAL_SITE_PATH) -> dict:
    """Posição de referência declarada pelo pesquisador.

    A declaração tem prioridade sobre a auto-calibração, mas não equivale à
    coordenada oficial nem confirma a portadora NR em uma sessão NSA.
    """
    data = {
        "lat": round(float(lat), 6),
        "lon": round(float(lon), 6),
        "source": "manual",
        "method": "posição de referência declarada pelo pesquisador",
        "notes": notes,
        "declared_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
    }
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False),
                    encoding="utf-8")
    return data


def load_active_site() -> dict | None:
    """Site ativo para calculo de distancias: MANUAL tem prioridade sobre
    o ESTIMADO (mesma politica do resto do projeto: dado declarado pelo
    aluno vence dado inferido)."""
    manual = load_manual_site()
    if manual:
        return {**manual, "source": "manual"}
    est = load_estimated_site()
    if est:
        return {**est, "source": "estimated"}
    return None


def haversine_m_vec(lat_arr, lon_arr, lat2: float, lon2: float):
    """Distancia em metros de um vetor de pontos ate (lat2, lon2)."""
    R = 6371000.0
    p1 = np.radians(np.asarray(lat_arr, dtype=float))
    dp = np.radians(lat2 - np.asarray(lat_arr, dtype=float))
    dl = np.radians(lon2 - np.asarray(lon_arr, dtype=float))
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * math.cos(math.radians(lat2)) * np.sin(dl / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def fit_logdist(d_m, rsrp):
    """OLS de RSRP = A - 10 n log10(d). Retorna (A, n, r2, n_pontos) ou None."""
    d_m = np.asarray(d_m, dtype=float)
    rsrp = np.asarray(rsrp, dtype=float)
    mask = (d_m > 10) & np.isfinite(d_m) & np.isfinite(rsrp)
    if mask.sum() < 20:
        return None
    x = np.log10(d_m[mask]); y = rsrp[mask]
    X = np.column_stack([np.ones_like(x), x])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    yhat = X @ beta
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    n_exp = -float(beta[1]) / 10.0
    # Broad plausibility gate. A numerical fit outside this range is not
    # displayed as an empirical propagation descriptor.
    if not np.isfinite(n_exp) or not 0.0 < n_exp <= 10.0:
        return None
    return float(beta[0]), n_exp, float(r2), int(mask.sum())


def estimate_site(df: pd.DataFrame, *, band: str = "L7") -> dict | None:
    """Grid search (grosso ±2 km, refino ±200 m). Outdoor + banda unica."""
    sub = df.copy()
    if "indoor_outdoor" in sub.columns:
        sub = sub[sub["indoor_outdoor"] == "outdoor"]
    if "band" in sub.columns:
        sub = sub[sub["band"] == band]
    sub = sub.dropna(subset=["latitude", "longitude", "rsrp_dbm"])
    if len(sub) < 100:
        return None

    lat = sub["latitude"].values.astype(float)
    lon = sub["longitude"].values.astype(float)
    rsrp = sub["rsrp_dbm"].values.astype(float)

    lat0, lon0 = float(np.mean(lat)), float(np.mean(lon))
    best = None
    for span, steps in [(0.02, 41), (0.002, 21)]:
        if best is not None:
            lat0, lon0 = best[0], best[1]
        for la in np.linspace(lat0 - span, lat0 + span, steps):
            for lo in np.linspace(lon0 - span, lon0 + span, steps):
                d = haversine_m_vec(lat, lon, la, lo)
                fit = fit_logdist(d, rsrp)
                if fit is None:
                    continue
                A, n, r2, npts = fit
                if n <= 0:          # fisica invalida
                    continue
                if best is None or r2 > best[4]:
                    best = (la, lo, A, n, r2, npts)

    if best is None:
        return None
    bla, blo, A, n, r2, npts = best
    d = haversine_m_vec(lat, lon, bla, blo)
    return {
        "lat": round(float(bla), 6),
        "lon": round(float(blo), 6),
        "intercept_dbm": round(A, 1),
        "path_loss_exponent": round(n, 2),
        "r2": round(r2, 3),
        "n_points_fit": npts,
        "band": band,
        "distance_min_m": round(float(np.min(d)), 0),
        "distance_max_m": round(float(np.max(d)), 0),
        "method": ("auto-calibracao: grid search minimizando erro do modelo "
                   "log-distancia sobre medicoes outdoor"),
        "caveat": ("posicao ESTIMADA a partir dos proprios dados de RSRP; "
                   "nao e a coordenada oficial da torre"),
        "fitted_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
    }


def estimate_and_save(df: pd.DataFrame, path: Path = SITE_PATH) -> dict | None:
    est = estimate_site(df)
    if est is not None:
        path.write_text(json.dumps(est, indent=2, ensure_ascii=False),
                        encoding="utf-8")
    return est
