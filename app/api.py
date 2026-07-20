"""
FastAPI routes. Each endpoint is small and delegates to the modules above.
Every database write goes through an IngestionRun id so the lineage of every
measurement is preserved.
"""
from __future__ import annotations
import json
import logging
import math
import re
import uuid
from datetime import datetime

import httpx
import numpy as np
import pandas as pd
from fastapi import APIRouter, UploadFile, File, Query, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from sqlalchemy import select, desc

from app import __version__
from app.analysis_data import deduplicate_measurements
from app.config import settings, EXPORT_DIR, RAW_DIR
from app.db import SessionLocal, Measurement, IngestionRun, ApiCallLog
from app.parser import parse_log_bytes
from app.audit import open_run, close_run, find_existing_run_by_sha
from app.enrichment import enrich_point
from app.stats import (
    anova_by_factor, pearson_with_fdr, stratified_summary,
    anova_robust, pairwise_hedges_g, summary_by_sector, summary_by_environment,
    minimum_detectable_effect,
)
from app.ml import (
    regression_train, classification_train,
    DEFAULT_FEATURES_REGRESSION, DEFAULT_FEATURES_CLASSIFICATION,
)
from app.unsupervised import (
    unsupervised_analysis, DEFAULT_UNSUPERVISED_FEATURES,
)
from app.sectors import (
    ControlPoint, fit_affine, save_calibration, load_calibration,
    build_sector_geojson, SectorClassifier,
)
from app.sectors.calibration import CALIBRATION_PATH, Calibration
from app.sectors.loader import load_local_sectors
from app.sectors.buffered_classifier import (
    BufferedSectorClassifier, BufferedHit, BUFFER_M_POR_CLASSE,
)


log = logging.getLogger(__name__)
router = APIRouter()


def _make_classifier() -> SectorClassifier:
    """Build a fresh strict classifier from the persisted calibration (or uncalibrated)."""
    cal = load_calibration()
    geo = build_sector_geojson(calibration=cal)
    return SectorClassifier(geo)


def _make_buffered_classifier() -> tuple[BufferedSectorClassifier | None, Calibration | None]:
    """Build a buffered classifier + return the calibration object.
    Returns (None, None) if there is no calibration yet — buffered mode
    requires calibration to convert WGS84 to local metres.
    """
    cal = load_calibration()
    if cal is None:
        return None, None
    secs = load_local_sectors()
    return BufferedSectorClassifier(secs), cal


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------
@router.get("/api/health")
def health() -> dict:
    return {"status": "ok", "version": __version__, "db_url": settings.db_url}


@router.get("/api/dashboard")
def dashboard_data(campaign_id: str | None = None,
                   run_id: int | None = None) -> dict:
    """
    Agregados completos para a aba DASHBOARD. Retorna em um único JSON:
      * KPIs do topo (totais, médias, contagens)
      * Distribuição de RSRP (histograma)
      * RSRP/SINR/ping por categoria (setor, tecnologia, superfície, hora)
      * Distribuição de signal_rating
      * Cobertura espacial por setor
      * Série temporal de RSRP por campanha
    """
    import numpy as np
    raw_df = _load_measurements(run_id=run_id, campaign_id=campaign_id)
    if raw_df.empty:
        return {"empty": True, "message": "Sem dados no banco para os filtros aplicados."}
    df, duplicate_rows, duplicate_key = deduplicate_measurements(raw_df)

    def _stats(col):
        if col not in df.columns:
            return None
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        if s.empty:
            return None
        return {
            "n":      int(s.size),
            "mean":   float(s.mean()),
            "median": float(s.median()),
            "std":    float(s.std()) if s.size > 1 else 0.0,
            "min":    float(s.min()),
            "max":    float(s.max()),
            "p10":    float(s.quantile(0.10)),
            "p25":    float(s.quantile(0.25)),
            "p75":    float(s.quantile(0.75)),
            "p90":    float(s.quantile(0.90)),
        }

    def _counts(col, top=20):
        if col not in df.columns:
            return []
        c = df[col].dropna().astype(str).value_counts().head(top)
        return [{"label": str(k), "n": int(v)} for k, v in c.items()]

    def _by(group_col, value_col):
        """Estatisticas de value_col agregadas por group_col."""
        if group_col not in df.columns or value_col not in df.columns:
            return []
        out = []
        vc = pd.to_numeric(df[value_col], errors="coerce")
        for level, idx in df.groupby(df[group_col].fillna("indefinido")).groups.items():
            sub = vc.loc[idx].dropna()
            if sub.empty:
                continue
            out.append({
                "group":  str(level),
                "n":      int(sub.size),
                "mean":   float(sub.mean()),
                "median": float(sub.median()),
                "p25":    float(sub.quantile(0.25)),
                "p75":    float(sub.quantile(0.75)),
                "min":    float(sub.min()),
                "max":    float(sub.max()),
            })
        out.sort(key=lambda x: -x["n"])
        return out

    # KPIs
    rsrp_stats = _stats("rsrp_dbm")
    sinr_stats = _stats("sinr_db")
    ping_stats = _stats("ping_avg_ms")
    dl_stats   = _stats("test_dl_max_kbps")
    ul_stats   = _stats("test_ul_max_kbps")

    n_5g = int((df.get("network_tech") == "5G").sum()) if "network_tech" in df.columns else 0
    n_4g = int((df.get("network_tech") == "4G").sum()) if "network_tech" in df.columns else 0

    # Sectors with data (via effective)
    sec_col = "sector_code_effective" if "sector_code_effective" in df.columns else "sector_code"
    n_sectors = int(df[sec_col].dropna().nunique()) if sec_col in df.columns else 0
    n_campaigns = int(df["campaign_id"].dropna().nunique()) if "campaign_id" in df.columns else 0

    # Histograma RSRP (bins de 5 dBm)
    hist = None
    if rsrp_stats:
        rsrp_vals = pd.to_numeric(df["rsrp_dbm"], errors="coerce").dropna()
        bins = list(range(-130, -55, 5))
        counts, edges = np.histogram(rsrp_vals, bins=bins)
        hist = {
            "bins":   [f"{int(edges[i])} a {int(edges[i+1])}" for i in range(len(counts))],
            "counts": [int(x) for x in counts],
        }

    # Sample temporal (RSRP por minuto) — limita pra nao explodir
    timeline = []
    if "timestamp_log" in df.columns and "rsrp_dbm" in df.columns:
        d_ts = df[["timestamp_log", "rsrp_dbm", "network_tech"]].dropna()
        d_ts = d_ts.sort_values("timestamp_log")
        # Reamostrar por minuto
        try:
            d_ts = d_ts.set_index(pd.to_datetime(d_ts["timestamp_log"]))
            rsrp_min = d_ts["rsrp_dbm"].resample("1min").mean().dropna()
            timeline = [{"t": str(idx), "rsrp_mean": float(v)}
                         for idx, v in list(rsrp_min.items())[:300]]
        except Exception:                                       # noqa: BLE001
            pass

    # ----- Ajuste log-distância descritivo com referência estimada/declarada
    propagation = None
    try:
        from app.site_estimate import load_active_site, fit_logdist
        site = load_active_site()
        if site and "distance_to_site_est_m" in df.columns:
            sub = df
            if "indoor_outdoor" in sub.columns:
                sub = sub[sub["indoor_outdoor"] == "outdoor"]
            if "band" in sub.columns:
                sub = sub[sub["band"] == site.get("band", "L7")]
            fit = fit_logdist(
                pd.to_numeric(sub["distance_to_site_est_m"], errors="coerce").values,
                pd.to_numeric(sub["rsrp_dbm"], errors="coerce").values,
            )
            o2i = None
            ind = pd.to_numeric(df.loc[df.get("indoor_outdoor") == "indoor", "rsrp_dbm"], errors="coerce").dropna()
            out = pd.to_numeric(df.loc[df.get("indoor_outdoor") == "outdoor", "rsrp_dbm"], errors="coerce").dropna()
            if len(ind) >= 20 and len(out) >= 20:
                o2i = float(out.mean() - ind.mean())
            if fit:
                A, n_exp, r2, npts = fit
                propagation = {
                    "site_lat": site["lat"], "site_lon": site["lon"],
                    "site_source": site.get("source", "estimated"),
                    "intercept_dbm": round(A, 1),
                    "path_loss_exponent": round(n_exp, 2),
                    "r2": round(r2, 3),
                    "n_points": npts,
                    "band": site.get("band", "L7"),
                    "o2i_db": round(o2i, 1) if o2i is not None else None,
                    "caveat": site.get("caveat"),
                }
    except Exception:                                       # noqa: BLE001
        pass

    return {
        "kpis": {
            "n_measurements": int(len(df)),
            "n_measurements_raw": int(len(raw_df)),
            "n_dropped_exact_duplicates": duplicate_rows,
            "n_campaigns":    n_campaigns,
            "n_sectors":      n_sectors,
            "n_5g":           n_5g,
            "n_4g":           n_4g,
            "rsrp":           rsrp_stats,
            "sinr":           sinr_stats,
            "ping":           ping_stats,
            "dl_kbps":        dl_stats,
            "ul_kbps":        ul_stats,
        },
        "propagation":      propagation,
        "data_scope": {
            "grain": "analytical_deduplicated",
            "raw_rows": int(len(raw_df)),
            "analytical_rows": int(len(df)),
            "duplicates_removed": duplicate_rows,
            "duplicate_key": duplicate_key,
        },
        "rsrp_histogram":   hist,
        "signal_rating":    _counts("signal_rating"),
        "by_sector":        _by(sec_col, "rsrp_dbm"),
        "by_tech":          _by("network_tech", "rsrp_dbm"),
        "by_band":          _by("band", "rsrp_dbm"),
        "by_surface":       _by("surface_type", "rsrp_dbm"),
        "by_period":        _by("period_of_day", "rsrp_dbm"),
        "by_environment":   _by("environment_class_effective" if "environment_class_effective" in df.columns else "environment_class", "rsrp_dbm"),
        "sinr_by_sector":   _by(sec_col, "sinr_db"),
        "ping_by_sector":   _by(sec_col, "ping_avg_ms"),
        "timeline":         timeline,
        "campaigns":        _counts("campaign_id"),
    }


@router.get("/api/site/estimate")
def get_site_estimate() -> dict:
    """Referência espacial ativa: declarada ou estimada dos próprios dados."""
    if not settings.enable_site_reference:
        return {
            "available": False,
            "message": "Referência espacial experimental desativada por configuração.",
        }
    from app.site_estimate import load_active_site
    site = load_active_site()
    if not site:
        return {"available": False,
                "message": "Sem posicao salva. POST /api/site/manual (declarar) "
                           "ou POST /api/site/estimate/refit (estimar)."}
    return {"available": True, **site}


@router.post("/api/site/manual")
def set_site_manual(lat: float = Query(..., description="Latitude da referência declarada"),
                    lon: float = Query(..., description="Longitude da referência declarada"),
                    notes: str = Query("", description="Ex.: 'referência observada em campo'")
                    ) -> dict:
    """Declara uma posição de referência informada pelo pesquisador.

    A associação definitiva da portadora LTE/NR deve ser verificada fora do
    app; esta entrada não transforma a coordenada em localização oficial.
    """
    if not settings.enable_site_reference:
        raise HTTPException(403, detail="Ative FEG_ENABLE_SITE_REFERENCE para usar o recurso experimental.")
    from app.site_estimate import save_manual_site
    data = save_manual_site(lat, lon, notes)
    return {"available": True, **data}


@router.post("/api/site/estimate/refit")
def refit_site_estimate() -> dict:
    """Recalcula a referência espacial experimental com os dados atuais."""
    if not settings.enable_site_reference:
        raise HTTPException(403, detail="Ative FEG_ENABLE_SITE_REFERENCE para usar o recurso experimental.")
    from app.site_estimate import estimate_and_save
    df = _load_measurements()
    est = estimate_and_save(df)
    if est is None:
        raise HTTPException(409, detail=(
            "Dados insuficientes para estimar (precisa de >=100 medicoes "
            "outdoor na banda L7)."))
    return {"available": True, **est}


@router.get("/api/summary")
def summary() -> dict:
    """
    Lightweight overview for the dashboard header badge.
    Returns raw and analytical counts, campaigns, runs, effective sector and
    valid historical-weather coverage, plus the calibration state.
    """
    from sqlalchemy import func
    s = SessionLocal()
    try:
        n_meas    = s.query(func.count(Measurement.id)).scalar() or 0
        n_runs    = s.query(func.count(IngestionRun.id)).scalar() or 0
        n_camps   = (s.query(func.count(func.distinct(Measurement.campaign_id)))
                       .filter(Measurement.campaign_id.isnot(None)).scalar() or 0)
    finally:
        s.close()

    frame = _load_measurements()
    analytical, n_duplicates, _ = deduplicate_measurements(frame)
    if analytical.empty:
        n_classified = 0
        weather_counts = {
            "manual_database": 0,
            "manual_notebook": 0,
            "archive_campaign_median": 0,
            "missing": 0,
        }
    else:
        sector_col = (
            "sector_code_effective"
            if "sector_code_effective" in analytical
            else "sector_code"
        )
        n_classified = int(analytical[sector_col].notna().sum())
        if "weather_source_eff" in analytical:
            counts = (
                analytical["weather_source_eff"]
                .fillna("missing")
                .value_counts()
            )
            weather_counts = {str(key): int(value) for key, value in counts.items()}
        else:
            weather_counts = {"missing": int(len(analytical))}
    n_weather_valid = int(
        sum(value for key, value in weather_counts.items() if key != "missing")
    )

    cal = load_calibration()
    cal_state = "uncalibrated"
    if cal is not None:
        cal_state = "synthetic" if cal.looks_synthetic() else "calibrated"

    return {
        "n_measurements": int(n_meas),
        "n_measurements_raw": int(n_meas),
        "n_measurements_analytical": int(len(analytical)),
        "n_duplicates_removed": int(n_duplicates),
        "n_runs":         int(n_runs),
        "n_campaigns":    int(n_camps),
        "n_classified":   int(n_classified),
        # Backward-compatible alias, now with the scientifically valid meaning.
        "n_enriched":     n_weather_valid,
        "n_weather_valid": n_weather_valid,
        "weather_sources": weather_counts,
        "calibration":    cal_state,
    }


# ---------------------------------------------------------------------------
# Upload + parse + (optional) enrich
# ---------------------------------------------------------------------------
@router.post("/api/upload")
async def upload_log(file: UploadFile = File(...),
                     enrich: bool = Query(True, description="Chamar APIs externas para enriquecer"),
                     campaign_id: str = Query("",
                         description="Tag de campanha (ex: 'manha-2026-05-04', 'uti-pico-18h'). "
                                     "Sem isso, dados não podem ser estratificados por horário/campanha."),
                     indoor_outdoor: str = Query("",
                         description="'indoor' ou 'outdoor' — anotação manual do Quadro 2 "
                                     "(propaga para todas as medições deste upload)."),
                     manual_sector: str = Query("",
                         description="Codigo do setor declarado pelo aluno (ex: 'S14'). "
                                     "Propaga para todas as medicoes do upload. Tem prioridade "
                                     "sobre a classificacao automatica nas analises por setor. "
                                     "Vazio = nao informado, sera usada a classificacao por buffer."),
                     surface_type: str = Query("",
                         description="Tipo de superficie predominante: 'grama' | 'terra' | "
                                     "'asfalto' | 'concreto' | 'misto'. Afeta reflexao do "
                                     "sinal. Propaga para todas as medicoes. Vazio = nao informado."),
                     avg_building_height_m: float | None = Query(None,
                         description="Altura media estimada dos predios no entorno (em metros). "
                                     "Preenche a lacuna do OSM. Propaga para todas as medicoes."),
                     avg_tree_height_m: float | None = Query(None,
                         description="Altura media estimada das arvores no entorno (em metros). "
                                     "Preenche a lacuna do OSM. Propaga para todas as medicoes."),
                     # ----- Anotacoes ambientais manuais -----
                     temperature_c: float | None = Query(None,
                         description="Temperatura observada (°C). Manual prevalece sobre Open-Meteo."),
                     humidity: float | None = Query(None,
                         description="Umidade relativa (%). Manual prevalece sobre Open-Meteo."),
                     cloud_cover_pct: float | None = Query(None,
                         description="Cobertura de nuvens (%). Manual prevalece sobre Open-Meteo."),
                     building_count: int | None = Query(None,
                         description="Quantidade de predios contados no entorno. Manual prevalece sobre Overpass."),
                     distance_to_building_m: float | None = Query(None,
                         description="Distancia estimada ao predio mais proximo (m). Manual prevalece sobre Overpass."),
                     tree_count: int | None = Query(None,
                         description="Quantidade de arvores contadas no entorno. Manual prevalece sobre Overpass."),
                     distance_to_tree_m: float | None = Query(None,
                         description="Distancia estimada a arvore mais proxima (m). Manual prevalece sobre Overpass."),
                     vegetation_density: int | None = Query(None,
                         description="Densidade visual de vegetacao (0=nenhuma, 1=esparsa, 2=media, 3=densa). "
                                     "Fallback do NDVI quando GEE/Earth Engine esta offline."),
                     precipitation_status: int | None = Query(None,
                         description="Situacao de chuva (0=seco, 1=garoa, 2=leve, 3=moderada, 4=forte)."),
                     visual_obstruction_grade: int | None = Query(None,
                         description="Obstrucao visual do ceu (0=livre/LoS, 1=parcial, 2=bloqueada/NLoS)."),
                     force: bool = Query(False,
                         description="Reingerir mesmo se a SHA-256 já existir."),
                     ) -> JSONResponse:
    contents = await file.read()
    if not contents:
        raise HTTPException(400, detail="Arquivo vazio")
    if len(contents) > settings.max_upload_bytes:
        raise HTTPException(
            413,
            detail=f"Arquivo excede o limite de {settings.max_upload_bytes} bytes.",
        )

    # Persist a copy of the raw upload for traceability (sha-prefixed name)
    safe_filename = _safe_upload_filename(file.filename)
    suffix = safe_filename.rsplit(".", 1)[-1].lower()
    if suffix not in {"txt", "csv", "tsv", "log"}:
        raise HTTPException(415, detail="Formato não permitido; use TXT, CSV, TSV ou LOG.")
    pr = parse_log_bytes(contents, filename=safe_filename)
    raw_path = RAW_DIR / f"{pr.file_sha256[:12]}__{safe_filename[-100:]}"
    if not raw_path.exists():
        raw_path.write_bytes(contents)

    if pr.df.empty:
        return JSONResponse({
            "status": "error",
            "variant": pr.variant,
            "delimiter": pr.delimiter,
            "warnings": pr.warnings,
            "columns_missing": pr.columns_missing,
            "rows_raw": pr.rows_raw,
            "processed_points": 0,
        }, status_code=200)

    session = SessionLocal()
    try:
        # --- Dedupe by SHA-256 -----------------------------------------------
        # Idempotent: same bytes → same result, no duplicate insertion. Use
        # ?force=true to override (e.g. you want to reingest after recalibrating
        # the sectors and need a fresh run to compare).
        if pr.file_sha256 and not force:
            existing = find_existing_run_by_sha(session, pr.file_sha256)
            if existing is not None:
                session.close()
                return JSONResponse({
                    "status": "already_ingested",
                    "run_id": existing.id,
                    "filename": existing.filename,
                    "rows_valid": existing.rows_valid,
                    "file_sha256": pr.file_sha256,
                    "campaign_id": existing.campaign_id,
                    "started_at": existing.started_at.isoformat() if existing.started_at else None,
                    "message": (
                        f"Arquivo já importado na run #{existing.id} "
                        f"({existing.filename}, {existing.rows_valid} pontos). "
                        f"Use ?force=true para reingerir."
                    ),
                })

        run = open_run(
            session,
            filename=safe_filename,
            file_sha256=pr.file_sha256,
            file_size_bytes=pr.file_size_bytes,
            log_variant=pr.variant,
            delimiter=pr.delimiter,
            columns_detected=pr.columns_detected,
            columns_missing=pr.columns_missing,
            accuracy_threshold_m=settings.gps_acc_low_m,
            campaign_id=(campaign_id.strip() or None),
        )

        df = pr.df.copy()
        n_inserted = 0
        # Sector classifier — uncalibrated unless data/calibration.json exists
        classifier = _make_classifier()
        buffered_cls, buffered_cal = _make_buffered_classifier()
        sector_tally: dict[str, int] = {}
        sector_tally_buffer: dict[str, int] = {}

        async with httpx.AsyncClient(
            timeout=settings.http_timeout_s,
            headers={"User-Agent": settings.http_user_agent,
                     "Accept": "application/json"},
        ) as client:
            for _, row in df.iterrows():
                lat = float(row["latitude"])
                lon = float(row["longitude"])
                rsrp = float(row["rsrp_dbm"])

                env_kwargs: dict = {}
                # Otimizacao: se TODAS as variaveis ambientais foram informadas
                # manualmente, nao chamamos nenhuma API. Se SOMENTE algumas,
                # ainda chamamos (cache resolve a duplicacao). Reduz drasticamente
                # o tempo de upload quando o aluno preenche tudo no painel.
                all_manual_filled = all([
                    temperature_c is not None,
                    humidity is not None,
                    cloud_cover_pct is not None,
                    building_count is not None,
                    distance_to_building_m is not None,
                    tree_count is not None,
                    distance_to_tree_m is not None,
                    avg_building_height_m is not None,
                    avg_tree_height_m is not None,
                    vegetation_density is not None,
                ])
                if enrich and not all_manual_filled:
                    need_alt = (row.get("altitude_source") != "log")
                    ep = await enrich_point(client, session, run.id, lat, lon,
                                            need_altitude=need_alt)
                    env_kwargs = {
                        "temperature_c":  ep.temperature_c,
                        "temperature_status": ep.temperature_status,
                        "humidity":      ep.humidity,
                        "humidity_status": ep.humidity_status,
                        "cloud_cover_pct": ep.cloud_cover_pct,
                        "cloud_cover_status": ep.cloud_cover_status,
                        "cloud_cover_label": ep.cloud_cover_label,
                        "building_count": ep.building_count,
                        "building_status": ep.building_status,
                        "avg_building_height": ep.avg_building_height,
                        "distance_to_building_m": ep.distance_to_building_m,
                        "tree_count":     ep.tree_count,
                        "tree_status":    ep.tree_status,
                        "avg_tree_height_m":  ep.avg_tree_height_m,
                        "distance_to_tree_m": ep.distance_to_tree_m,
                        "tree_density_ndvi": ep.tree_density_ndvi,
                        "ndvi_status":   ep.ndvi_status,
                    }
                    if need_alt and ep.altitude_m is not None:
                        env_kwargs["altitude_m"] = ep.altitude_m
                        env_kwargs["altitude_source"] = "api"

                hit = classifier.classify(lat, lon)
                key = hit.sector_code or "unclassified"
                sector_tally[key] = sector_tally.get(key, 0) + 1

                # Buffered classification (zona de influencia em metros locais)
                if buffered_cls is not None and buffered_cal is not None:
                    bhit = buffered_cls.classify_wgs84(lat, lon, buffered_cal)
                else:
                    bhit = BufferedHit(None, None, None, None, None, None, None)
                bkey = bhit.sector_code_buffer or "unclassified"
                sector_tally_buffer[bkey] = sector_tally_buffer.get(bkey, 0) + 1

                # Build base kwargs from the row, then let env_kwargs (API
                # enrichment) override — that way `altitude_m` from the
                # elevation API replaces a missing log altitude without a
                # duplicate-kwarg TypeError.
                base_kwargs = dict(
                    altitude_m=_safe(row.get("altitude_m")),
                    altitude_source=row.get("altitude_source"),
                )
                base_kwargs.update(env_kwargs)

                m = Measurement(
                    run_id=run.id,
                    campaign_id=run.campaign_id,
                    indoor_outdoor=(indoor_outdoor.strip().lower() or None),
                    frequency_hz=_safe(row.get("frequency_hz")),
                    # pd.isna cobre NaT (data invalida no log) — sem isso o
                    # SQLite explode com "cannot convert float NaN to integer"
                    timestamp_log=(None if pd.isna(row.get("timestamp_log"))
                                   else row.get("timestamp_log")),
                    latitude=lat, longitude=lon,
                    sector_code=hit.sector_code,
                    sector_name=hit.sector_name,
                    environment_class=hit.environment_class,
                    sector_code_buffer=bhit.sector_code_buffer,
                    sector_name_buffer=bhit.sector_name_buffer,
                    environment_class_buffer=bhit.environment_class_buffer,
                    sector_distance_m=bhit.sector_distance_m,
                    sector_code_manual=(manual_sector.strip().upper() or None),
                    surface_type=(surface_type.strip().lower() or None),
                    avg_building_height_manual=avg_building_height_m,
                    avg_tree_height_manual=avg_tree_height_m,
                    # Anotacoes ambientais manuais (override de APIs)
                    temperature_c_manual=temperature_c,
                    humidity_manual=humidity,
                    cloud_cover_pct_manual=cloud_cover_pct,
                    building_count_manual=building_count,
                    distance_to_building_m_manual=distance_to_building_m,
                    tree_count_manual=tree_count,
                    distance_to_tree_m_manual=distance_to_tree_m,
                    vegetation_density_manual=vegetation_density,
                    precipitation_status=precipitation_status,
                    visual_obstruction_grade=visual_obstruction_grade,
                    period_of_day=row.get("period_of_day"),
                    light_condition=row.get("light_condition"),
                    gps_accuracy_m=_safe(row.get("gps_accuracy_m")),
                    gps_quality=row.get("gps_quality"),
                    operator=_str_or_none(row.get("operator")),
                    operator_name=_str_or_none(row.get("operator_name")),
                    cgi=_str_or_none(row.get("cgi")),
                    serving_cell_id=_str_or_none(row.get("serving_cell_id")),
                    serving_cell_lat=_safe(row.get("serving_cell_lat")),
                    serving_cell_lon=_safe(row.get("serving_cell_lon")),
                    distance_to_serving_m=_safe(row.get("distance_to_serving_m")),
                    distance_source=row.get("distance_source"),
                    network_tech=_str_or_none(row.get("network_tech")),
                    network_mode=_str_or_none(row.get("network_mode")),
                    band=_str_or_none(row.get("band")),
                    arfcn=_int_or_none(row.get("arfcn")),
                    bandwidth=_str_or_none(row.get("bandwidth")),
                    speed_kmh=_safe(row.get("speed_kmh")),
                    rsrp_dbm=rsrp, rsrp_status=row.get("rsrp_dbm_status", "ok"),
                    rsrq_db=_safe(row.get("rsrq_db")),
                    rsrq_status=row.get("rsrq_db_status"),
                    sinr_db=_safe(row.get("sinr_db")),
                    sinr_status=row.get("sinr_db_status"),
                    cqi=_int_or_none(row.get("cqi")),
                    cqi_status=row.get("cqi_status"),
                    ltersssi_dbm=_safe(row.get("ltersssi_dbm")),
                    ltersssi_status=row.get("ltersssi_dbm_status"),
                    csi_rsrp_dbm=_safe(row.get("csi_rsrp_dbm")),
                    csi_rsrp_status=row.get("csi_rsrp_dbm_status"),
                    csi_rsrq_db=_safe(row.get("csi_rsrq_db")),
                    csi_rsrq_status=row.get("csi_rsrq_db_status"),
                    csi_snr_db=_safe(row.get("csi_snr_db")),
                    csi_snr_status=row.get("csi_snr_db_status"),
                    ping_avg_ms=_safe(row.get("ping_avg_ms")),
                    ping_avg_status=row.get("ping_avg_ms_status"),
                    ping_min_ms=_safe(row.get("ping_min_ms")),
                    ping_max_ms=_safe(row.get("ping_max_ms")),
                    ping_stdev_ms=_safe(row.get("ping_stdev_ms")),
                    ping_jitter_status=row.get("ping_stdev_ms_status"),
                    ping_loss_pct=_safe(row.get("ping_loss_pct")),
                    dl_bitrate_kbps=_safe(row.get("dl_bitrate_kbps")),
                    dl_bitrate_status=row.get("dl_bitrate_kbps_status"),
                    ul_bitrate_kbps=_safe(row.get("ul_bitrate_kbps")),
                    ul_bitrate_status=row.get("ul_bitrate_kbps_status"),
                    test_dl_max_kbps=_safe(row.get("test_dl_max_kbps")),
                    test_dl_max_status=row.get("test_dl_max_kbps_status"),
                    test_ul_max_kbps=_safe(row.get("test_ul_max_kbps")),
                    test_ul_max_status=row.get("test_ul_max_kbps_status"),
                    event_type=_str_or_none(row.get("event_type")),
                    event_details=_str_or_none(row.get("event_details")),
                    n_neighbors=None,                # TODO: parse_neighbor_columns
                    signal_rating=row.get("signal_rating"),
                    **base_kwargs,
                )
                session.add(m)
                n_inserted += 1

        close_run(session, run,
                  rows_raw=pr.rows_raw,
                  rows_dropped_essential=pr.rows_dropped_essential,
                  rows_dropped_gps=pr.rows_dropped_gps,
                  rows_valid=n_inserted,
                  notes=";".join(pr.warnings) if pr.warnings else None)

        # Audit summary on API calls for this run
        api_summary = (
            session.query(ApiCallLog.api_name, ApiCallLog.status)
            .filter(ApiCallLog.run_id == run.id).all()
        )
        api_table: dict[str, dict[str, int]] = {}
        for name, status in api_summary:
            api_table.setdefault(name, {}).setdefault(status, 0)
            api_table[name][status] += 1

        return JSONResponse({
            "status": "success",
            "run_id": run.id,
            "campaign_id": run.campaign_id,
            "variant": pr.variant,
            "delimiter": pr.delimiter,
            "file_sha256": pr.file_sha256,
            "rows_raw": pr.rows_raw,
            "rows_dropped_essential": pr.rows_dropped_essential,
            "rows_dropped_gps": pr.rows_dropped_gps,
            "rows_inserted": n_inserted,
            "columns_missing": pr.columns_missing,
            "warnings": pr.warnings,
            "api_summary": api_table,
            "sectors": {
                "calibrated": classifier.calibrated,
                "n_sectors": classifier.n_sectors,
                "tally": sector_tally,
            },
            "raw_file_persisted": str(raw_path.relative_to(RAW_DIR.parent)),
        })
    finally:
        session.close()


@router.post("/api/enrich")
async def enrich_existing(run_id: int | None = Query(None,
                              description="Restringe a um run específico"),
                          campaign_id: str | None = Query(None,
                              description="Restringe a uma campanha"),
                          only_missing: bool = Query(True,
                              description="Pula pontos já enriquecidos (status='ok' "
                                          "em temperature_status, building_status, ndvi_status).")
                          ) -> dict:
    """
    Roda enrichment APIs sobre medições já ingeridas. Utilidade real:
    quando o aluno upou rapidamente em campo com `enrich=false` e agora,
    com internet estável, quer popular temperatura/edificações/NDVI sem
    reprocessar o log inteiro.

    Não bloqueia: usa o mesmo `enrich_point` + bucketing geográfico do
    upload. Honra o cache; só pontos com status != 'ok' (em qualquer das
    APIs configuradas) disparam chamadas reais.
    """
    s = SessionLocal()
    try:
        q = s.query(Measurement)
        if run_id is not None:
            q = q.filter(Measurement.run_id == run_id)
        if campaign_id:
            q = q.filter(Measurement.campaign_id == campaign_id)
        rows = q.all()
        if not rows:
            return {"error": "Nenhuma medição corresponde aos filtros."}

        n_total = len(rows)
        n_skipped = 0
        n_done = 0
        async with httpx.AsyncClient(
            timeout=settings.http_timeout_s,
            headers={"User-Agent": settings.http_user_agent,
                     "Accept": "application/json"},
        ) as client:
            for m in rows:
                already_ok = (
                    m.temperature_status == "ok"
                    and m.building_status   == "ok"
                    and (m.ndvi_status     == "ok" or not settings.gee_project)
                )
                if only_missing and already_ok:
                    n_skipped += 1
                    continue
                need_alt = (m.altitude_source != "log") and (m.altitude_m is None)
                ep = await enrich_point(
                    client, s, m.run_id, m.latitude, m.longitude,
                    need_altitude=need_alt,
                )
                m.temperature_c   = ep.temperature_c
                m.temperature_status = ep.temperature_status
                m.humidity        = ep.humidity
                m.humidity_status = ep.humidity_status
                m.cloud_cover_pct = ep.cloud_cover_pct
                m.cloud_cover_status = ep.cloud_cover_status
                m.cloud_cover_label = ep.cloud_cover_label
                m.building_count  = ep.building_count
                m.building_status = ep.building_status
                m.avg_building_height = ep.avg_building_height
                m.distance_to_building_m = ep.distance_to_building_m
                m.tree_count      = ep.tree_count
                m.tree_status     = ep.tree_status
                m.avg_tree_height_m = ep.avg_tree_height_m
                m.distance_to_tree_m = ep.distance_to_tree_m
                m.tree_density_ndvi   = ep.tree_density_ndvi
                m.ndvi_status     = ep.ndvi_status
                if need_alt and ep.altitude_m is not None:
                    m.altitude_m      = ep.altitude_m
                    m.altitude_source = "api"
                n_done += 1
                # Commit periodico: preserva progresso (e o cache de API) mesmo
                # se o processo for interrompido no meio de um backfill longo.
                # Rodadas seguintes com only_missing=true pulam o que ja foi.
                if n_done % 50 == 0:
                    s.commit()
        s.commit()

        # Audit summary of API calls done in this enrich run
        from sqlalchemy import func
        api_summary: dict[str, dict[str, int]] = {}
        if rows:
            run_ids = list({m.run_id for m in rows})
            calls = (s.query(ApiCallLog.api_name, ApiCallLog.status,
                             func.count(ApiCallLog.id))
                       .filter(ApiCallLog.run_id.in_(run_ids))
                       .group_by(ApiCallLog.api_name, ApiCallLog.status).all())
            for name, status, n in calls:
                api_summary.setdefault(name, {})[status or "null"] = int(n)

        return {
            "status": "success",
            "filters": {"run_id": run_id, "campaign_id": campaign_id,
                        "only_missing": only_missing},
            "n_measurements_total": n_total,
            "n_skipped_already_ok": n_skipped,
            "n_enriched_this_call": n_done,
            "api_summary_for_runs": api_summary,
        }
    finally:
        s.close()


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
@router.get("/api/statistics")
def statistics(
    factor: str = Query("network_tech",
                        description="ANOVA factor (network_tech, sector_code, environment_class, campaign_id)"),
    response: str = Query("rsrp_dbm",
                          description="Continuous response variable (rsrp_dbm, sinr_db, ping_avg_ms, ...)"),
    robust: bool = Query(False,
                          description="Run anova_robust (Welch + Kruskal + Shapiro/Levene + ω²)"),
    run_id: int | None = None,
    campaign_id: str | None = None,
) -> dict:
    df = _load_measurements(run_id=run_id, campaign_id=campaign_id)
    if df.empty:
        return {"error": "Sem dados no banco. Faça upload primeiro."}
    df, n_duplicates, duplicate_key = deduplicate_measurements(df)

    if robust:
        anova = anova_robust(df, factor_col=factor, response_col=response)
        anova["pairwise_hedges_g"] = pairwise_hedges_g(
            df, factor_col=factor, response_col=response,
        )
        anova["minimum_detectable_effect"] = minimum_detectable_effect(
            df, factor_col=factor, response_col=response,
        )
    else:
        anova = anova_by_factor(df, factor_col=factor, response_col=response)

    corr = pearson_with_fdr(
        df,
        cols=["rsrp_dbm", "sinr_db", "cqi", "ltersssi_dbm",
              "ping_avg_ms", "ping_stdev_ms", "test_dl_max_kbps",
              "altitude_m", "distance_to_serving_m",
              "temperature_c_eff", "humidity_eff", "cloud_cover_pct_eff",
              "building_count", "avg_building_height", "tree_density_ndvi"],
        alpha=0.05,
    )

    summary = stratified_summary(df)

    return {
        "n_total": int(len(df)),
        "anova": anova,
        "pearson_fdr": corr,
        "stratified_by_tech": summary,
        "policy": {
            "grain": "analytical_deduplicated",
            "duplicates_removed": n_duplicates,
            "duplicate_key": duplicate_key,
            "fillna": "PROHIBITED — pairwise complete observations, no zero filling",
            "fdr_method": "Benjamini-Hochberg",
            "anova_min_group": 5,
        },
    }


@router.get("/api/statistics/by_sector")
def statistics_by_sector(run_id: int | None = None) -> dict:
    df = _load_measurements(run_id=run_id)
    if df.empty:
        return {"error": "Sem dados."}
    df, _, _ = deduplicate_measurements(df)
    return summary_by_sector(df)


@router.get("/api/statistics/by_environment")
def statistics_by_environment(run_id: int | None = None) -> dict:
    df = _load_measurements(run_id=run_id)
    if df.empty:
        return {"error": "Sem dados."}
    df, _, _ = deduplicate_measurements(df)
    return summary_by_environment(df)


def _parse_features_param(features: str | None) -> list[str] | None:
    if not features:
        return None
    parts = [p.strip() for p in features.split(",") if p.strip()]
    return parts or None


@router.get("/api/ml/regression")
def ml_regression(target: str = Query("rsrp_dbm"),
                  features: str | None = Query(None,
                      description="Comma-separated override; empty = default."),
                  group_by: str | None = Query(
                      "campaign_id",
                      description=(
                          "campaign_id | sector_code_effective | date | run_id | "
                          "random. Grouped validation is the scientific default."
                      ),
                  ),
                  compute_importance: bool = Query(True),
                  run_id: int | None = None) -> dict:
    df = _load_measurements(run_id=run_id)
    if df.empty:
        return {"error": "Sem dados."}
    feat = _parse_features_param(features) or DEFAULT_FEATURES_REGRESSION
    return regression_train(
        df, features=feat, target=target, group_by=group_by,
        compute_importance=compute_importance,
    )


@router.get("/api/ml/classification")
def ml_classification(target: str = Query("signal_rating",
                          description="signal_rating | environment_class | network_tech"),
                      features: str | None = Query(None,
                          description="Comma-separated override; empty = default."),
                      group_by: str | None = Query(
                          "campaign_id",
                          description=(
                              "campaign_id | sector_code_effective | date | run_id | "
                              "random."
                          ),
                      ),
                      compute_importance: bool = Query(True),
                      run_id: int | None = None) -> dict:
    df = _load_measurements(run_id=run_id)
    if df.empty:
        return {"error": "Sem dados."}
    feat = _parse_features_param(features) or DEFAULT_FEATURES_CLASSIFICATION
    return classification_train(
        df, features=feat, target=target, group_by=group_by,
        compute_importance=compute_importance,
    )


@router.get("/api/ml/unsupervised")
def ml_unsupervised(
    features: str | None = Query(
        None, description="Comma-separated numeric override; empty = default."
    ),
    run_id: int | None = None,
    campaign_id: str | None = None,
) -> dict:
    """Run PCA, k-means and DBSCAN on measured propagation profiles."""
    df = _load_measurements(run_id=run_id, campaign_id=campaign_id)
    if df.empty:
        return {"error": "Sem dados."}
    selected_features = (
        _parse_features_param(features) or DEFAULT_UNSUPERVISED_FEATURES
    )
    return unsupervised_analysis(df, features=selected_features)


# ---------------------------------------------------------------------------
# Audit endpoints
# ---------------------------------------------------------------------------
@router.get("/api/campaigns")
def list_campaigns() -> dict:
    """
    List distinct campaign_id values present in the DB with summary stats:
    n_runs, n_measurements, n_distinct_sectors, time range.

    Goal: let the dashboard offer a campaign filter and let the analyst spot
    "S07 was only measured at 14h on one day" before reporting an effect that
    is really an hour-of-day artefact.
    """
    from sqlalchemy import func
    s = SessionLocal()
    try:
        rows = (
            s.query(
                Measurement.campaign_id,
                func.count(Measurement.id),
                func.count(func.distinct(Measurement.sector_code)),
                func.min(Measurement.timestamp_log),
                func.max(Measurement.timestamp_log),
            )
            .group_by(Measurement.campaign_id)
            .all()
        )
        runs_per_campaign = dict(
            s.query(IngestionRun.campaign_id, func.count(IngestionRun.id))
             .group_by(IngestionRun.campaign_id).all()
        )
        out = []
        for cid, n_meas, n_sec, t0, t1 in rows:
            out.append({
                "campaign_id": cid,         # may be None
                "n_runs": int(runs_per_campaign.get(cid, 0)),
                "n_measurements": int(n_meas or 0),
                "n_distinct_sectors": int(n_sec or 0),
                "time_first": t0.isoformat() if t0 else None,
                "time_last":  t1.isoformat() if t1 else None,
            })
        out.sort(key=lambda x: (x["campaign_id"] is None, x["campaign_id"] or ""))
        return {"campaigns": out, "n_total": len(out)}
    finally:
        s.close()


@router.get("/api/sectors/temporal_coverage")
def sector_temporal_coverage(run_id: int | None = None) -> dict:
    """
    Per-sector temporal coverage table — answers the banca's "was S07 measured
    at multiple times of day or always at 14h?" question. For each sector,
    reports n_measurements, distinct campaigns it appears in, distinct hours of
    day, time range.
    """
    df = _load_measurements(run_id=run_id)
    if df.empty:
        return {"error": "Sem dados."}
    if "sector_code" not in df.columns:
        return {"error": "sector_code ausente — calibre e reclassifique."}

    # Ensure timestamp_log is datetime; rows without timestamps still count.
    if "timestamp_log" in df.columns:
        df = df.copy()
        df["timestamp_log"] = pd.to_datetime(df["timestamp_log"], errors="coerce")
        df["hour"] = df["timestamp_log"].dt.hour

    grouper = df["sector_code"].fillna("unclassified")
    out = []
    for code, grp in df.groupby(grouper, dropna=False):
        camps = (grp.get("campaign_id", pd.Series(dtype=object))
                    .dropna().unique().tolist())
        hours = (grp.get("hour", pd.Series(dtype=float))
                    .dropna().astype(int).unique().tolist()) \
                if "hour" in grp.columns else []
        ts = grp.get("timestamp_log", pd.Series(dtype="datetime64[ns]")).dropna()
        out.append({
            "sector_code": str(code),
            "n_measurements": int(len(grp)),
            "n_campaigns":   int(len(camps)),
            "campaigns":     [str(c) for c in camps],
            "n_distinct_hours": int(len(hours)),
            "hours":         sorted(hours),
            "time_first":    ts.min().isoformat() if not ts.empty else None,
            "time_last":     ts.max().isoformat() if not ts.empty else None,
        })
    out.sort(key=lambda x: x["sector_code"])
    return {
        "sectors": out,
        "policy": ("n_distinct_hours == 1 sinaliza viés temporal — qualquer "
                   "efeito atribuído ao setor pode ser efeito do horário."),
    }


@router.get("/api/audit/runs")
def list_runs(limit: int = 20) -> list[dict]:
    s = SessionLocal()
    try:
        rows = (s.query(IngestionRun)
                  .order_by(desc(IngestionRun.id)).limit(limit).all())
        out = []
        for r in rows:
            out.append({
                "id": r.id, "filename": r.filename,
                "sha256": (r.file_sha256 or "")[:16],
                "variant": r.log_variant, "delimiter": r.delimiter,
                "campaign_id": r.campaign_id,
                "rows_raw": r.rows_raw,
                "rows_dropped_essential": r.rows_dropped_essential,
                "rows_dropped_gps": r.rows_dropped_gps,
                "rows_valid": r.rows_valid,
                "started_at": r.started_at.isoformat() if r.started_at else None,
                "ended_at":   r.ended_at.isoformat()   if r.ended_at   else None,
                "accuracy_threshold_m": r.accuracy_threshold_m,
                "notes": r.notes,
            })
        return out
    finally:
        s.close()


@router.get("/api/audit/run/{run_id}")
def run_detail(run_id: int) -> dict:
    s = SessionLocal()
    try:
        r = s.query(IngestionRun).filter_by(id=run_id).first()
        if not r:
            raise HTTPException(404, detail="run_id not found")
        api_calls = (
            s.query(ApiCallLog.api_name, ApiCallLog.status)
            .filter(ApiCallLog.run_id == run_id).all()
        )
        api_table: dict[str, dict[str, int]] = {}
        for name, status in api_calls:
            api_table.setdefault(name, {}).setdefault(status, 0)
            api_table[name][status] += 1

        statuses = {}
        # Field-level coverage in this run
        cols_with_status = ["rsrp", "rsrq", "sinr", "cqi", "ltersssi",
                            "csi_rsrp", "csi_rsrq", "csi_snr",
                            "ping_avg", "ping_jitter",
                            "dl_bitrate", "ul_bitrate",
                            "test_dl_max", "test_ul_max",
                            "ndvi", "humidity", "temperature", "building"]
        for cs in cols_with_status:
            col_name = f"{cs}_status"
            try:
                rows = s.execute(
                    select(getattr(Measurement, col_name)).where(Measurement.run_id == run_id)
                ).all()
                tally: dict[str, int] = {}
                for (v,) in rows:
                    tally[v or "null"] = tally.get(v or "null", 0) + 1
                statuses[cs] = tally
            except Exception:
                continue

        return {
            "id": r.id, "filename": r.filename,
            "sha256": r.file_sha256,
            "variant": r.log_variant, "delimiter": r.delimiter,
            "campaign_id": r.campaign_id,
            "rows": {
                "raw": r.rows_raw,
                "dropped_essential": r.rows_dropped_essential,
                "dropped_gps": r.rows_dropped_gps,
                "valid": r.rows_valid,
            },
            "columns_detected": json.loads(r.columns_detected or "[]"),
            "columns_missing":  json.loads(r.columns_missing  or "[]"),
            "api_summary": api_table,
            "field_status_tally": statuses,
            "started_at": r.started_at.isoformat() if r.started_at else None,
            "ended_at":   r.ended_at.isoformat()   if r.ended_at   else None,
            "notes": r.notes,
        }
    finally:
        s.close()


# ---------------------------------------------------------------------------
# Geo endpoints
# ---------------------------------------------------------------------------
@router.get("/api/points")
def points(run_id: int | None = None, limit: int = 5000) -> list[dict]:
    df = _load_measurements(run_id=run_id, limit=limit)
    if df.empty:
        return []
    df, _, _ = deduplicate_measurements(df)
    keep = ["id", "latitude", "longitude", "rsrp_dbm", "sinr_db", "cqi",
            "network_tech", "band", "frequency_hz",
            "ping_avg_ms", "ping_stdev_ms",
            "test_dl_max_kbps", "test_ul_max_kbps",
            "signal_rating", "gps_accuracy_m", "gps_quality",
            # Setor: campos brutos + manual + derivado efetivo
            "sector_code", "sector_name", "environment_class",
            "sector_code_buffer", "sector_code_manual",
            "sector_code_effective", "environment_class_effective",
            "sector_distance_m",
            "campaign_id", "indoor_outdoor", "surface_type",
            "period_of_day", "precipitation_status", "visual_obstruction_grade",
            # Quadro 2 ambientais — bruto + efetivo (manual > API):
            "altitude_m",
            "temperature_c", "temperature_c_eff",
            "humidity", "humidity_eff",
            "cloud_cover_pct", "cloud_cover_pct_eff", "cloud_cover_label",
            "weather_source_eff", "weather_missing_eff",
            # Predios: API + efetivo
            "building_count", "building_count_eff",
            "avg_building_height", "avg_building_height_eff_m",
            "distance_to_building_m", "distance_to_building_m_eff",
            # Arvores: API + efetivo
            "tree_count", "tree_count_eff",
            "avg_tree_height_m", "avg_tree_height_eff_m",
            "distance_to_tree_m", "distance_to_tree_m_eff",
            "tree_density_ndvi", "vegetation_density_manual",
            # Distancia ao sitio ESTIMADO (auto-calibracao; ver site_estimate.py)
            "distance_to_site_est_m"]
    keep = [c for c in keep if c in df.columns]
    out = df[keep].to_dict(orient="records")
    # JSON-safe NaNs
    for r in out:
        for k, v in list(r.items()):
            if isinstance(v, float) and math.isnan(v):
                r[k] = None
    return out


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------
# --- Quadro 2 column projection -------------------------------------------
# Maps the 18 variables of the research plan's Quadro 2 to columns currently
# stored in `measurements`, with the requested unit conversions. The order
# below is the order they appear in Quadro 2 of the project.
QUADRO2_COLUMNS: list[tuple[str, str, callable | None]] = [
    # (column_label_in_xlsx, source_column,                    transform)
    ("Avaliação do sinal",        "signal_rating",             None),
    ("Localização do quadrante",  "indoor_outdoor",            None),
    ("Tipo de superfície",        "surface_type",              None),
    ("Download (Mbps)",           "test_dl_max_kbps",          lambda v: v / 1000.0 if v is not None else None),
    ("Upload (Mbps)",             "test_ul_max_kbps",          lambda v: v / 1000.0 if v is not None else None),
    ("Latência (ms)",             "ping_avg_ms",               None),
    ("Jitter (ms)",               "ping_stdev_ms",             None),
    ("Potência do sinal (dBm)",   "rsrp_dbm",                  None),
    ("Frequência (Hz)",           "frequency_hz",              None),
    ("Altitude (m)",              "altitude_m",                None),
    ("Cobertura de nuvens",       "cloud_cover_label",         None),
    ("Temperatura (°C)",          "temperature_c",             None),
    ("Umidade relativa (%)",      "humidity",                  None),
    # Vegetação — usa altura efetiva (manual > API)
    ("Quantidade de árvores",     "tree_count",                None),
    ("Altura média das árvores (m)", "avg_tree_height_eff_m",  None),
    ("Distância média para as árvores (m)", "distance_to_tree_m", None),
    # Construção — usa altura efetiva (manual > API)
    ("Quantidade de prédios",     "building_count",            None),
    ("Altura média dos prédios (m)", "avg_building_height_eff_m", None),
    ("Distância média para os prédios (m)", "distance_to_building_m", None),
    # Mínimo de rastreabilidade para revisar a exportação:
    ("latitude",                  "latitude",                  None),
    ("longitude",                 "longitude",                 None),
    ("timestamp_log",             "timestamp_log",             None),
    # Setor: usa efetivo (manual > buffer > strict)
    ("setor",                     "sector_code_effective",     None),
    ("ambiente",                  "environment_class_effective", None),
    ("campanha",                  "campaign_id",               None),
    ("network_tech",              "network_tech",              None),
    ("banda",                     "band",                      None),
]


def _project_quadro2(df: pd.DataFrame) -> pd.DataFrame:
    """Build a DataFrame whose columns match Quadro 2 exactly, with units
    converted (Mbps from kbps, etc.). Missing source columns become empty."""
    out = pd.DataFrame()
    for label, src, fn in QUADRO2_COLUMNS:
        if src in df.columns:
            col = df[src]
            out[label] = col.map(fn) if fn is not None else col
        else:
            out[label] = None
    return out


@router.get("/api/export")
def export(run_id: int | None = None,
           campaign_id: str | None = None,
           mode: str = Query("scientific", pattern="^(scientific|full)$",
               description="'scientific' = só Quadro 2 (recomendado para "
                           "relatório). 'full' = todas as colunas internas "
                           "(útil para debugging).")) -> FileResponse:
    df = _load_measurements(run_id=run_id, campaign_id=campaign_id)
    if df.empty:
        raise HTTPException(404, detail="Sem dados para exportar.")

    if mode == "scientific":
        df, _, _ = deduplicate_measurements(df)
        out_df = _project_quadro2(df)
        suffix = "quadro2"
    else:
        out_df = df
        suffix = "completo"

    fname = (f"feg_research_{suffix}_"
             f"{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_"
             f"{uuid.uuid4().hex[:6]}.xlsx")
    path = EXPORT_DIR / fname
    out_df.to_excel(path, index=False)
    return FileResponse(path=str(path), filename=fname,
                        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")


# ---------------------------------------------------------------------------
# Sectors / calibration
# ---------------------------------------------------------------------------
@router.get("/api/sectors/calibration")
def get_calibration() -> dict:
    cal = load_calibration()
    if cal is None:
        return {
            "calibrated": False,
            "calibration_path": str(CALIBRATION_PATH),
            "message": (
                "Sem calibração ativa. Forneça pontos de controle em POST "
                "/api/sectors/calibration ou rode "
                "`python -m app.sectors.fit_calibration`."
            ),
        }
    payload = {
        "calibrated": True,
        "calibration_path": str(CALIBRATION_PATH),
        "looks_synthetic": cal.looks_synthetic(),
        **cal.to_dict(),
    }
    if payload["looks_synthetic"]:
        payload["warning"] = (
            "Esta calibração parece sintética/de teste (RMS < 1 cm ou notes "
            "contém pytest/test/synthetic). NÃO use para classificar dados "
            "reais — meça pontos de controle no campo e refaça o ajuste."
        )
    return payload


@router.post("/api/sectors/calibration")
def post_calibration(payload: dict) -> dict:
    """
    Body:
    {
      "control_points": [
         {"name": "...", "x_local": 32.5, "y_local": 25.0,
          "lat": -23.21, "lon": -45.878},
         ...
      ],
      "notes": "optional free text",
      "max_rms_m": 5.0   // optional quality gate; rejects fit if exceeded
    }
    """
    cps_raw = payload.get("control_points") or []
    if not isinstance(cps_raw, list) or len(cps_raw) < 3:
        raise HTTPException(400, detail="Forneça ao menos 3 control_points.")
    try:
        cps = [
            ControlPoint(
                name=str(c["name"]),
                x_local=float(c["x_local"]),
                y_local=float(c["y_local"]),
                lat=float(c["lat"]),
                lon=float(c["lon"]),
            )
            for c in cps_raw
        ]
    except (KeyError, TypeError, ValueError) as e:
        raise HTTPException(400, detail=f"control_points inválido: {e}")

    cal = fit_affine(cps, notes=str(payload.get("notes", "")))
    max_rms = payload.get("max_rms_m")
    if max_rms is not None and cal.rms_m > float(max_rms):
        raise HTTPException(
            422,
            detail=f"RMS {cal.rms_m:.3f} m excede max_rms_m={max_rms}. "
                   f"Calibração NÃO foi salva."
        )
    save_calibration(cal)
    return {"saved": True, "calibration_path": str(CALIBRATION_PATH),
            **cal.to_dict()}


@router.get("/api/sectors")
def get_sectors_geojson() -> dict:
    cal = load_calibration()
    return build_sector_geojson(calibration=cal)


@router.get("/api/sectors/aggregates")
def sectors_aggregates(
    metric: str = Query("rsrp_dbm",
                        description="Numeric column to aggregate per sector"),
    agg: str = Query("mean", pattern="^(mean|median)$"),
    run_id: int | None = None,
) -> dict:
    """
    Per-sector aggregate of `metric` for choropleth visualisation.
    Returns: { metric, agg, scale:{min,max,p10,p90}, sectors:[{sector_code, n, value, ...}] }
    Sectors with n_valid < 3 are returned with value=None (not coloured).
    """
    df = _load_measurements(run_id=run_id)
    if df.empty or metric not in df.columns:
        return {"error": "Sem dados ou métrica inexistente.", "metric": metric}
    df, _, _ = deduplicate_measurements(df)
    sector_col = (
        "sector_code_effective"
        if "sector_code_effective" in df
        else "sector_code"
    )
    environment_col = (
        "environment_class_effective"
        if "environment_class_effective" in df
        else "environment_class"
    )
    if sector_col not in df.columns:
        return {"error": "sector_code ausente. Calibre e reclassifique."}

    out = []
    vals_for_scale: list[float] = []
    grouper = df[sector_col].fillna("unclassified")
    for code, grp in df.groupby(grouper, dropna=False):
        s = grp[metric].dropna()
        n_valid = int(len(s))
        if n_valid < 3:
            value = None
        else:
            value = float(s.mean() if agg == "mean" else s.median())
            vals_for_scale.append(value)
        env = grp[environment_col].dropna().unique().tolist() \
            if environment_col in grp.columns else []
        name = grp["sector_name"].dropna().unique().tolist() \
            if "sector_name" in grp.columns else []
        out.append({
            "sector_code": str(code),
            "sector_name": name[0] if name else None,
            "environment_class": env[0] if env else None,
            "n_rows": int(len(grp)),
            "n_valid": n_valid,
            "value": value,
        })

    out.sort(key=lambda x: x["sector_code"])
    if vals_for_scale:
        arr = np.array(vals_for_scale)
        scale = {"min": float(arr.min()), "max": float(arr.max()),
                 "p10": float(np.percentile(arr, 10)),
                 "p90": float(np.percentile(arr, 90))}
    else:
        scale = {"min": None, "max": None, "p10": None, "p90": None}

    return {"metric": metric, "agg": agg, "scale": scale,
            "min_n_valid": 3, "sectors": out}


@router.get("/api/sectors/classify")
def classify_point(lat: float = Query(...), lon: float = Query(...)) -> dict:
    cls = _make_classifier()
    hit = cls.classify(lat, lon)
    buffered_cls, buffered_cal = _make_buffered_classifier()
    if buffered_cls is not None and buffered_cal is not None:
        bhit = buffered_cls.classify_wgs84(lat, lon, buffered_cal)
    else:
        bhit = BufferedHit(None, None, None, None, None, None, None)
    return {
        "calibrated": cls.calibrated,
        "lat": lat, "lon": lon,
        # Estrito (DENTRO do retangulo)
        "sector_code": hit.sector_code,
        "sector_name": hit.sector_name,
        "environment_class": hit.environment_class,
        # Buffer (zona de influencia)
        "sector_code_buffer":       bhit.sector_code_buffer,
        "sector_name_buffer":       bhit.sector_name_buffer,
        "environment_class_buffer": bhit.environment_class_buffer,
        "sector_distance_m":        bhit.sector_distance_m,
        "buffer_m_por_classe":      dict(BUFFER_M_POR_CLASSE),
    }


@router.post("/api/sectors/reclassify")
def reclassify_existing(run_id: int | None = None) -> dict:
    """
    Re-runs both classifiers (estrito + buffer) over already-ingested
    measurements. Useful after fitting (or refitting) the calibration, or
    after changing the buffer radii.
    """
    cls = _make_classifier()
    if not cls.calibrated:
        raise HTTPException(409, detail=(
            "Sem calibração ativa — não há polígonos em WGS84 para classificar."
        ))

    buffered_cls, buffered_cal = _make_buffered_classifier()

    s = SessionLocal()
    try:
        q = s.query(Measurement)
        if run_id is not None:
            q = q.filter(Measurement.run_id == run_id)
        updated = 0
        tally_strict: dict[str, int] = {}
        tally_buffer: dict[str, int] = {}
        for m in q.all():
            hit = cls.classify(m.latitude, m.longitude)
            m.sector_code = hit.sector_code
            m.sector_name = hit.sector_name
            m.environment_class = hit.environment_class
            key_s = hit.sector_code or "unclassified"
            tally_strict[key_s] = tally_strict.get(key_s, 0) + 1

            if buffered_cls is not None and buffered_cal is not None:
                bhit = buffered_cls.classify_wgs84(m.latitude, m.longitude, buffered_cal)
            else:
                bhit = BufferedHit(None, None, None, None, None, None, None)
            m.sector_code_buffer       = bhit.sector_code_buffer
            m.sector_name_buffer       = bhit.sector_name_buffer
            m.environment_class_buffer = bhit.environment_class_buffer
            m.sector_distance_m        = bhit.sector_distance_m
            key_b = bhit.sector_code_buffer or "unclassified"
            tally_buffer[key_b] = tally_buffer.get(key_b, 0) + 1

            updated += 1
        s.commit()
        return {
            "run_id": run_id,
            "updated": updated,
            "tally_strict": tally_strict,
            "tally_buffer": tally_buffer,
            "n_sectors": cls.n_sectors,
            "buffer_m_por_classe": dict(BUFFER_M_POR_CLASSE),
        }
    finally:
        s.close()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _load_measurements(*, run_id: int | None = None,
                       campaign_id: str | None = None,
                       limit: int | None = None) -> pd.DataFrame:
    s = SessionLocal()
    try:
        q = s.query(Measurement)
        if run_id is not None:
            q = q.filter(Measurement.run_id == run_id)
        if campaign_id:
            q = q.filter(Measurement.campaign_id == campaign_id)
        if limit:
            q = q.limit(limit)
        df = pd.read_sql(q.statement, s.bind)
    finally:
        s.close()
    # Acrescenta colunas derivadas (_effective) com COALESCE manual > API.
    # Toda análise (ML, stats, exports, painel) usa essas colunas por padrão.
    from app.derivations import add_effective_columns
    df = add_effective_columns(df)
    return df


def _safe_upload_filename(filename: str | None) -> str:
    """Return a basename safe for persistence below ``data/raw``."""
    basename = (filename or "upload.txt").replace("\\", "/").split("/")[-1]
    basename = re.sub(r"[^A-Za-z0-9._-]+", "_", basename).strip("._")
    return basename[:100] or "upload.txt"


def _safe(v):
    if v is None:
        return None
    if isinstance(v, float) and math.isnan(v):
        return None
    if isinstance(v, (np.floating,)):
        x = float(v)
        return None if math.isnan(x) else x
    if isinstance(v, (np.integer,)):
        return int(v)
    try:
        x = float(v)
        return None if math.isnan(x) else x
    except (TypeError, ValueError):
        return None


def _str_or_none(v) -> str | None:
    if v is None:
        return None
    if isinstance(v, float) and math.isnan(v):
        return None
    s = str(v).strip()
    return s or None


def _int_or_none(v) -> int | None:
    x = _safe(v)
    return int(x) if x is not None else None
