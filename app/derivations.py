"""
Derivacoes "efetivas" que combinam anotacao manual com dado de API.

Politica do projeto:
  - Para clima historico: MANUAL > Open-Meteo Archive > ausente.
  - Os campos meteorologicos legados ``temperature_c``, ``humidity`` e
    ``cloud_cover_pct`` vieram do endpoint ``current`` e, portanto, NUNCA
    participam das colunas efetivas historicas.
  - Para as demais variaveis, a precedencia e documentada em cada bloco.
  - Nenhuma imputacao silenciosa: NaN continua sendo NaN.

Esta funcao deve ser chamada uma vez logo apos carregar o DataFrame do
banco. Todas as analises (ML, estatistica, exports) usam as colunas
'_effective' resultantes.
"""
from __future__ import annotations
import pandas as pd

from app.config import settings
from app.sectors.legend import LEGEND, SPECIAL_SECTORS


def _empty_series(df: pd.DataFrame, dtype="object") -> pd.Series:
    return pd.Series([None] * len(df), index=df.index, dtype=dtype)


def _legend_env_for(code: str | None) -> str | None:
    """Mapeia codigo de setor para a classe ambiental.
    Aceita S01..S21 (poligonos) e codigos especiais (VIA, EST)."""
    if not code:
        return None
    c = str(code).strip().upper()
    if c in SPECIAL_SECTORS:
        return SPECIAL_SECTORS[c].environment_class
    try:
        idx = int(c.lstrip("S").lstrip("0") or "0")
        meta = LEGEND.get(idx)
        return meta.environment_class if meta else None
    except (ValueError, AttributeError):
        return None


def add_effective_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Acrescenta as colunas derivadas:
      * sector_code_effective       = COALESCE(manual, buffer, strict)
      * sector_name_effective       = COALESCE(name_buffer, name_strict)
      * environment_class_effective = COALESCE(env_via_LEGEND(manual),
                                               env_buffer, env_strict)
      * avg_building_height_eff_m   = COALESCE(manual, api)
      * avg_tree_height_eff_m       = COALESCE(manual, api)
      * temperature_c_eff           = COALESCE(manual, archive)
      * humidity_eff                = COALESCE(manual, archive)
      * cloud_cover_pct_eff         = COALESCE(manual, archive)
      * weather_source_eff          = manual_database | manual_notebook |
                                      archive_campaign_median | missing
      * weather_missing_eff         = indicador explicito de ausencia
      * building_count_eff          = COALESCE(manual, api)
      * distance_to_building_m_eff  = COALESCE(manual, api)
      * tree_count_eff              = COALESCE(manual, api)
      * distance_to_tree_m_eff      = COALESCE(manual, api)

    Idempotente: se as colunas '_effective' ja existirem, sao sobrescritas.
    Se as colunas-fonte nao existirem (ex.: banco antigo), a derivacao
    daquela coluna e silenciosamente pulada.
    """
    if df.empty:
        return df

    df = df.copy()

    # -------- Setor --------
    if "sector_code_manual" in df.columns or "sector_code_buffer" in df.columns:
        manual = df["sector_code_manual"] if "sector_code_manual" in df.columns else pd.Series([None] * len(df), index=df.index)
        buf    = df["sector_code_buffer"] if "sector_code_buffer" in df.columns else pd.Series([None] * len(df), index=df.index)
        strict = df["sector_code"]        if "sector_code"        in df.columns else pd.Series([None] * len(df), index=df.index)
        df["sector_code_effective"] = manual.fillna(buf).fillna(strict)

    # Nome do setor: preferir buffer (mais cobertura), depois strict
    if "sector_name_buffer" in df.columns or "sector_name" in df.columns:
        buf  = df["sector_name_buffer"] if "sector_name_buffer" in df.columns else pd.Series([None] * len(df), index=df.index)
        stct = df["sector_name"]        if "sector_name"        in df.columns else pd.Series([None] * len(df), index=df.index)
        df["sector_name_effective"] = buf.fillna(stct)

    # Classe ambiental: prioridade = LEGEND[manual] > buffer > strict
    if "environment_class_buffer" in df.columns or "environment_class" in df.columns or "sector_code_manual" in df.columns:
        # Via LEGEND a partir do setor manual
        if "sector_code_manual" in df.columns:
            env_legend = df["sector_code_manual"].map(_legend_env_for)
        else:
            env_legend = _empty_series(df)
        buf  = df["environment_class_buffer"] if "environment_class_buffer" in df.columns else _empty_series(df)
        stct = df["environment_class"]        if "environment_class"        in df.columns else _empty_series(df)
        df["environment_class_effective"] = env_legend.fillna(buf).fillna(stct)

    # -------- Altura media de predios --------
    if "avg_building_height_manual" in df.columns or "avg_building_height" in df.columns:
        man = df["avg_building_height_manual"] if "avg_building_height_manual" in df.columns else pd.Series([None] * len(df), index=df.index, dtype="float64")
        api = df["avg_building_height"]        if "avg_building_height"        in df.columns else pd.Series([None] * len(df), index=df.index, dtype="float64")
        df["avg_building_height_eff_m"] = pd.to_numeric(man, errors="coerce").fillna(pd.to_numeric(api, errors="coerce"))

    # -------- Altura media de arvores --------
    if "avg_tree_height_manual" in df.columns or "avg_tree_height_m" in df.columns:
        man = df["avg_tree_height_manual"] if "avg_tree_height_manual" in df.columns else pd.Series([None] * len(df), index=df.index, dtype="float64")
        api = df["avg_tree_height_m"]      if "avg_tree_height_m"      in df.columns else pd.Series([None] * len(df), index=df.index, dtype="float64")
        df["avg_tree_height_eff_m"] = pd.to_numeric(man, errors="coerce").fillna(pd.to_numeric(api, errors="coerce"))

    # -------- Clima historico (manual > Open-Meteo Archive > ausente) -----
    # IMPORTANTE: as colunas sem sufixo ``_archive`` sao mantidas apenas para
    # rastreabilidade do enriquecimento legado via endpoint ``current``. Um
    # valor meteorologico atual nao pode ser associado retroativamente a uma
    # medicao de campo e, por isso, nao e usado como fallback aqui.
    climate_pairs = [
        ("temperature_c_manual",   "temperature_c_archive",   "temperature_c_eff"),
        ("humidity_manual",        "humidity_archive",        "humidity_eff"),
        ("cloud_cover_pct_manual", "cloud_cover_pct_archive", "cloud_cover_pct_eff"),
    ]
    for man_col, archive_col, eff_col in climate_pairs:
        if man_col in df.columns or archive_col in df.columns:
            man = (
                pd.to_numeric(df[man_col], errors="coerce")
                if man_col in df.columns
                else _empty_series(df, dtype="float64")
            )
            campaign_archive_col = f"{archive_col}_campaign_median"
            if campaign_archive_col in df.columns:
                archive = pd.to_numeric(df[campaign_archive_col], errors="coerce")
            elif archive_col in df.columns:
                archive = pd.to_numeric(df[archive_col], errors="coerce")
            else:
                archive = _empty_series(df, dtype="float64")
            df[eff_col] = man.fillna(archive)

    climate_effective = [
        column
        for column in ("temperature_c_eff", "humidity_eff", "cloud_cover_pct_eff")
        if column in df.columns
    ]
    if climate_effective:
        complete_manual = pd.Series(True, index=df.index)
        complete_archive = pd.Series(True, index=df.index)
        any_manual = pd.Series(False, index=df.index)
        for man_col, archive_col, _ in climate_pairs:
            manual_available = (
                pd.to_numeric(df[man_col], errors="coerce").notna()
                if man_col in df.columns
                else False
            )
            complete_manual &= manual_available
            any_manual |= manual_available
            campaign_archive_col = f"{archive_col}_campaign_median"
            complete_archive &= (
                pd.to_numeric(df[campaign_archive_col], errors="coerce").notna()
                if campaign_archive_col in df.columns
                else (
                    pd.to_numeric(df[archive_col], errors="coerce").notna()
                    if archive_col in df.columns
                    else False
                )
            )

        complete_effective = df[climate_effective].notna().all(axis=1)
        source = pd.Series("missing", index=df.index, dtype="object")
        source.loc[complete_effective] = "mixed_manual_archive"
        source.loc[complete_archive & ~any_manual] = "archive_campaign_median"
        source.loc[complete_manual] = "manual_database"
        if "manual_weather_provenance_v5" in df.columns:
            declared = df["manual_weather_provenance_v5"].astype("string")
            source.loc[complete_manual & declared.eq("manual_notebook_user_declaration")] = (
                "manual_notebook"
            )
        df["weather_source_eff"] = source
        df["weather_missing_eff"] = ~complete_effective

    # -------- Construcao e vegetacao: CONTAGENS/DISTANCIAS = API primeiro --
    # Decisao do aluno (2026-07-07): para contagem e distancia de predios e
    # arvores, o Overpass/OSM (valor POR PONTO, raio de 50 m) e mais fiel que
    # a anotacao manual (que era UM numero para a campanha inteira e tinha
    # escalas inconsistentes entre coletas). O manual permanece como reserva
    # quando a API nao tem o dado.  [Excecao: ALTURAS continuam manual>API,
    # porque o OSM local nao tem a tag 'height'.]
    for (man_col, api_col, eff_col) in [
        ("building_count_manual",         "building_count",          "building_count_eff"),
        ("distance_to_building_m_manual", "distance_to_building_m",  "distance_to_building_m_eff"),
        ("tree_count_manual",             "tree_count",              "tree_count_eff"),
        ("distance_to_tree_m_manual",     "distance_to_tree_m",      "distance_to_tree_m_eff"),
    ]:
        if man_col in df.columns or api_col in df.columns:
            man = df[man_col] if man_col in df.columns else _empty_series(df, dtype="float64")
            api = df[api_col] if api_col in df.columns else _empty_series(df, dtype="float64")
            df[eff_col] = pd.to_numeric(api, errors="coerce").fillna(pd.to_numeric(man, errors="coerce"))

    # -------- Referência espacial experimental (desativada por padrão) -----
    if settings.enable_site_reference:
        try:
            from app.site_estimate import load_active_site, haversine_m_vec

            site = load_active_site()
            if site and "latitude" in df.columns and "longitude" in df.columns:
                lat = pd.to_numeric(df["latitude"], errors="coerce")
                lon = pd.to_numeric(df["longitude"], errors="coerce")
                df["distance_to_site_est_m"] = haversine_m_vec(
                    lat.values, lon.values, float(site["lat"]), float(site["lon"])
                )
                df["site_source"] = site.get("source", "estimated")
        except Exception:                                   # noqa: BLE001
            pass

    return df
