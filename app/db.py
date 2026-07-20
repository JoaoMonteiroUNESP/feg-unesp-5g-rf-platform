"""
Database schema. Each numeric metric carries a parallel `<field>_status`
column with one of:
    - 'ok'             : value came from log/api as a valid number
    - 'missing_field'  : field was present in source but blank or '-'
    - 'api_failed'     : enrichment API errored out for this point
    - 'cached'         : value was fetched from local cache
    - 'log_absent'     : column not present in this log variant
    - 'wifi_invalid'   : point was collected with the handset on Wi-Fi (eduroam);
                         the RSRP/RSRQ/SINR remain valid (passive radio reads),
                         but ping/throughput went through Wi-Fi, not the cellular
                         network, so those QoS values are nulled and flagged.
This makes the difference between 'measured zero' and 'unknown' explicit and
auditable. The pipeline NEVER substitutes a constant for unknown values.
"""
from __future__ import annotations
from datetime import datetime
from sqlalchemy import (
    Column, Integer, Float, String, DateTime, Text, ForeignKey, Index, create_engine
)
from sqlalchemy.orm import declarative_base, sessionmaker, relationship

from app.config import settings


engine = create_engine(
    settings.db_url,
    # timeout=30: se outro processo estiver escrevendo, ESPERA ate 30 s pela
    # liberacao do banco em vez de falhar na hora com "database is locked".
    connect_args=({"check_same_thread": False, "timeout": 30}
                  if settings.db_url.startswith("sqlite") else {}),
    future=True,
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine, future=True)
Base = declarative_base()


# ---------------------------------------------------------------------------
# Audit tables
# ---------------------------------------------------------------------------
class IngestionRun(Base):
    __tablename__ = "ingestion_runs"
    id = Column(Integer, primary_key=True)
    filename       = Column(String, nullable=False)
    file_sha256    = Column(String, nullable=False, index=True)
    file_size_bytes = Column(Integer)
    log_variant    = Column(String)            # gnettrack_full / gnettrack_cellfind / unknown
    delimiter      = Column(String)            # 'tab' / 'comma' / 'semicolon'

    rows_raw                = Column(Integer, default=0)
    rows_dropped_essential  = Column(Integer, default=0)  # missing lat/lon/rsrp
    rows_dropped_gps        = Column(Integer, default=0)  # accuracy too low
    rows_valid              = Column(Integer, default=0)

    columns_detected = Column(Text)   # JSON list
    columns_missing  = Column(Text)   # JSON list of expected-but-absent

    accuracy_threshold_m = Column(Float)
    started_at = Column(DateTime, default=datetime.utcnow)
    ended_at   = Column(DateTime)
    notes      = Column(Text)

    # P6 — campaign tagging. Free-text identifier supplied at upload time
    # (e.g. "manha-2026-05-04", "uti-pico-18h"). Propagated to every
    # Measurement so analyses can stratify by campaign or run two-way ANOVA
    # (sector × campaign).
    campaign_id = Column(String, index=True)

    measurements = relationship("Measurement", back_populates="run")
    api_calls    = relationship("ApiCallLog",  back_populates="run")


class ApiCallLog(Base):
    __tablename__ = "api_call_log"
    id = Column(Integer, primary_key=True)
    run_id     = Column(Integer, ForeignKey("ingestion_runs.id"), index=True)
    timestamp  = Column(DateTime, default=datetime.utcnow)
    api_name   = Column(String, nullable=False)  # open_meteo / overpass / gee / open_meteo_elev
    lat = Column(Float)
    lon = Column(Float)
    status        = Column(String, nullable=False)  # ok|failed|cached|skipped
    http_code     = Column(Integer)
    error_message = Column(Text)
    latency_ms    = Column(Float)

    run = relationship("IngestionRun", back_populates="api_calls")


class GeoCache(Base):
    __tablename__ = "geo_cache"
    id = Column(Integer, primary_key=True)
    cache_key  = Column(String, unique=True, index=True)   # api:lat_r4:lon_r4
    api_name   = Column(String, nullable=False)
    payload_json = Column(Text)
    fetched_at = Column(DateTime, default=datetime.utcnow)


# ---------------------------------------------------------------------------
# Measurements
# ---------------------------------------------------------------------------
class Measurement(Base):
    __tablename__ = "measurements"
    id = Column(Integer, primary_key=True)
    run_id    = Column(Integer, ForeignKey("ingestion_runs.id"), index=True)
    timestamp_log = Column(DateTime, index=True)        # from log file (local time)
    inserted_at   = Column(DateTime, default=datetime.utcnow)

    # GPS -------------------------------------------------------------------
    latitude       = Column(Float, nullable=False)
    longitude      = Column(Float, nullable=False)
    gps_accuracy_m = Column(Float)
    gps_quality    = Column(String)        # high|normal|low
    altitude_m       = Column(Float)
    altitude_source  = Column(String)      # log|api|missing

    # Cell / radio context --------------------------------------------------
    operator         = Column(String)
    operator_name    = Column(String)
    cgi              = Column(String)
    serving_cell_id  = Column(String)
    serving_cell_lat = Column(Float)
    serving_cell_lon = Column(Float)
    distance_to_serving_m = Column(Float)
    distance_source   = Column(String)     # log|computed|missing
    network_tech     = Column(String, index=True)   # 5G|4G
    network_mode     = Column(String)               # 5G NSA|LTE|...
    band             = Column(String)
    arfcn            = Column(Integer)
    bandwidth        = Column(String)
    speed_kmh        = Column(Float)

    # RF metrics + per-field status -----------------------------------------
    rsrp_dbm = Column(Float);  rsrp_status = Column(String)
    rsrq_db  = Column(Float);  rsrq_status = Column(String)
    sinr_db  = Column(Float);  sinr_status = Column(String)
    cqi      = Column(Integer); cqi_status = Column(String)
    ltersssi_dbm = Column(Float); ltersssi_status = Column(String)
    csi_rsrp_dbm = Column(Float); csi_rsrp_status = Column(String)
    csi_rsrq_db  = Column(Float); csi_rsrq_status = Column(String)
    csi_snr_db   = Column(Float); csi_snr_status  = Column(String)

    # QoS / throughput ------------------------------------------------------
    ping_avg_ms    = Column(Float); ping_avg_status = Column(String)
    ping_min_ms    = Column(Float)
    ping_max_ms    = Column(Float)
    ping_stdev_ms  = Column(Float); ping_jitter_status = Column(String)
    ping_loss_pct  = Column(Float)
    dl_bitrate_kbps  = Column(Float); dl_bitrate_status = Column(String)
    ul_bitrate_kbps  = Column(Float); ul_bitrate_status = Column(String)
    test_dl_max_kbps = Column(Float); test_dl_max_status = Column(String)
    test_ul_max_kbps = Column(Float); test_ul_max_status = Column(String)

    # Tipo de conexao de dados no momento da coleta (coluna DataConnection_Type
    # do log G-NetTrack): 'M' = dados moveis (Vivo) | 'WIFI' = eduroam.
    # Quando 'WIFI', as metricas de QoS (ping/DL/UL/testes) foram medidas sobre
    # o Wi-Fi e por isso sao anuladas (status 'wifi_invalid'); RSRP/RSRQ/SINR
    # permanecem validos por serem leituras passivas do radio celular.
    data_connection_type = Column(String, index=True)   # 'M' | 'WIFI' | null

    # Mobility / event context ---------------------------------------------
    event_type    = Column(String)     # e.g. CELL_RESELECTION_4G4G, IRAT_*, PERIODIC
    event_details = Column(String)
    n_neighbors   = Column(Integer)

    # Environmental enrichment (API; optional) ------------------------------
    temperature_c        = Column(Float); temperature_status = Column(String)
    humidity             = Column(Float); humidity_status    = Column(String)
    cloud_cover_pct      = Column(Float); cloud_cover_status = Column(String)
    cloud_cover_label    = Column(String)        # "Sem nuvens" | "Parcialmente" | "Nublado"
    building_count       = Column(Integer); building_status  = Column(String)
    avg_building_height  = Column(Float)
    distance_to_building_m = Column(Float)       # haversine ao prédio mais próximo
    tree_count           = Column(Integer); tree_status      = Column(String)
    avg_tree_height_m    = Column(Float)
    distance_to_tree_m   = Column(Float)
    tree_density_ndvi    = Column(Float); ndvi_status        = Column(String)
    # Frequência derivada (ARFCN → Hz). Mesma origem do parser, mas
    # facilita ANOVA por banda em Hz nominais.
    frequency_hz         = Column(Float)
    # Anotação manual da campanha — Indoor/Outdoor por upload (Quadro 2).
    indoor_outdoor       = Column(String, index=True)   # 'indoor'|'outdoor'|null
    # Anotação manual: tipo de superficie predominante onde a coleta ocorreu.
    # Afeta reflexao do sinal de radio. Opcoes:
    #   'grama' | 'terra' | 'asfalto' | 'concreto' | 'misto' | null
    # Propaga para todas as medicoes do upload.
    surface_type         = Column(String, index=True)
    # Anotacao manual: altura media (em metros) dos predios e das arvores no
    # entorno da coleta. Preenche a lacuna do OSM (tag 'height' raramente
    # existe no centro de Guaratinguetá). Quando informado, tem prioridade
    # sobre 'avg_building_height' / 'avg_tree_height_m' devolvidos pela API.
    # Propaga para todas as medicoes do upload.
    avg_building_height_manual = Column(Float)
    avg_tree_height_manual     = Column(Float)
    # ----- Anotacoes ambientais manuais (adicionado 2026-06-08) -----
    # Politica: se informado, tem prioridade sobre a API; se vazio, API tenta.
    # Clima manual. Para análise histórica, o fallback válido é o arquivo
    # Open-Meteo alinhado ao período da campanha; os campos sem sufixo são
    # legados e permanecem apenas para rastreabilidade.
    temperature_c_manual       = Column(Float)
    humidity_manual            = Column(Float)
    cloud_cover_pct_manual     = Column(Float)
    temperature_c_archive      = Column(Float)
    humidity_archive           = Column(Float)
    cloud_cover_pct_archive    = Column(Float)
    temperature_c_archive_campaign_median = Column(Float)
    humidity_archive_campaign_median      = Column(Float)
    cloud_cover_pct_archive_campaign_median = Column(Float)
    weather_archive_hour       = Column(String)
    weather_archive_query_id   = Column(String)
    manual_weather_provenance_v5 = Column(String)
    weather_source_v5          = Column(String)
    weather_missing_v5         = Column(Integer)
    # Construcao (Overpass seria fallback):
    building_count_manual          = Column(Integer)
    distance_to_building_m_manual  = Column(Float)
    # Vegetacao (Overpass + Sentinel-2 seria fallback):
    tree_count_manual           = Column(Integer)
    distance_to_tree_m_manual   = Column(Float)
    # Densidade visual de vegetacao (fallback se GEE/NDVI offline):
    # 0 = nenhuma, 1 = esparsa, 2 = media, 3 = densa
    vegetation_density_manual   = Column(Integer)
    # ----- Variaveis ordinais puramente manuais -----
    # 0=seco, 1=garoa, 2=chuva leve, 3=chuva moderada, 4=chuva forte
    precipitation_status        = Column(Integer)
    # 0=visao livre (LoS), 1=parcial, 2=bloqueada (NLoS)
    visual_obstruction_grade    = Column(Integer)
    # ----- Variaveis temporais derivadas automaticamente -----
    # manha (06-11) | tarde (12-17) | noite (18-23) | madrugada (00-05)
    period_of_day               = Column(String, index=True)
    # 'dia' | 'noite' — derivado de timestamp + sunrise/sunset
    light_condition             = Column(String)

    # Derived / labels (filled by analysis layer) ---------------------------
    # Strict classification: ponto literalmente DENTRO do retangulo do setor.
    sector_code       = Column(String, index=True)   # P1+P2: classifier output (estrito)
    sector_name       = Column(String)
    environment_class = Column(String)               # edificado|aberto|arborizado|fora
    # Buffered classification: setor mais proximo dentro do raio de influencia
    # (15m edificado / 5m aberto / 10m arborizado). Captura pontos do perimetro
    # imediato — caminho na calcada do DFI conta como DFI.
    sector_code_buffer       = Column(String, index=True)
    sector_name_buffer       = Column(String)
    environment_class_buffer = Column(String)
    sector_distance_m        = Column(Float)         # metros ate a borda do poligono (negativo se dentro)
    # Manual override: setor declarado pelo aluno no upload. NULL quando nao
    # informado — analises usam COALESCE(manual, buffer, strict). Sempre
    # preferir 'manual' quando ele existir, porque os poligonos da planilha
    # original tem desalinhamento conhecido com a geografia real do campus.
    sector_code_manual = Column(String, index=True)
    signal_rating     = Column(String)               # six labels derived from RSRP
    campaign_id       = Column(String, index=True)   # P6: copied from IngestionRun

    run = relationship("IngestionRun", back_populates="measurements")

    __table_args__ = (
        Index("ix_meas_tech_time", "network_tech", "timestamp_log"),
    )


class NeighborCell(Base):
    __tablename__ = "neighbor_cells"
    id = Column(Integer, primary_key=True)
    measurement_id = Column(Integer, ForeignKey("measurements.id"), index=True)
    rank = Column(Integer)            # 1..18
    network_tech = Column(String)
    cell_name    = Column(String)
    cell_id      = Column(String)
    lac          = Column(Integer)
    pci          = Column(Integer)
    arfcn        = Column(Integer)
    rxlev_dbm    = Column(Float)
    qual         = Column(Float)
    distance_m   = Column(Float)
    bearing_deg  = Column(Float)


def init_db() -> None:
    Base.metadata.create_all(bind=engine)
    _apply_lightweight_migrations()


def _apply_lightweight_migrations() -> None:
    """
    Add columns introduced AFTER an earlier `init_db()` ran, without forcing
    the developer to delete the SQLite file. Idempotent: each ALTER is wrapped
    in a try/except so re-running on an up-to-date schema is a no-op.

    We only handle additive migrations (new columns). Anything more invasive
    (renames, type changes) should be a real Alembic migration; we don't ship
    Alembic for an academic project where the alternative is `make clean`.
    """
    from sqlalchemy import inspect
    insp = inspect(engine)

    additive = [
        # (table, column_name, DDL fragment)
        ("ingestion_runs", "campaign_id",          "VARCHAR"),
        ("measurements",   "campaign_id",          "VARCHAR"),
        # Quadro 2 fields added 2026-05-08
        ("measurements",   "cloud_cover_pct",      "FLOAT"),
        ("measurements",   "cloud_cover_status",   "VARCHAR"),
        ("measurements",   "cloud_cover_label",    "VARCHAR"),
        ("measurements",   "distance_to_building_m", "FLOAT"),
        ("measurements",   "tree_count",           "INTEGER"),
        ("measurements",   "tree_status",          "VARCHAR"),
        ("measurements",   "avg_tree_height_m",    "FLOAT"),
        ("measurements",   "distance_to_tree_m",   "FLOAT"),
        ("measurements",   "frequency_hz",         "FLOAT"),
        ("measurements",   "indoor_outdoor",       "VARCHAR"),
        # Buffered sector classification (adicionado 2026-06-08)
        ("measurements",   "sector_code_buffer",        "VARCHAR"),
        ("measurements",   "sector_name_buffer",        "VARCHAR"),
        ("measurements",   "environment_class_buffer",  "VARCHAR"),
        ("measurements",   "sector_distance_m",         "FLOAT"),
        # Manual override do setor (adicionado 2026-06-08)
        ("measurements",   "sector_code_manual",        "VARCHAR"),
        # Caracterizacao ambiental funcional (adicionado 2026-06-08)
        ("measurements",   "surface_type",                "VARCHAR"),
        ("measurements",   "avg_building_height_manual",  "FLOAT"),
        ("measurements",   "avg_tree_height_manual",      "FLOAT"),
        # Anotacoes ambientais manuais (adicionado 2026-06-08 - 2a leva)
        ("measurements",   "temperature_c_manual",        "FLOAT"),
        ("measurements",   "humidity_manual",             "FLOAT"),
        ("measurements",   "cloud_cover_pct_manual",      "FLOAT"),
        # Clima histórico válido (Open-Meteo Archive), adicionado 2026-07.
        ("measurements",   "temperature_c_archive",       "FLOAT"),
        ("measurements",   "humidity_archive",            "FLOAT"),
        ("measurements",   "cloud_cover_pct_archive",     "FLOAT"),
        ("measurements",   "temperature_c_archive_campaign_median", "FLOAT"),
        ("measurements",   "humidity_archive_campaign_median",      "FLOAT"),
        ("measurements",   "cloud_cover_pct_archive_campaign_median", "FLOAT"),
        ("measurements",   "weather_archive_hour",        "VARCHAR"),
        ("measurements",   "weather_archive_query_id",     "VARCHAR"),
        ("measurements",   "manual_weather_provenance_v5", "VARCHAR"),
        ("measurements",   "weather_source_v5",            "VARCHAR"),
        ("measurements",   "weather_missing_v5",           "INTEGER"),
        ("measurements",   "building_count_manual",       "INTEGER"),
        ("measurements",   "distance_to_building_m_manual", "FLOAT"),
        ("measurements",   "tree_count_manual",           "INTEGER"),
        ("measurements",   "distance_to_tree_m_manual",   "FLOAT"),
        ("measurements",   "vegetation_density_manual",   "INTEGER"),
        ("measurements",   "precipitation_status",        "INTEGER"),
        ("measurements",   "visual_obstruction_grade",    "INTEGER"),
        ("measurements",   "period_of_day",               "VARCHAR"),
        ("measurements",   "light_condition",             "VARCHAR"),
        # Tipo de conexao de dados M/WIFI (adicionado 2026-07-10)
        ("measurements",   "data_connection_type",        "VARCHAR"),
    ]
    with engine.begin() as conn:
        for table, col, ddl in additive:
            if not insp.has_table(table):
                continue
            cols = {c["name"] for c in insp.get_columns(table)}
            if col in cols:
                continue
            try:
                conn.exec_driver_sql(f"ALTER TABLE {table} ADD COLUMN {col} {ddl}")
            except Exception:                                  # noqa: BLE001
                pass

    # Best-effort indexes for new columns (SQLite tolerates IF NOT EXISTS).
    with engine.begin() as conn:
        for table, col in [("ingestion_runs", "campaign_id"),
                           ("measurements",   "campaign_id"),
                           ("measurements",   "sector_code_buffer"),
                           ("measurements",   "sector_code_manual"),
                           ("measurements",   "surface_type"),
                           ("measurements",   "period_of_day")]:
            try:
                conn.exec_driver_sql(
                    f"CREATE INDEX IF NOT EXISTS ix_{table}_{col} "
                    f"ON {table}({col})"
                )
            except Exception:                                  # noqa: BLE001
                pass


def get_session():
    """Yield a session; caller is responsible for closing or use as context."""
    return SessionLocal()
