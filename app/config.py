"""
Central configuration. All thresholds and external endpoints declared here so
they can be audited and versioned with the codebase. Override via environment
variables (FEG_*) when needed.
"""
from __future__ import annotations
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR     = PROJECT_ROOT / "data"
RAW_DIR      = DATA_DIR / "raw"
EXPORT_DIR   = DATA_DIR / "exports"
DB_DIR       = DATA_DIR / "db"
for _d in (RAW_DIR, EXPORT_DIR, DB_DIR):
    _d.mkdir(parents=True, exist_ok=True)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="FEG_", env_file=".env", extra="ignore")

    # --- Database -----------------------------------------------------------
    db_url: str = f"sqlite:///{DB_DIR / 'feg_research.db'}"

    # --- GPS quality thresholds (metres) ------------------------------------
    # Used to classify and to filter measurements before sector-level analyses.
    gps_acc_high_m:   float = 10.0
    gps_acc_normal_m: float = 25.0
    gps_acc_low_m:    float = 50.0   # above this → discarded

    # --- Signal rating thresholds (dBm) -------------------------------------
    # The complete six-level rule is implemented in parser._signal_rating.
    # These two values are retained for backward-compatible filtering only.
    rsrp_excelent_dbm: float = -85.0
    rsrp_good_dbm:     float = -105.0

    # --- External APIs ------------------------------------------------------
    open_meteo_forecast_url:  str = "https://api.open-meteo.com/v1/forecast"
    open_meteo_archive_url:   str = "https://archive-api.open-meteo.com/v1/archive"
    # Open-Meteo deprecated `elevation-api.open-meteo.com`; current host is
    # `api.open-meteo.com/v1/elevation` (same JSON shape).
    open_meteo_elevation_url: str = "https://api.open-meteo.com/v1/elevation"
    overpass_url:             str = "https://overpass-api.de/api/interpreter"
    overpass_radius_m:        int = 50
    http_timeout_s:           float = 10.0
    # Overpass and many other public APIs require a descriptive User-Agent.
    # Without it, Overpass returns HTTP 406 (Not Acceptable).
    http_user_agent:          str = "feg-unesp-rf/0.3 (local research app)"

    # --- Earth Engine -------------------------------------------------------
    gee_project: str = ""             # set via env FEG_GEE_PROJECT; empty → disabled
    gee_max_cloud_pct: int = 15

    # --- ML -----------------------------------------------------------------
    cv_splits: int  = 5
    cv_repeats: int = 3
    random_state: int = 42

    # --- Experimental descriptive site reference ---------------------------
    # Disabled in the public profile. It may be enabled for controlled local
    # studies, but never represents an official antenna coordinate.
    enable_site_reference: bool = False

    # --- Audit / parser -----------------------------------------------------
    null_markers: tuple[str, ...] = ("-", "", "nan", "NaN", "NULL", "null", "None")

    # --- Server -------------------------------------------------------------
    host: str = "127.0.0.1"
    port: int = 8000
    max_upload_bytes: int = 25 * 1024 * 1024


settings = Settings()
