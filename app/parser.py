"""
G-NetTrack Pro log parser. Detects two real-world variants observed in the
field campaign and produces a *typed* DataFrame plus a structured audit
record.

Variants
--------
gnettrack_full     : the full export (~200 cols, includes PINGAVG, TESTDOWNLINK,
                     Altitude, ARFCN, BAND, BANDWIDTH, EVENT, CSI_*, neighbors,
                     etc.).
gnettrack_cellfind : the *_cellfind.txt variant (~51 cols, includes
                     Cell_Latitude, Cell_Longitude, NCellName1..6 but NO
                     PINGAVG / TEST*).

Policy
------
* NULL markers ('-', '', 'NaN', 'None') are converted to NaN.
* Each numeric metric receives a parallel `<col>_status` column:
    - 'ok'           value is a real number;
    - 'missing_field' the source cell was blank or '-';
    - 'log_absent'   the column itself isn't in this log variant.
* No constant default is ever silently substituted.
* Rows missing latitude, longitude or rsrp_dbm are dropped (counted in audit).
* Rows whose GPS Accuracy exceeds the configured threshold are dropped (also
  counted; see config.gps_acc_low_m).
"""
from __future__ import annotations
import io
import math
import hashlib
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from app.config import settings


# ---------------------------------------------------------------------------
# Constants & alias maps
# ---------------------------------------------------------------------------
TS_FORMAT_LOG = "%Y.%m.%d_%H.%M.%S"

# Aliases per standardised column name. First match wins.
ALIASES_FULL: dict[str, list[str]] = {
    "timestamp":        ["Timestamp"],
    "latitude":         ["Latitude", "LAT"],
    "longitude":        ["Longitude", "LON"],
    "speed_kmh":        ["Speed"],
    "operator_name":    ["Operatorname"],
    "operator":         ["Operator"],
    "cgi":              ["CGI"],
    "serving_cell_id":  ["CellID"],
    "lac":              ["LAC"],
    "network_tech":     ["NetworkTech"],
    "network_mode":     ["NetworkMode"],
    "rsrp_dbm":         ["Level", "RSRP"],
    "rsrq_db":          ["Qual", "RSRQ"],
    "sinr_db":          ["SNR", "SINR"],
    "cqi":              ["CQI"],
    "ltersssi_dbm":     ["LTERSSI"],
    "arfcn":            ["ARFCN"],
    "dl_bitrate_kbps":  ["DL_bitrate"],
    "ul_bitrate_kbps":  ["UL_bitrate"],
    "altitude_m":       ["Altitude"],
    "gps_accuracy_m":   ["Accuracy"],
    "ping_avg_ms":      ["PINGAVG"],
    "ping_min_ms":      ["PINGMIN"],
    "ping_max_ms":      ["PINGMAX"],
    "ping_stdev_ms":    ["PINGSTDEV"],
    "ping_loss_pct":    ["PINGLOSS"],
    "test_dl_kbps":     ["TESTDOWNLINK"],
    "test_ul_kbps":     ["TESTUPLINK"],
    "test_dl_max_kbps": ["TESTDOWNLINKMAX"],
    "test_ul_max_kbps": ["TESTUPLINKMAX"],
    "distance_to_serving_m": ["Distance"],
    "event_type":       ["EVENT"],
    "event_details":    ["EVENTDETAILS"],
    "band":             ["BAND"],
    "bandwidth":        ["BANDWIDTH"],
    "csi_rsrp_dbm":     ["CSI_RSRP"],
    "csi_rsrq_db":      ["CSI_RSRQ"],
    "csi_snr_db":       ["CSI_SNR"],
}

ALIASES_CELLFIND: dict[str, list[str]] = {
    "timestamp":        ["Timestamp"],
    "latitude":         ["Latitude"],
    "longitude":        ["Longitude"],
    "serving_cell_lat": ["Cell_Latitude"],
    "serving_cell_lon": ["Cell_Longitude"],
    "gps_accuracy_m":   ["Accuracy"],
    "speed_kmh":        ["Speed"],
    "operator_name":    ["Operatorname"],
    "operator":         ["Operator"],
    "cgi":              ["CGI"],
    "serving_cell_id":  ["CellID"],
    "lac":              ["LAC"],
    "network_tech":     ["NetworkTech"],
    "rsrp_dbm":         ["Level"],
    "rsrq_db":          ["Qual"],
    "sinr_db":          ["SNR"],
    "cqi":              ["CQI"],
    "ltersssi_dbm":     ["LTERSSI"],
    "dl_bitrate_kbps":  ["DL_bitrate"],
    "ul_bitrate_kbps":  ["UL_bitrate"],
}

REQUIRED = ["latitude", "longitude", "rsrp_dbm"]

# Columns that must be coerced to numeric and that get a status column.
NUMERIC_FIELDS = {
    "rsrp_dbm", "rsrq_db", "sinr_db", "cqi", "ltersssi_dbm",
    "csi_rsrp_dbm", "csi_rsrq_db", "csi_snr_db",
    "ping_avg_ms", "ping_min_ms", "ping_max_ms", "ping_stdev_ms", "ping_loss_pct",
    "dl_bitrate_kbps", "ul_bitrate_kbps",
    "test_dl_kbps", "test_ul_kbps", "test_dl_max_kbps", "test_ul_max_kbps",
    "altitude_m", "gps_accuracy_m", "speed_kmh",
    "distance_to_serving_m", "arfcn",
    "serving_cell_lat", "serving_cell_lon",
    "latitude", "longitude",
}


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------
@dataclass
class ParseResult:
    df: pd.DataFrame                     # one row per measurement
    variant: str                         # gnettrack_full | gnettrack_cellfind | unknown
    delimiter: str                       # human-readable
    columns_detected: list[str] = field(default_factory=list)
    columns_missing:  list[str] = field(default_factory=list)
    rows_raw: int = 0
    rows_dropped_essential: int = 0
    rows_dropped_gps: int = 0
    warnings: list[str] = field(default_factory=list)
    file_sha256: str = ""
    file_size_bytes: int = 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _detect_delimiter(sample: str) -> tuple[str, str]:
    counts = {"\t": sample.count("\t"),
              ",": sample.count(","),
              ";": sample.count(";")}
    sep = max(counts, key=counts.get)
    name = {"\t": "tab", ",": "comma", ";": "semicolon"}[sep]
    return sep, name


def _detect_variant(columns: set[str]) -> str:
    if "PINGAVG" in columns or "TESTDOWNLINK" in columns or "NetworkMode" in columns:
        return "gnettrack_full"
    if "Cell_Latitude" in columns and "Cell_Longitude" in columns:
        return "gnettrack_cellfind"
    return "unknown"


def _is_null_marker(v) -> bool:
    if v is None:
        return True
    if isinstance(v, float) and math.isnan(v):
        return True
    if isinstance(v, str) and v.strip() in settings.null_markers:
        return True
    return False


def _coerce_numeric_with_status(series: pd.Series) -> tuple[pd.Series, pd.Series]:
    """
    Returns (numeric_values, status) where status is per-row 'ok' or
    'missing_field'. Non-parseable strings → NaN, status='missing_field'.
    """
    s = series.astype("object")
    null_mask = s.map(_is_null_marker)
    coerced = pd.to_numeric(s.where(~null_mask, other=np.nan), errors="coerce")
    status = np.where(coerced.isna(), "missing_field", "ok")
    return coerced, pd.Series(status, index=series.index)


def _haversine_m(lat1, lon1, lat2, lon2) -> float:
    R = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp/2)**2 + math.cos(p1) * math.cos(p2) * math.sin(dl/2)**2
    return 2 * R * math.asin(math.sqrt(a))


def _gps_quality(acc_m: float | None) -> str | None:
    if acc_m is None or (isinstance(acc_m, float) and math.isnan(acc_m)):
        return None
    if acc_m <= settings.gps_acc_high_m:
        return "high"
    if acc_m <= settings.gps_acc_normal_m:
        return "normal"
    if acc_m <= settings.gps_acc_low_m:
        return "low"
    return "discard"


def _period_of_day(ts) -> str | None:
    """Categoria de horario derivada do timestamp_log.
    manha (06-11) | tarde (12-17) | noite (18-23) | madrugada (00-05)."""
    if ts is None:
        return None
    try:
        h = ts.hour if hasattr(ts, "hour") else None
        if h is None:
            return None
        if   6 <= h < 12: return "manha"
        elif 12 <= h < 18: return "tarde"
        elif 18 <= h < 24: return "noite"
        else: return "madrugada"
    except Exception:                                       # noqa: BLE001
        return None


def _light_condition_fallback(ts) -> str | None:
    """Aproximacao 'dia/noite' sem chamar Open-Meteo: usa 06h-18h.
    Pode ser refinada por sunrise/sunset durante o enrichment."""
    if ts is None:
        return None
    try:
        h = ts.hour if hasattr(ts, "hour") else None
        if h is None:
            return None
        return "dia" if 6 <= h < 18 else "noite"
    except Exception:                                       # noqa: BLE001
        return None


def _signal_rating(rsrp: float | None) -> str | None:
    """
    6-level rating per the research plan (Quadro 2):
        Excelente  : RSRP > −85 dBm
        Bom        : −85 ≥ RSRP > −95
        Satisfatório: −95 ≥ RSRP > −105
        Ruim       : −105 ≥ RSRP > −115
        Péssimo    : −115 ≥ RSRP > −125
        Nulo       : RSRP ≤ −125  (or unable to attach)
    """
    if rsrp is None or (isinstance(rsrp, float) and math.isnan(rsrp)):
        return "Nulo"
    if rsrp > -85:  return "Excelente"
    if rsrp > -95:  return "Bom"
    if rsrp > -105: return "Satisfatório"
    if rsrp > -115: return "Ruim"
    if rsrp > -125: return "Péssimo"
    return "Nulo"


# --- Frequency derivation from ARFCN + band ------------------------------
# Nominal band-centre frequencies in Hz. Used when the log carries the band
# label but the analyst wants frequency for cross-band comparison. For the
# campus campaign the operator uses LTE 1800/2600 + NR 3.5 GHz, which is the
# coverage of this table. Unknown bands → None (we never fabricate values).
_BAND_CENTRE_HZ: dict[str, float] = {
    # LTE FDD — accept both "B7" and "L7" (G-NetTrack uses the "L" prefix
    # in some firmware versions).
    "B1":  2_140e6, "B2":  1_960e6, "B3":  1_842.5e6, "B4":  2_132.5e6,
    "B5":    881.5e6, "B7":  2_655e6, "B8":  942.5e6, "B12":  731e6,
    "B13":   746e6, "B17":   734e6, "B20":  806e6,  "B28":  778.5e6,
    "L1":  2_140e6, "L2":  1_960e6, "L3":  1_842.5e6, "L4":  2_132.5e6,
    "L5":    881.5e6, "L7":  2_655e6, "L8":  942.5e6, "L12":  731e6,
    "L13":   746e6, "L17":   734e6, "L20":  806e6,  "L28":  778.5e6,
    # NR
    "n1":  2_140e6, "n3":  1_842.5e6, "n7":  2_655e6, "n28":   778.5e6,
    "n38": 2_595e6, "n40": 2_350e6, "n41": 2_593e6,
    "n66": 2_155e6, "n71": 627e6,
    "n77": 3_700e6, "n78": 3_500e6, "n79": 4_700e6,
}


def _frequency_hz(band: str | None, arfcn: int | None) -> float | None:
    """Best-effort nominal centre frequency from the band label. Accepts
    both "B7"/"L7" prefixes (LTE) and "n78" (NR). Returns None for unknown."""
    if not band:
        return None
    b = str(band).strip()
    f = _BAND_CENTRE_HZ.get(b)
    return float(f) if f is not None else None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def parse_log_bytes(content: bytes, *, filename: str = "") -> ParseResult:
    file_sha = hashlib.sha256(content).hexdigest()
    text = content.decode("utf-8", errors="replace")
    return parse_log_text(text, filename=filename, file_sha256=file_sha,
                          file_size_bytes=len(content))


def parse_log_text(text: str, *, filename: str = "",
                   file_sha256: str = "", file_size_bytes: int = 0) -> ParseResult:
    sample = text[:4000]
    sep, sep_name = _detect_delimiter(sample)
    warnings: list[str] = []

    try:
        # index_col=False: os logs cellfind do G-NetTrack tem o cabecalho com
        # 53 colunas mas as linhas de dados com 54 (uma coluna de vizinho a
        # mais / tab final). Sem isso, o pandas assume que a 1a coluna e um
        # indice e DESLOCA todas as colunas uma casa -> Latitude recebe o valor
        # de Cell_Longitude, RSRP recebe o valor errado, etc. Forcar index_col
        # =False mantem o alinhamento cabecalho<->dados.
        df = pd.read_csv(io.StringIO(text), sep=sep,
                         engine="python", on_bad_lines="skip", dtype=str,
                         index_col=False)
    except Exception as exc:                           # noqa: BLE001
        return ParseResult(
            df=pd.DataFrame(), variant="unknown", delimiter=sep_name,
            warnings=[f"Falha ao ler CSV: {exc}"],
            file_sha256=file_sha256, file_size_bytes=file_size_bytes,
        )

    df.columns = [str(c).strip().replace("\ufeff", "") for c in df.columns]
    cols_detected = list(df.columns)
    variant = _detect_variant(set(cols_detected))
    aliases = ALIASES_FULL if variant == "gnettrack_full" else ALIASES_CELLFIND

    # Build map standard_name -> actual column
    col_map: dict[str, str] = {}
    cols_missing: list[str] = []
    for std, opts in aliases.items():
        for opt in opts:
            if opt in df.columns:
                col_map[std] = opt
                break
        else:
            cols_missing.append(std)

    missing_required = [r for r in REQUIRED if r not in col_map]
    if missing_required:
        warnings.append(
            f"Colunas essenciais ausentes: {missing_required}. "
            f"Variante detectada={variant}, separador={sep_name}."
        )
        return ParseResult(
            df=pd.DataFrame(), variant=variant, delimiter=sep_name,
            columns_detected=cols_detected, columns_missing=cols_missing,
            warnings=warnings,
            file_sha256=file_sha256, file_size_bytes=file_size_bytes,
        )

    # Build standardised dataframe with renamed columns
    df_std = df[list(col_map.values())].rename(columns={v: k for k, v in col_map.items()}).copy()
    rows_raw = len(df_std)

    # Numeric coercion + status columns
    for std in list(df_std.columns):
        if std in NUMERIC_FIELDS:
            vals, status = _coerce_numeric_with_status(df_std[std])
            df_std[std] = vals
            # Only keep status for the metrics that we model
            if std in {"rsrp_dbm", "rsrq_db", "sinr_db", "cqi", "ltersssi_dbm",
                       "csi_rsrp_dbm", "csi_rsrq_db", "csi_snr_db",
                       "ping_avg_ms", "ping_stdev_ms",
                       "dl_bitrate_kbps", "ul_bitrate_kbps",
                       "test_dl_max_kbps", "test_ul_max_kbps"}:
                df_std[f"{std}_status"] = status

    # Add log_absent status for fields the variant doesn't carry
    expected_status_cols = {"rsrp_dbm", "rsrq_db", "sinr_db", "cqi", "ltersssi_dbm",
                            "csi_rsrp_dbm", "csi_rsrq_db", "csi_snr_db",
                            "ping_avg_ms", "ping_stdev_ms",
                            "dl_bitrate_kbps", "ul_bitrate_kbps",
                            "test_dl_max_kbps", "test_ul_max_kbps"}
    for std in expected_status_cols:
        col = f"{std}_status"
        if col not in df_std.columns:
            df_std[col] = "log_absent"
            df_std[std] = np.nan

    # Timestamp parsing (best-effort)
    if "timestamp" in df_std.columns:
        df_std["timestamp_log"] = pd.to_datetime(
            df_std["timestamp"], format=TS_FORMAT_LOG, errors="coerce"
        )
    else:
        df_std["timestamp_log"] = pd.NaT

    # Drop rows missing essentials
    pre_drop = len(df_std)
    df_std = df_std.dropna(subset=REQUIRED)
    rows_dropped_essential = pre_drop - len(df_std)

    # GPS quality and accuracy filter
    if "gps_accuracy_m" in df_std.columns:
        df_std["gps_quality"] = df_std["gps_accuracy_m"].apply(_gps_quality)
    else:
        df_std["gps_quality"] = None

    pre_gps = len(df_std)
    df_std = df_std[df_std["gps_quality"] != "discard"].reset_index(drop=True)
    rows_dropped_gps = pre_gps - len(df_std)

    # Distance to serving cell (compute from Cell_Lat/Lon if Distance not in log)
    # G-NetTrack uses -1 as a "not computed" sentinel; treat as missing.
    if "distance_to_serving_m" in df_std.columns:
        df_std.loc[df_std["distance_to_serving_m"] == -1, "distance_to_serving_m"] = np.nan
    if "distance_to_serving_m" not in df_std.columns or df_std["distance_to_serving_m"].isna().all():
        if {"serving_cell_lat", "serving_cell_lon"}.issubset(df_std.columns):
            d, src = [], []
            for _, r in df_std.iterrows():
                cl, co = r.get("serving_cell_lat"), r.get("serving_cell_lon")
                la, lo = r["latitude"], r["longitude"]
                if pd.notna(cl) and pd.notna(co):
                    d.append(_haversine_m(la, lo, cl, co))
                    src.append("computed")
                else:
                    d.append(np.nan); src.append("missing")
            df_std["distance_to_serving_m"] = d
            df_std["distance_source"] = src
        else:
            df_std["distance_to_serving_m"] = np.nan
            df_std["distance_source"] = "missing"
    else:
        # Mixed: try to fill missing entries from cell coords if available
        if {"serving_cell_lat", "serving_cell_lon"}.issubset(df_std.columns):
            mask = df_std["distance_to_serving_m"].isna()
            for idx in df_std.index[mask]:
                cl = df_std.at[idx, "serving_cell_lat"]
                co = df_std.at[idx, "serving_cell_lon"]
                if pd.notna(cl) and pd.notna(co):
                    df_std.at[idx, "distance_to_serving_m"] = _haversine_m(
                        df_std.at[idx, "latitude"], df_std.at[idx, "longitude"], cl, co
                    )
        df_std["distance_source"] = np.where(
            df_std["distance_to_serving_m"].notna(),
            np.where(df_std.get("serving_cell_lat", pd.Series([np.nan]*len(df_std))).notna()
                     & df_std["distance_to_serving_m"].notna(),
                     "log_or_computed", "log"),
            "missing",
        )

    # Altitude source flag
    if "altitude_m" in df_std.columns:
        df_std["altitude_source"] = np.where(
            df_std["altitude_m"].notna(), "log", "missing"
        )
    else:
        df_std["altitude_m"] = np.nan
        df_std["altitude_source"] = "missing"

    # NetworkTech normalisation: keep '5G' / '4G' as-is; strip whitespace
    if "network_tech" in df_std.columns:
        df_std["network_tech"] = df_std["network_tech"].astype(str).str.strip()

    # Derived signal_rating (documented threshold)
    df_std["signal_rating"] = df_std["rsrp_dbm"].apply(_signal_rating)

    # Derived period_of_day e light_condition (aprox. dia/noite por hora)
    if "timestamp_log" in df_std.columns:
        df_std["period_of_day"]   = df_std["timestamp_log"].apply(_period_of_day)
        df_std["light_condition"] = df_std["timestamp_log"].apply(_light_condition_fallback)
    else:
        df_std["period_of_day"]   = None
        df_std["light_condition"] = None

    # Derived frequency from band + ARFCN (Quadro 2). NaN for unknown bands.
    if "band" in df_std.columns:
        df_std["frequency_hz"] = df_std.apply(
            lambda r: _frequency_hz(r.get("band"), r.get("arfcn")), axis=1,
        )
    else:
        df_std["frequency_hz"] = np.nan

    return ParseResult(
        df=df_std,
        variant=variant,
        delimiter=sep_name,
        columns_detected=cols_detected,
        columns_missing=cols_missing,
        rows_raw=rows_raw,
        rows_dropped_essential=rows_dropped_essential,
        rows_dropped_gps=rows_dropped_gps,
        warnings=warnings,
        file_sha256=file_sha256,
        file_size_bytes=file_size_bytes,
    )


def parse_neighbor_columns(df_raw: pd.DataFrame, max_n: int = 18) -> pd.DataFrame:
    """
    Extracts the long-format neighbor table from an already-loaded raw frame.
    Columns expected: NCellName{i}, NCellid{i}, NLAC{i}, NCell{i}, NARFCN{i},
                       NRxLev{i}, NQual{i}, NDistance{i}, NBearing{i}, NTech{i}
    Missing per-i columns are tolerated.
    Returns a DataFrame with columns:
        row_index, rank, network_tech, cell_name, cell_id, lac, pci, arfcn,
        rxlev_dbm, qual, distance_m, bearing_deg
    Empty rows (no rxlev) are dropped.
    """
    rows = []
    for i in range(1, max_n + 1):
        get = lambda c: df_raw[c] if c in df_raw.columns else pd.Series([np.nan] * len(df_raw))
        chunk = pd.DataFrame({
            "row_index": df_raw.index,
            "rank":      i,
            "network_tech": get(f"NTech{i}"),
            "cell_name":    get(f"NCellName{i}"),
            "cell_id":      get(f"NCellid{i}"),
            "lac":          get(f"NLAC{i}"),
            "pci":          get(f"NCell{i}"),
            "arfcn":        get(f"NARFCN{i}"),
            "rxlev_dbm":    get(f"NRxLev{i}"),
            "qual":         get(f"NQual{i}"),
            "distance_m":   get(f"NDistance{i}"),
            "bearing_deg":  get(f"NBearing{i}"),
        })
        rows.append(chunk)
    out = pd.concat(rows, ignore_index=True)
    # Coerce numerics with NULL marker awareness
    for c in ["lac", "pci", "arfcn", "rxlev_dbm", "qual", "distance_m", "bearing_deg"]:
        out[c] = pd.to_numeric(
            out[c].astype("object").where(~out[c].map(_is_null_marker), other=np.nan),
            errors="coerce",
        )
    out = out.dropna(subset=["rxlev_dbm"]).reset_index(drop=True)
    return out
