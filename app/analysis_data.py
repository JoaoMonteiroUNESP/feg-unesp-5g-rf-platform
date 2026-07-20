"""Shared preparation rules for scientific analyses.

The raw SQLite tables remain untouched.  These helpers consolidate exact
analytical pseudoreplications in an in-memory DataFrame and report the rule
used, so every model can disclose how many rows were excluded.  Complementary
QoS fields are coalesced before a repeated state is collapsed to one row.
"""

from __future__ import annotations

import pandas as pd


DUPLICATE_KEY_CANDIDATES = [
    "timestamp_log",
    "latitude",
    "longitude",
    "network_tech",
    "band",
    "rsrp_dbm",
    "rsrq_db",
    "sinr_db",
]


QOS_STATUS_BY_VALUE = {
    "ping_avg_ms": "ping_avg_status",
    "ping_stdev_ms": "ping_jitter_status",
    "dl_bitrate_kbps": "dl_bitrate_status",
    "ul_bitrate_kbps": "ul_bitrate_status",
    "test_dl_max_kbps": "test_dl_max_status",
    "test_ul_max_kbps": "test_ul_max_status",
}

QOS_VALUE_CANDIDATES = [
    "ping_avg_ms",
    "ping_min_ms",
    "ping_max_ms",
    "ping_stdev_ms",
    "ping_loss_pct",
    "dl_bitrate_kbps",
    "ul_bitrate_kbps",
    "test_dl_max_kbps",
    "test_ul_max_kbps",
]


def _valid_qos_value(df: pd.DataFrame, column: str) -> pd.Series:
    """Return valid mobile-QoS rows for one numeric field.

    Throughput, bitrate and latency/jitter must be strictly positive.  Zero is
    only meaningful for packet loss.  Wi-Fi rows and fields explicitly marked
    ``wifi_invalid`` are never used as coalesce sources.
    """
    values = pd.to_numeric(df[column], errors="coerce")
    valid = values.ge(0) if column == "ping_loss_pct" else values.gt(0)
    if "data_connection_type" in df.columns:
        valid &= (
            df["data_connection_type"]
            .fillna("")
            .astype(str)
            .str.upper()
            .ne("WIFI")
        )
    status_column = QOS_STATUS_BY_VALUE.get(column)
    if status_column in df.columns:
        valid &= (
            df[status_column]
            .fillna("")
            .astype(str)
            .str.lower()
            .ne("wifi_invalid")
        )
    return valid


def deduplicate_measurements(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, int, list[str]]:
    """Coalesce QoS and collapse repeated states without changing SQLite.

    `run_id` and `campaign_id` are intentionally not part of the key.  The same
    physical log was occasionally ingested more than once under different run
    or campaign identifiers; keeping those copies would leak identical rows
    across cross-validation folds.

    G-NetTrack may emit several rows with the same timestamp/GPS/radio state
    while only a later row contains the completed active QoS test.  Therefore,
    the original first-occurrence survivor is enriched by coalescing every
    valid QoS field from its corresponding repeated-state group.  This avoids
    losing a real test merely because the first row had a null result, without
    changing the non-QoS row selected for downstream models.
    Sparse CellFind rows are still preserved when their radio fields differ or
    are missing.
    """
    if df.empty:
        return df.copy(), 0, []
    key = [column for column in DUPLICATE_KEY_CANDIDATES if column in df.columns]
    if len(key) < 4:
        return df.copy(), 0, key

    work = df.copy()
    work["_dedup_order"] = range(len(work))
    work["_dedup_group"] = work.groupby(
        key, dropna=False, sort=False
    ).ngroup()

    qos_columns = [column for column in QOS_VALUE_CANDIDATES if column in work]
    valid_by_column = {
        column: _valid_qos_value(work, column) for column in qos_columns
    }
    if qos_columns:
        work["_dedup_qos_score"] = sum(
            mask.astype("int8") for mask in valid_by_column.values()
        )
    else:
        work["_dedup_qos_score"] = 0

    ranked = work.sort_values(
        ["_dedup_group", "_dedup_qos_score", "_dedup_order"],
        ascending=[True, False, True],
        kind="stable",
    )
    # Keep the same non-QoS survivor as the original rule (first occurrence),
    # then enrich only its complementary QoS fields.  This prevents the fix
    # from silently changing campaign/environment metadata used by RF models.
    survivors = work.drop_duplicates("_dedup_group", keep="first").copy()
    survivors = survivors.set_index("_dedup_group", drop=False)

    # Coalesce each QoS value from a valid row.  Its status travels with it so
    # a copied value can never retain the status of a different source row.
    for column in qos_columns:
        candidates = ranked.loc[valid_by_column[column]].drop_duplicates(
            "_dedup_group", keep="first"
        )
        if candidates.empty:
            continue
        candidates = candidates.set_index("_dedup_group", drop=False)
        survivors.loc[candidates.index, column] = candidates[column]
        status_column = QOS_STATUS_BY_VALUE.get(column)
        if status_column in work.columns:
            survivors.loc[candidates.index, status_column] = candidates[
                status_column
            ]

    # If a valid mobile QoS row enriched a first-occurrence Wi-Fi/null row,
    # carry the connection type from the QoS source to avoid a contradiction.
    if qos_columns and "data_connection_type" in work.columns:
        any_valid_qos = pd.concat(valid_by_column.values(), axis=1).any(axis=1)
        connection_sources = ranked.loc[
            any_valid_qos.reindex(ranked.index, fill_value=False)
        ].drop_duplicates("_dedup_group", keep="first")
        if not connection_sources.empty:
            connection_sources = connection_sources.set_index(
                "_dedup_group", drop=False
            )
            survivors.loc[
                connection_sources.index, "data_connection_type"
            ] = connection_sources["data_connection_type"]

    # Preserve route order by the first appearance of each repeated state,
    # independently of which row supplied its QoS values.
    first_order = work.groupby("_dedup_group", sort=False)["_dedup_order"].min()
    survivors["_dedup_group_order"] = first_order
    survivors = survivors.sort_values("_dedup_group_order", kind="stable")

    helper_columns = [
        "_dedup_order",
        "_dedup_group",
        "_dedup_qos_score",
        "_dedup_group_order",
    ]
    clean = survivors.drop(columns=helper_columns, errors="ignore").reset_index(
        drop=True
    )
    n_dropped = int(len(df) - len(clean))
    return clean, n_dropped, key
