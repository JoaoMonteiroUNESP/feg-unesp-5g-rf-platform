"""Auditable PCA, k-means and DBSCAN for empirical 5G propagation profiles.

The algorithms describe similarity inside the measured routes.  They are not
coverage extrapolators and they do not identify physical multipath parameters.
All numeric features are standardized; sparse features are disclosed and
dropped; remaining incomplete rows are handled by complete-case analysis.
"""

from __future__ import annotations

import hashlib
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from app.analysis_data import deduplicate_measurements
from app.config import settings


DEFAULT_UNSUPERVISED_FEATURES = [
    # Radio metrics reported by the primary carrier in the 5G NSA session.
    "rsrp_dbm",
    "rsrq_db",
    "sinr_db",
    "frequency_hz",
    # Relative geometry and terrain.
    "distance_to_serving_m",
    "altitude_m",
    # Historically valid weather only: field notes > Open-Meteo Archive.
    "temperature_c_eff",
    "humidity_eff",
    "cloud_cover_pct_eff",
    # Built environment and vegetation.
    "building_count_eff",
    "avg_building_height_eff_m",
    "distance_to_building_m_eff",
    "tree_count_eff",
    "avg_tree_height_eff_m",
    "distance_to_tree_m_eff",
    "tree_density_ndvi",
    "visual_obstruction_grade",
    "vegetation_density_manual",
]

SPARSE_FEATURE_THRESHOLD = 0.50
MAX_POINTS_RETURNED = 1500


def _prepare_numeric_matrix(
    df: pd.DataFrame,
    features: Iterable[str],
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    clean, duplicate_rows, duplicate_key = deduplicate_measurements(df)
    requested = list(dict.fromkeys(features))
    missing_columns = [column for column in requested if column not in clean.columns]
    available = [column for column in requested if column in clean.columns]

    numeric = clean[available].apply(pd.to_numeric, errors="coerce")
    sparse_columns = [
        column
        for column in numeric.columns
        if float(numeric[column].isna().mean()) >= SPARSE_FEATURE_THRESHOLD
    ]
    numeric = numeric.drop(columns=sparse_columns)
    constant_columns = [
        column for column in numeric.columns if numeric[column].nunique(dropna=True) <= 1
    ]
    numeric = numeric.drop(columns=constant_columns)

    complete_mask = numeric.notna().all(axis=1)
    complete_numeric = numeric.loc[complete_mask].copy()
    metadata_columns = [
        column
        for column in [
            "id",
            "latitude",
            "longitude",
            "timestamp_log",
            "campaign_id",
            "sector_code_effective",
            "indoor_outdoor",
            "network_tech",
        ]
        if column in clean.columns
    ]
    metadata = clean.loc[complete_numeric.index, metadata_columns].copy()

    audit = {
        "rows_input": int(len(df)),
        "rows_dropped_exact_duplicates": duplicate_rows,
        "duplicate_key": duplicate_key,
        "rows_after_deduplication": int(len(clean)),
        "rows_complete_case": int(len(complete_numeric)),
        "rows_dropped_incomplete": int(len(clean) - len(complete_numeric)),
        "features_requested": requested,
        "features_used": list(complete_numeric.columns),
        "features_missing": missing_columns,
        "features_dropped_sparse": sparse_columns,
        "features_dropped_constant": constant_columns,
        "features_excluded_quality": [
            "temperature_c",
            "humidity",
            "cloud_cover_pct",
        ],
        "quality_exclusion_reason": (
            "legacy current-weather fields are excluded; only manual/archive "
            "effective weather is eligible for analytical use"
        ),
        "missingness_rule": (
            "features with >=50% missing values are excluded; remaining rows use "
            "complete-case analysis; no value imputation"
        ),
    }
    return complete_numeric, metadata, audit


def _matrix_hash(matrix: pd.DataFrame) -> str:
    digest = hashlib.sha256()
    digest.update(np.asarray(matrix.values, dtype=float, order="C").tobytes())
    digest.update("|".join(matrix.columns).encode("utf-8"))
    return digest.hexdigest()


def _safe_silhouette(matrix: np.ndarray, labels: np.ndarray) -> float | None:
    unique = np.unique(labels)
    if len(unique) < 2 or len(unique) >= len(labels):
        return None
    sample_size = min(2000, len(labels))
    try:
        return float(
            silhouette_score(
                matrix,
                labels,
                sample_size=sample_size if sample_size < len(labels) else None,
                random_state=settings.random_state,
            )
        )
    except ValueError:
        return None


def _cluster_profiles(
    original: pd.DataFrame,
    labels: np.ndarray,
    *,
    noise_label: int | None = None,
) -> list[dict]:
    work = original.copy()
    work["cluster"] = labels
    profiles: list[dict] = []
    for label, group in work.groupby("cluster", sort=True):
        if noise_label is not None and int(label) == noise_label:
            label_name = "noise"
        else:
            label_name = str(int(label))
        profiles.append(
            {
                "cluster": label_name,
                "n": int(len(group)),
                "share_pct": round(100 * len(group) / len(work), 2),
                "means": {
                    column: float(group[column].mean()) for column in original.columns
                },
                "medians": {
                    column: float(group[column].median()) for column in original.columns
                },
            }
        )
    return profiles


def _point_payload(
    metadata: pd.DataFrame,
    pca_scores: np.ndarray,
    kmeans_labels: np.ndarray,
    dbscan_labels: np.ndarray,
) -> list[dict]:
    if len(metadata) > MAX_POINTS_RETURNED:
        positions = np.linspace(0, len(metadata) - 1, MAX_POINTS_RETURNED).astype(int)
    else:
        positions = np.arange(len(metadata))
    points: list[dict] = []
    for position in positions:
        row = metadata.iloc[position]
        point = {}
        for column, value in row.to_dict().items():
            if pd.isna(value):
                point[column] = None
            elif isinstance(value, (pd.Timestamp,)):
                point[column] = value.isoformat()
            elif isinstance(value, np.generic):
                point[column] = value.item()
            else:
                point[column] = value
        point.update(
            {
                "pc1": float(pca_scores[position, 0]),
                "pc2": float(pca_scores[position, 1]),
                "kmeans_cluster": int(kmeans_labels[position]),
                "dbscan_cluster": int(dbscan_labels[position]),
            }
        )
        points.append(point)
    return points


def unsupervised_analysis(
    df: pd.DataFrame,
    features: Iterable[str] = DEFAULT_UNSUPERVISED_FEATURES,
    *,
    k_min: int = 2,
    k_max: int = 6,
) -> dict:
    matrix, metadata, audit = _prepare_numeric_matrix(df, features)
    if len(matrix) < 30:
        return {
            "error": (
                "Dados insuficientes para PCA/clustering depois da análise de "
                f"casos completos (n={len(matrix)} < 30)."
            ),
            "audit": audit,
        }
    if matrix.shape[1] < 2:
        return {
            "error": "PCA/clustering requer ao menos duas variáveis numéricas válidas.",
            "audit": audit,
        }

    scaler = StandardScaler()
    standardized = scaler.fit_transform(matrix)

    pca_full = PCA(random_state=settings.random_state).fit(standardized)
    cumulative = np.cumsum(pca_full.explained_variance_ratio_)
    selected_components = int(np.searchsorted(cumulative, 0.90) + 1)
    selected_components = max(2, min(selected_components, matrix.shape[1]))
    pca = PCA(n_components=selected_components, random_state=settings.random_state)
    scores = pca.fit_transform(standardized)
    loadings = []
    for component_index in range(selected_components):
        component_values = pca.components_[component_index]
        ordered = np.argsort(np.abs(component_values))[::-1]
        loadings.append(
            {
                "component": f"PC{component_index + 1}",
                "explained_variance_ratio": float(
                    pca.explained_variance_ratio_[component_index]
                ),
                "top_loadings": [
                    {
                        "feature": matrix.columns[position],
                        "loading": float(component_values[position]),
                    }
                    for position in ordered[: min(8, len(ordered))]
                ],
            }
        )

    k_candidates: list[dict] = []
    k_models: dict[int, tuple[KMeans, np.ndarray, float | None]] = {}
    upper_k = min(k_max, len(matrix) - 1)
    for k in range(max(2, k_min), upper_k + 1):
        model = KMeans(
            n_clusters=k,
            n_init=30,
            random_state=settings.random_state,
        )
        labels = model.fit_predict(standardized)
        silhouette = _safe_silhouette(standardized, labels)
        k_models[k] = (model, labels, silhouette)
        k_candidates.append(
            {
                "k": k,
                "inertia": float(model.inertia_),
                "silhouette": silhouette,
            }
        )
    valid_k = [item for item in k_candidates if item["silhouette"] is not None]
    selected_k = max(valid_k, key=lambda item: item["silhouette"])["k"]
    k_model, k_labels, k_silhouette = k_models[selected_k]
    kmeans_warning = None
    if selected_k == upper_k:
        kmeans_warning = (
            "best silhouette occurs at the upper tested boundary; k is an "
            "exploratory partition, not evidence of a uniquely identified "
            "number of physical propagation regimes"
        )

    min_samples = max(5, 2 * matrix.shape[1])
    min_samples = min(min_samples, max(5, len(matrix) - 1))
    neighbors = NearestNeighbors(n_neighbors=min_samples).fit(standardized)
    distances, _ = neighbors.kneighbors(standardized)
    kth_distances = np.sort(distances[:, -1])
    dbscan_candidates: list[dict] = []
    fitted_dbscan: list[tuple[dict, np.ndarray]] = []
    for quantile in (0.80, 0.85, 0.90, 0.93, 0.95, 0.97):
        eps = float(np.quantile(kth_distances, quantile))
        labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(standardized)
        clustered_mask = labels != -1
        clusters = sorted(set(labels[clustered_mask]))
        silhouette = None
        if len(clusters) >= 2 and clustered_mask.sum() > len(clusters):
            silhouette = _safe_silhouette(
                standardized[clustered_mask], labels[clustered_mask]
            )
        candidate = {
            "eps_quantile": quantile,
            "eps": eps,
            "min_samples": min_samples,
            "n_clusters_excluding_noise": len(clusters),
            "noise_fraction": float(np.mean(labels == -1)),
            "silhouette_excluding_noise": silhouette,
        }
        dbscan_candidates.append(candidate)
        fitted_dbscan.append((candidate, labels))

    viable = [
        item
        for item in fitted_dbscan
        if item[0]["silhouette_excluding_noise"] is not None
        and 2 <= item[0]["n_clusters_excluding_noise"] <= 8
        and item[0]["noise_fraction"] <= 0.60
    ]
    if viable:
        selected_dbscan, dbscan_labels = max(
            viable,
            key=lambda item: item[0]["silhouette_excluding_noise"],
        )
        dbscan_warning = None
    else:
        selected_dbscan, dbscan_labels = min(
            fitted_dbscan,
            key=lambda item: abs(item[0]["eps_quantile"] - 0.90),
        )
        dbscan_warning = (
            "Nenhum ajuste testado produziu de 2 a 8 clusters com <=60% de "
            "ruído; o resultado DBSCAN deve ser tratado como diagnóstico de "
            "separação fraca, não como segmentação estável."
        )

    return {
        "scope_label": (
            "descrição exploratória dentro das rotas medidas; não generaliza para "
            "áreas sem observação"
        ),
        "radio_metric_label": (
            "métricas da portadora primária reportada na sessão 5G NSA; a "
            "associação definitiva LTE-NR permanece a verificar"
        ),
        "dataset_sha256": _matrix_hash(matrix),
        "audit": audit,
        "standardization": {
            "method": "StandardScaler",
            "means": {
                column: float(value)
                for column, value in zip(matrix.columns, scaler.mean_)
            },
            "scales": {
                column: float(value)
                for column, value in zip(matrix.columns, scaler.scale_)
            },
        },
        "pca": {
            "n_components_for_90_pct": selected_components,
            "explained_variance_ratio": [
                float(value) for value in pca.explained_variance_ratio_
            ],
            "cumulative_explained_variance": [
                float(value) for value in np.cumsum(pca.explained_variance_ratio_)
            ],
            "loadings": loadings,
        },
        "kmeans": {
            "selection_rule": (
                f"highest silhouette among k={max(2, k_min)}..{upper_k}"
            ),
            "candidates": k_candidates,
            "selected_k": int(selected_k),
            "silhouette": k_silhouette,
            "warning": kmeans_warning,
            "cluster_centers_original_units": [
                {
                    column: float(value)
                    for column, value in zip(
                        matrix.columns,
                        scaler.inverse_transform(k_model.cluster_centers_)[cluster],
                    )
                }
                for cluster in range(selected_k)
            ],
            "profiles": _cluster_profiles(matrix, k_labels),
        },
        "dbscan": {
            "selection_rule": (
                "k-distance quantiles 0.80..0.97; maximize silhouette outside "
                "noise subject to 2..8 clusters and <=60% noise"
            ),
            "candidates": dbscan_candidates,
            "selected": selected_dbscan,
            "warning": dbscan_warning,
            "profiles": _cluster_profiles(matrix, dbscan_labels, noise_label=-1),
        },
        "points_sample": _point_payload(
            metadata, scores, k_labels, dbscan_labels
        ),
        "points_sample_is_downsampled": bool(len(metadata) > MAX_POINTS_RETURNED),
    }
