"""
Machine-learning pipelines.

Two pipelines, both auditable
-----------------------------
1. REGRESSION — predict `rsrp_dbm` from environment / mobility / spatial
   features. Models:  RandomForest, XGBoost, SVR (with StandardScaler).
   Validation:  RepeatedKFold (n_splits=cv_splits, n_repeats=cv_repeats).
   Metrics:     R², MAE, RMSE — reported as mean ± std and 95 % CI.
   Stratification: per `network_tech` AND a pooled run.

2. CLASSIFICATION — predict the six-level `signal_rating` from the
   SAME feature set (no rsrp/rsrq/sinr leaked into X) so the inverse
   problem is well-posed. Models: RandomForest, XGBoost (multiclass), SVC
   (RBF + scaler). Validation: RepeatedStratifiedKFold. Metrics:
   accuracy, balanced accuracy, macro-F1, per-class precision/recall/F1,
   AND the **mean confusion matrix** across folds.

Auditability
------------
Every fit registers
  * dataset hash (sha256 of feature/target arrays),
  * row count, dropped count, feature list, target,
  * model hyperparameters,
  * per-fold metrics + summary statistics (mean, std, ci95),
  * per-model feature importances (RF/XGB) or permutation importance (SVR/SVC).
"""
from __future__ import annotations
import hashlib
import json
import logging
from dataclasses import dataclass, field, asdict
from typing import Iterable

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.dummy import DummyRegressor, DummyClassifier
from sklearn.svm import SVR, SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import (
    GroupKFold, RepeatedKFold, RepeatedStratifiedKFold, StratifiedGroupKFold,
)
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, balanced_accuracy_score, f1_score,
    precision_recall_fscore_support, confusion_matrix,
)
from sklearn.inspection import permutation_importance
from xgboost import XGBRegressor, XGBClassifier

from app.analysis_data import deduplicate_measurements
from app.config import settings


log = logging.getLogger(__name__)


# Default candidate feature set. Anything not in the dataframe is dropped at
# runtime and reported in the audit.
#
# A partir de 2026-06-08, usamos as colunas '_effective' produzidas por
# `app.derivations.add_effective_columns` para:
#   * altura de prédios/árvores: anotação manual TEM PRIORIDADE sobre a API
#     (porque o OSM raramente tem a tag 'height' no centro de Guaratinguetá);
#   * setor: anotação manual TEM PRIORIDADE sobre o classificador automático
#     (porque os polígonos da planilha original têm desalinhamento conhecido).
DEFAULT_FEATURES_REGRESSION = [
    # Contexto físico do ponto
    "altitude_m", "gps_accuracy_m", "distance_to_serving_m",
    "speed_kmh", "arfcn", "n_neighbors",
    # Clima historicamente válido: manual em campo > Open-Meteo Archive.
    # As colunas legadas sem sufixo `_eff` nunca entram no modelo.
    "temperature_c_eff", "humidity_eff", "cloud_cover_pct_eff",
    # Construção (API + manual via COALESCE)
    "building_count_eff", "avg_building_height_eff_m",
    "distance_to_building_m_eff",
    # Vegetação (API + manual via COALESCE)
    "tree_count_eff", "avg_tree_height_eff_m", "distance_to_tree_m_eff",
    "tree_density_ndvi",
    # Categóricas (one-hot)
    "network_tech", "band",
    "indoor_outdoor", "surface_type",
    "environment_class_effective", "sector_code_effective",
]
DEFAULT_FEATURES_CLASSIFICATION = list(DEFAULT_FEATURES_REGRESSION)
# Target leakage guard. Never include any RF metric in X for either task.
# `signal_rating` is derived from rsrp_dbm and is therefore leaky against any
# RF target. When the TARGET is `signal_rating` itself, classification_train
# also forbids X from containing the leaky set (see _prepare_features).
LEAKY = {"rsrp_dbm", "rsrq_db", "sinr_db",
         "csi_rsrp_dbm", "csi_rsrq_db", "csi_snr_db",
         "ltersssi_dbm", "signal_rating"}

# Small-sample regime: when len(X) is below this we shrink tree depth and
# n_estimators automatically and add a warning to the audit. The rationale is
# to make tree models report something defensible (instead of pretending an
# unrestricted forest can learn from 30 examples).
SMALL_N_THRESHOLD = 80


# ---------------------------------------------------------------------------
@dataclass
class FoldMetrics:
    fold: int
    metric: str
    value: float


@dataclass
class ModelReport:
    name: str
    task: str                            # 'regression' | 'classification'
    n_train_total: int
    n_features: int
    feature_names: list[str]
    target: str
    hyperparameters: dict
    per_fold: list[FoldMetrics] = field(default_factory=list)
    summary: dict = field(default_factory=dict)        # metric -> {mean,std,ci95}
    confusion_matrix_mean: list[list[float]] | None = None
    confusion_matrix_labels: list[str] | None = None
    classification_report: dict | None = None          # per-class precision/recall/F1
    feature_importance: dict[str, float] | None = None  # native (RF/XGB) when available
    permutation_importance: dict[str, float] | None = None  # always reported when feasible
    is_baseline: bool = False
    # Regression-only diagnostics (None for classifiers / baselines):
    residual_diagnostics: dict | None = None


@dataclass
class TrainingAudit:
    task: str
    target: str
    features_requested: list[str]
    features_used: list[str]
    features_dropped: list[str]
    rows_input: int
    rows_dropped_exact_duplicates: int
    duplicate_key: list[str]
    rows_used: int
    rows_dropped_nan: int
    dataset_sha256: str
    cv_scheme: str
    group_by: str | None
    n_groups: int | None
    random_state: int
    models: list[ModelReport] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Feature preparation (one-hot for categoricals; drop NaN rows)
# ---------------------------------------------------------------------------
NAN_DROP_THRESHOLD = 0.5  # remover feature numerica com >50% NaN antes do dropna


def _prepare_features(df: pd.DataFrame, requested: Iterable[str], target: str
                      ) -> tuple[pd.DataFrame, pd.Series, list[str], list[str], int]:
    requested = [c for c in requested if c not in LEAKY and c != target]
    available = [c for c in requested if c in df.columns]
    dropped_missing_col = [c for c in requested if c not in df.columns]

    # Decide categorical vs numeric by dtype
    cats = [c for c in available if df[c].dtype == object]
    nums = [c for c in available if c not in cats]

    # ANTI dataset-vazio: features numericas com >50% NaN sao removidas ANTES
    # do dropna. Caso contrario UMA unica coluna 100% NaN (ex: distance_to_
    # serving_m sem cellfind) zera todo o treino. Auditamos quem foi removido.
    high_nan_dropped: list[str] = []
    if nums:
        n = max(1, len(df))
        nums_kept = []
        for c in nums:
            nan_rate = float(df[c].isna().mean()) if n > 0 else 1.0
            if nan_rate >= NAN_DROP_THRESHOLD:
                high_nan_dropped.append(f"{c} ({nan_rate*100:.0f}% NaN)")
            else:
                nums_kept.append(c)
        nums = nums_kept
        available = nums + cats
        dropped_missing_col = dropped_missing_col + high_nan_dropped

    work = df[available + [target]].copy()
    rows_input = len(work)

    # Drop rows where target is NaN
    work = work.dropna(subset=[target])

    # For numerics, drop rows with NaN (no imputation by policy)
    if nums:
        work = work.dropna(subset=nums)

    # One-hot encode categoricals; rows where all categoricals NaN are kept as
    # 'unknown' rather than dropped, but we mark as a separate level.
    if cats:
        for c in cats:
            work[c] = work[c].fillna("unknown").astype(str)
        work = pd.get_dummies(work, columns=cats, drop_first=False)

    rows_used = len(work)
    rows_dropped = rows_input - rows_used

    y = work[target]
    X = work.drop(columns=[target])
    feature_names = list(X.columns)
    return X, y, feature_names, dropped_missing_col, rows_dropped


def _dataset_hash(X: pd.DataFrame, y: pd.Series) -> str:
    h = hashlib.sha256()
    h.update(np.asarray(X.values, dtype=float, order="C").tobytes())
    if y.dtype.kind in ("i", "f", "u", "b"):
        h.update(np.asarray(y.values, order="C").tobytes())
    else:
        h.update(("|".join(map(str, y.values))).encode("utf-8"))
    h.update("|".join(X.columns).encode("utf-8"))
    return h.hexdigest()


def _normalise_group_by(group_by: str | None) -> str | None:
    if group_by is None:
        return None
    value = str(group_by).strip().lower()
    if value in {"", "none", "random", "row"}:
        return None
    aliases = {
        "campaign": "campaign_id",
        "sector": "sector_code_effective",
        "run": "run_id",
        "day": "date",
    }
    return aliases.get(value, value)


def _group_series(
    df: pd.DataFrame,
    index: pd.Index,
    group_by: str | None,
) -> tuple[pd.Series | None, str | None, int | None, str | None]:
    """Return groups aligned to the prepared feature matrix.

    Missing/blank group labels are retained as an explicit ``<unclassified>``
    level and disclosed in the warning.  Campaigns and dates in the current
    database are complete; the explicit level mainly protects sector analyses.
    """
    normalized = _normalise_group_by(group_by)
    if normalized is None:
        return None, None, None, None

    source_column = normalized
    if normalized == "date":
        if "timestamp_log" not in df.columns:
            raise ValueError("Agrupamento por data requer a coluna timestamp_log.")
        values = pd.to_datetime(
            df.loc[index, "timestamp_log"], errors="coerce"
        ).dt.strftime("%Y-%m-%d")
    else:
        if source_column not in df.columns and source_column == "sector_code_effective":
            source_column = "sector_code"
        if source_column not in df.columns:
            raise ValueError(
                f"Agrupamento '{normalized}' requer a coluna {source_column}."
            )
        values = df.loc[index, source_column]

    values = values.astype("string").str.strip()
    missing_mask = values.isna() | values.eq("")
    values = values.mask(missing_mask, "<unclassified>").astype(str)
    n_groups = int(values.nunique())
    if n_groups < 2:
        raise ValueError(
            f"Validação agrupada por {normalized} requer ao menos 2 grupos; "
            f"foram encontrados {n_groups}."
        )
    warning = None
    if bool(missing_mask.any()):
        warning = (
            f"{int(missing_mask.sum())} linhas sem grupo em {normalized} foram "
            "mantidas no nível explícito <unclassified>."
        )
    return values, normalized, n_groups, warning


def _regression_cv(
    groups: pd.Series | None,
    n_groups: int | None,
):
    if groups is None:
        cv = RepeatedKFold(
            n_splits=settings.cv_splits,
            n_repeats=settings.cv_repeats,
            random_state=settings.random_state,
        )
        scheme = (
            f"RepeatedKFold(splits={settings.cv_splits}, "
            f"repeats={settings.cv_repeats})"
        )
        return cv, scheme
    n_splits = min(settings.cv_splits, int(n_groups or 0))
    if n_splits < 2:
        raise ValueError("Validação agrupada requer ao menos 2 folds.")
    return GroupKFold(n_splits=n_splits), f"GroupKFold(splits={n_splits})"


def _classification_cv(
    groups: pd.Series | None,
    n_groups: int | None,
):
    if groups is None:
        cv = RepeatedStratifiedKFold(
            n_splits=settings.cv_splits,
            n_repeats=settings.cv_repeats,
            random_state=settings.random_state,
        )
        scheme = (
            f"RepeatedStratifiedKFold(splits={settings.cv_splits}, "
            f"repeats={settings.cv_repeats})"
        )
        return cv, scheme
    n_splits = min(settings.cv_splits, int(n_groups or 0))
    if n_splits < 2:
        raise ValueError("Validação agrupada requer ao menos 2 folds.")
    return (
        StratifiedGroupKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=settings.random_state,
        ),
        f"StratifiedGroupKFold(splits={n_splits})",
    )


def _shrink_for_small_n(name: str, est, n: int):
    """Defensive hyperparameter shrinkage for n < SMALL_N_THRESHOLD.
    Returns the (possibly modified) estimator + a warning string or None."""
    if n >= SMALL_N_THRESHOLD:
        return est, None
    if name == "RandomForest":
        try:
            est.set_params(n_estimators=200, max_depth=6, min_samples_leaf=3)
            return est, (f"{name}: n={n}<{SMALL_N_THRESHOLD} → "
                         f"max_depth=6, min_samples_leaf=3, n_estimators=200.")
        except Exception:                                  # noqa: BLE001
            return est, None
    if name == "XGBoost":
        try:
            est.set_params(n_estimators=200, max_depth=4,
                           learning_rate=0.05, reg_lambda=2.0)
            return est, (f"{name}: n={n}<{SMALL_N_THRESHOLD} → "
                         f"max_depth=4, n_estimators=200, reg_lambda=2.")
        except Exception:                                  # noqa: BLE001
            return est, None
    return est, None


def _topk(d: dict[str, float] | None, k: int = 30) -> dict[str, float] | None:
    if not d:
        return d
    items = sorted(d.items(), key=lambda kv: abs(kv[1]), reverse=True)[:k]
    return {k_: float(v_) for k_, v_ in items}


def _residual_diagnostics(y_true: np.ndarray, y_pred: np.ndarray,
                          max_points: int = 600,
                          n_bins: int = 20) -> dict:
    """
    Lightweight diagnostics that the dashboard can plot without heavy deps:
      * up to `max_points` (y_true, y_pred) pairs for the scatter
      * residual histogram (bin edges + counts)
      * Shapiro-Wilk on residuals (n>=3) — flags non-normal residuals
      * basic stats: mean, std, MAE, RMSE
    """
    from scipy import stats as _sp
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    resid = y_true - y_pred
    n = len(resid)

    if n > max_points:
        idx = np.linspace(0, n - 1, max_points).astype(int)
    else:
        idx = np.arange(n)

    counts, edges = np.histogram(resid, bins=n_bins)
    shapiro = {"W": None, "p": None, "non_normal_at_0_05": None}
    if 3 <= n <= 5000:
        try:
            W, p = _sp.shapiro(resid)
            shapiro = {"W": float(W), "p": float(p),
                       "non_normal_at_0_05": bool(p < 0.05)}
        except Exception:                                  # noqa: BLE001
            pass

    return {
        "n_points": int(n),
        "scatter": {
            "y_true": [float(v) for v in y_true[idx]],
            "y_pred": [float(v) for v in y_pred[idx]],
        },
        "histogram": {
            "edges":  [float(v) for v in edges],
            "counts": [int(v)   for v in counts],
        },
        "stats": {
            "residual_mean":  float(resid.mean()),
            "residual_std":   float(resid.std(ddof=1)) if n > 1 else 0.0,
            "mae":  float(np.mean(np.abs(resid))),
            "rmse": float(np.sqrt(np.mean(resid ** 2))),
            "y_true_min": float(y_true.min()),
            "y_true_max": float(y_true.max()),
        },
        "shapiro": shapiro,
        "interpretation": (
            "Resíduos com média ~0 e simétricos em torno de zero indicam ajuste "
            "razoável. Padrão sistemático no scatter (ex. heteroscedasticidade) "
            "indica que o modelo está mal especificado e o R² superestima."
        ),
    }


def _summarise_metric(values: list[float]) -> dict:
    arr = np.asarray(values, dtype=float)
    n = len(arr)
    mean = float(arr.mean())
    std  = float(arr.std(ddof=1)) if n > 1 else 0.0
    ci95 = 1.96 * std / np.sqrt(n) if n > 1 else 0.0
    return {"mean": mean, "std": std, "ci95": float(ci95), "n_folds": n,
            "min": float(arr.min()), "max": float(arr.max())}


# ---------------------------------------------------------------------------
# REGRESSION pipeline
# ---------------------------------------------------------------------------
def regression_train(df: pd.DataFrame,
                     features: Iterable[str] = DEFAULT_FEATURES_REGRESSION,
                     target: str = "rsrp_dbm",
                     group_by: str | None = None,
                     compute_importance: bool = True) -> dict:
    requested = list(features)
    clean, duplicate_rows, duplicate_key = deduplicate_measurements(df)
    X, y, feat_names, dropped_cols, rows_dropped = _prepare_features(
        clean, requested, target
    )
    if len(X) < 20:
        return {"error": f"Dados insuficientes para validação cruzada (n={len(X)} < 20)."}

    try:
        groups, normalized_group, n_groups, group_warning = _group_series(
            clean, X.index, group_by
        )
        cv, cv_scheme = _regression_cv(groups, n_groups)
    except ValueError as error:
        return {"error": str(error)}

    audit = TrainingAudit(
        task="regression", target=target,
        features_requested=requested, features_used=feat_names,
        features_dropped=dropped_cols,
        rows_input=int(len(df)),
        rows_dropped_exact_duplicates=duplicate_rows,
        duplicate_key=duplicate_key,
        rows_used=int(len(X)),
        rows_dropped_nan=int(rows_dropped),
        dataset_sha256=_dataset_hash(X, y),
        cv_scheme=cv_scheme,
        group_by=normalized_group,
        n_groups=n_groups,
        random_state=settings.random_state,
    )
    if duplicate_rows:
        audit.warnings.append(
            f"{duplicate_rows} pseudorreplicações exatas foram excluídas em "
            "memória antes da validação; o banco bruto foi preservado."
        )
    if group_warning:
        audit.warnings.append(group_warning)
    if normalized_group:
        audit.warnings.append(
            f"Validação agrupada por {normalized_group}: cada grupo fica inteiro "
            "em treino ou teste. Esta é a estimativa principal de generalização."
        )
    else:
        audit.warnings.append(
            "Validação aleatória por linha é otimista em rotas espacialmente "
            "autocorrelacionadas; use group_by=campaign_id, sector_code_effective "
            "ou date para a estimativa principal."
        )

    # Honest sample-size warning
    if len(X) < SMALL_N_THRESHOLD:
        audit.warnings.append(
            f"n={len(X)} < {SMALL_N_THRESHOLD}: árvores com hiperparâmetros "
            f"reduzidos automaticamente; trate o R² como exploratório."
        )

    models: dict[str, tuple[object, bool]] = {
        # (estimator, is_baseline)
        "Baseline_Mean": (DummyRegressor(strategy="mean"), True),
        "Baseline_Median": (DummyRegressor(strategy="median"), True),
        "LinearRegression": (LinearRegression(), False),
        "RandomForest": (RandomForestRegressor(
            n_estimators=300, max_depth=None, n_jobs=-1,
            random_state=settings.random_state), False),
        "XGBoost": (XGBRegressor(
            n_estimators=400, learning_rate=0.05, max_depth=6,
            subsample=0.9, colsample_bytree=0.9,
            random_state=settings.random_state, n_jobs=-1, verbosity=0), False),
        "SVR": (Pipeline([("scaler", StandardScaler()),
                         ("svr", SVR(kernel="rbf", C=10.0, epsilon=0.5))]),
                False),
    }

    for name, (est, is_baseline) in models.items():
        if not is_baseline:
            est, msg = _shrink_for_small_n(name, est, len(X))
            if msg:
                audit.warnings.append(msg)

        per_fold: list[FoldMetrics] = []
        r2s, maes, rmses = [], [], []
        # Out-of-fold predictions for honest residual diagnostics. Every row
        # of X gets a prediction made by a model that did NOT see it during
        # training (the fold where the row is in the test split). Averaging
        # across repeats keeps the prediction stable.
        oof_sum = np.zeros(len(X), dtype=float)
        oof_cnt = np.zeros(len(X), dtype=int)
        split_iterator = (
            cv.split(X, y, groups) if groups is not None else cv.split(X, y)
        )
        for k, (tr, te) in enumerate(split_iterator, start=1):
            m = _clone(est)
            m.fit(X.iloc[tr], y.iloc[tr])
            pred = m.predict(X.iloc[te])
            oof_sum[te] += pred
            oof_cnt[te] += 1
            r2  = r2_score(y.iloc[te], pred)
            mae = mean_absolute_error(y.iloc[te], pred)
            rmse = float(np.sqrt(mean_squared_error(y.iloc[te], pred)))
            r2s.append(r2); maes.append(mae); rmses.append(rmse)
            per_fold += [
                FoldMetrics(k, "R2",   float(r2)),
                FoldMetrics(k, "MAE",  float(mae)),
                FoldMetrics(k, "RMSE", float(rmse)),
            ]

        # Refit on all data for inspection
        full_est = _clone(est)
        full_est.fit(X, y)
        native = None
        perm   = None
        residuals = None
        if not is_baseline and compute_importance:
            native = _native_importance(full_est, feat_names)
            perm   = _permutation_importance(full_est, X, y, feat_names)
        # Residual diagnostics — compute for every model (incl. baseline) so
        # the dashboard can plot all of them on the same axes for comparison.
        oof_seen = oof_cnt > 0
        if oof_seen.any():
            y_pred_oof = oof_sum[oof_seen] / oof_cnt[oof_seen]
            residuals = _residual_diagnostics(
                y.values[oof_seen], y_pred_oof,
            )

        report = ModelReport(
            name=name, task="regression",
            n_train_total=int(len(X)), n_features=len(feat_names),
            feature_names=feat_names, target=target,
            hyperparameters=_hp(full_est),
            per_fold=per_fold,
            summary={
                "R2":   _summarise_metric(r2s),
                "MAE":  _summarise_metric(maes),
                "RMSE": _summarise_metric(rmses),
            },
            feature_importance=_topk(native),
            permutation_importance=_topk(perm),
            is_baseline=is_baseline,
            residual_diagnostics=residuals,
        )
        audit.models.append(report)

    return _audit_to_dict(audit)


# ---------------------------------------------------------------------------
# CLASSIFICATION pipeline (with confusion matrix)
# ---------------------------------------------------------------------------
def classification_train(df: pd.DataFrame,
                         features: Iterable[str] = DEFAULT_FEATURES_CLASSIFICATION,
                         target: str = "signal_rating",
                         group_by: str | None = None,
                         compute_importance: bool = True) -> dict:
    requested = list(features)
    clean, duplicate_rows, duplicate_key = deduplicate_measurements(df)
    X, y_raw, feat_names, dropped_cols, rows_dropped = _prepare_features(
        clean, requested, target
    )
    if len(X) < 20:
        return {"error": f"Dados insuficientes para validação estratificada (n={len(X)} < 20)."}

    try:
        groups, normalized_group, n_groups, group_warning = _group_series(
            clean, X.index, group_by
        )
        cv, cv_scheme = _classification_cv(groups, n_groups)
    except ValueError as error:
        return {"error": str(error)}

    # Encode target labels
    le = LabelEncoder()
    y = pd.Series(le.fit_transform(y_raw), index=y_raw.index)
    classes = list(le.classes_)
    if len(classes) < 2:
        return {"error": f"Apenas 1 classe presente em '{target}': {classes}."}

    # Need at least cv_splits per class
    counts = pd.Series(y).value_counts()
    if counts.min() < settings.cv_splits:
        return {
            "error": f"Classe(s) com poucas amostras para StratifiedKFold "
                     f"(min={int(counts.min())} < {settings.cv_splits}).",
            "class_counts": {classes[i]: int(counts[i]) for i in counts.index},
        }

    audit = TrainingAudit(
        task="classification", target=target,
        features_requested=requested, features_used=feat_names,
        features_dropped=dropped_cols,
        rows_input=int(len(df)),
        rows_dropped_exact_duplicates=duplicate_rows,
        duplicate_key=duplicate_key,
        rows_used=int(len(X)),
        rows_dropped_nan=int(rows_dropped),
        dataset_sha256=_dataset_hash(X, y_raw),
        cv_scheme=cv_scheme,
        group_by=normalized_group,
        n_groups=n_groups,
        random_state=settings.random_state,
    )
    if duplicate_rows:
        audit.warnings.append(
            f"{duplicate_rows} pseudorreplicações exatas foram excluídas em "
            "memória antes da validação; o banco bruto foi preservado."
        )
    if group_warning:
        audit.warnings.append(group_warning)
    if normalized_group:
        audit.warnings.append(
            f"Validação agrupada por {normalized_group}: cada grupo fica inteiro "
            "em treino ou teste."
        )
    else:
        audit.warnings.append(
            "Validação aleatória por linha é otimista em rotas espacialmente "
            "autocorrelacionadas."
        )

    if len(X) < SMALL_N_THRESHOLD:
        audit.warnings.append(
            f"n={len(X)} < {SMALL_N_THRESHOLD}: árvores reduzidas "
            f"automaticamente; trate métricas como exploratórias."
        )

    models: dict[str, tuple[object, bool]] = {
        "Baseline_Stratified": (DummyClassifier(strategy="stratified",
                                                random_state=settings.random_state),
                                True),
        "Baseline_MostFrequent": (DummyClassifier(strategy="most_frequent"),
                                  True),
        "RandomForest": (RandomForestClassifier(
            n_estimators=400, n_jobs=-1, class_weight="balanced",
            random_state=settings.random_state), False),
        # XGBClassifier auto-detects binary vs multi-class from y; passing
        # num_class + multi:softprob with len(classes)==2 makes it return a
        # one-hot prediction that sklearn's metrics reject. Pass the right
        # objective explicitly per arity.
        "XGBoost": (XGBClassifier(
            n_estimators=400, learning_rate=0.05, max_depth=6,
            subsample=0.9, colsample_bytree=0.9,
            random_state=settings.random_state, n_jobs=-1, verbosity=0,
            **({"objective": "binary:logistic", "eval_metric": "logloss"}
               if len(classes) == 2
               else {"objective": "multi:softprob",
                     "eval_metric": "mlogloss",
                     "num_class": len(classes)})), False),
        "SVC": (Pipeline([("scaler", StandardScaler()),
                         ("svc", SVC(kernel="rbf", C=2.0, gamma="scale",
                                     class_weight="balanced",
                                     random_state=settings.random_state))]),
                False),
    }

    for name, (est, is_baseline) in models.items():
        if not is_baseline:
            est, msg = _shrink_for_small_n(name, est, len(X))
            if msg:
                audit.warnings.append(msg)
        per_fold: list[FoldMetrics] = []
        accs, baccs, f1m = [], [], []
        cm_accum = np.zeros((len(classes), len(classes)), dtype=float)
        prfs_accum: list[np.ndarray] = []
        n_folds = 0
        split_iterator = (
            cv.split(X, y, groups) if groups is not None else cv.split(X, y)
        )
        folds_with_unseen_test_classes: list[int] = []
        for k, (tr, te) in enumerate(split_iterator, start=1):
            train_classes = np.sort(pd.Series(y.iloc[tr]).unique())
            test_classes = set(pd.Series(y.iloc[te]).unique())
            if not test_classes.issubset(set(train_classes)):
                folds_with_unseen_test_classes.append(k)

            # XGBoost requires labels in each training fold to be contiguous
            # from zero. Grouped validation can leave one global class wholly
            # in the test fold, so remap only for fitting and invert afterwards.
            if name == "XGBoost" and len(train_classes) >= 2:
                fold_map = {int(label): i for i, label in enumerate(train_classes)}
                inverse_map = {i: label for label, i in fold_map.items()}
                m = _clone(est)
                if len(train_classes) == 2:
                    m.set_params(
                        objective="binary:logistic",
                        eval_metric="logloss",
                        num_class=None,
                    )
                else:
                    m.set_params(
                        objective="multi:softprob",
                        eval_metric="mlogloss",
                        num_class=len(train_classes),
                    )
                y_train_fold = y.iloc[tr].map(fold_map).astype(int)
                m.fit(X.iloc[tr], y_train_fold)
                pred_fold = np.asarray(m.predict(X.iloc[te]), dtype=int)
                pred = np.asarray([inverse_map[int(value)] for value in pred_fold])
            elif name == "XGBoost":
                pred = np.repeat(train_classes[0], len(te))
            else:
                m = _clone(est)
                m.fit(X.iloc[tr], y.iloc[tr])
                pred = m.predict(X.iloc[te])
            acc  = accuracy_score(y.iloc[te], pred)
            bacc = balanced_accuracy_score(y.iloc[te], pred)
            f1   = f1_score(y.iloc[te], pred, average="macro", zero_division=0)
            cm   = confusion_matrix(y.iloc[te], pred,
                                    labels=list(range(len(classes))))
            cm_accum += cm
            n_folds += 1
            prf = precision_recall_fscore_support(
                y.iloc[te], pred, labels=list(range(len(classes))),
                zero_division=0)
            prfs_accum.append(np.stack([prf[0], prf[1], prf[2]], axis=0))
            accs.append(acc); baccs.append(bacc); f1m.append(f1)
            per_fold += [
                FoldMetrics(k, "accuracy", float(acc)),
                FoldMetrics(k, "balanced_accuracy", float(bacc)),
                FoldMetrics(k, "f1_macro", float(f1)),
            ]

        if folds_with_unseen_test_classes:
            audit.warnings.append(
                f"{name}: classe ausente no treino e presente no teste nas "
                f"dobras {folds_with_unseen_test_classes}; as metricas agrupadas "
                "nessas dobras evidenciam extrapolacao entre campanhas."
            )

        # Mean confusion matrix (rounded for display) + normalised per row
        cm_mean = (cm_accum / n_folds).tolist()
        prf_mean = np.mean(np.stack(prfs_accum, axis=0), axis=0)  # 3 x C
        per_class_report = {}
        for ci, cname in enumerate(classes):
            per_class_report[str(cname)] = {
                "precision": float(prf_mean[0, ci]),
                "recall":    float(prf_mean[1, ci]),
                "f1":        float(prf_mean[2, ci]),
                "support_total_test": int(cm_accum[ci].sum()),
            }

        # Refit + importance
        full_est = _clone(est)
        full_est.fit(X, y)
        native = None
        perm   = None
        if not is_baseline and compute_importance:
            native = _native_importance(full_est, feat_names)
            perm   = _permutation_importance(full_est, X, y, feat_names)

        report = ModelReport(
            name=name, task="classification",
            n_train_total=int(len(X)), n_features=len(feat_names),
            feature_names=feat_names, target=target,
            hyperparameters=_hp(full_est),
            per_fold=per_fold,
            summary={
                "accuracy":          _summarise_metric(accs),
                "balanced_accuracy": _summarise_metric(baccs),
                "f1_macro":          _summarise_metric(f1m),
            },
            confusion_matrix_mean=cm_mean,
            confusion_matrix_labels=[str(c) for c in classes],
            classification_report=per_class_report,
            feature_importance=_topk(native),
            permutation_importance=_topk(perm),
            is_baseline=is_baseline,
        )
        audit.models.append(report)

    out = _audit_to_dict(audit)
    out["class_counts"] = {classes[i]: int(counts[i]) for i in counts.index}
    return out


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _clone(est):
    from sklearn.base import clone
    try:
        return clone(est)
    except Exception:                                   # XGBoost edge case
        from copy import deepcopy
        return deepcopy(est)


def _hp(est) -> dict:
    """Stable, JSON-serialisable hyperparameter snapshot."""
    try:
        params = est.get_params(deep=False)
    except Exception:                                   # noqa: BLE001
        return {"_repr": repr(est)[:300]}
    out = {}
    for k, v in params.items():
        try:
            json.dumps(v)
            out[k] = v
        except Exception:                               # noqa: BLE001
            out[k] = repr(v)[:120]
    return out


def _native_importance(est, names: list[str]) -> dict[str, float] | None:
    """Tree-native impurity-based importance, when available. Biased toward
    high-cardinality features — pair with permutation importance below."""
    underlying = est
    if hasattr(est, "named_steps"):
        underlying = list(est.named_steps.values())[-1]
    if hasattr(underlying, "feature_importances_"):
        imp = underlying.feature_importances_
        if len(imp) == len(names):
            return {names[i]: float(imp[i]) for i in range(len(names))}
    return None


def _permutation_importance(est, X: pd.DataFrame, y: pd.Series,
                            names: list[str], n_repeats: int = 5
                            ) -> dict[str, float] | None:
    """Permutation importance — model-agnostic, less biased than native."""
    try:
        if len(X) > 5000:
            return None
        r = permutation_importance(est, X, y, n_repeats=n_repeats,
                                   random_state=settings.random_state,
                                   n_jobs=-1)
        return {names[i]: float(r.importances_mean[i]) for i in range(len(names))}
    except Exception:                                       # noqa: BLE001
        return None


def _audit_to_dict(a: TrainingAudit) -> dict:
    d = asdict(a)
    # Replace the per_fold dataclasses with plain dicts already (asdict handled).
    return d
