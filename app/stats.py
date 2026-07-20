"""
Scientific statistics. All routines:
  * never substitute NaN with 0 (uses pairwise-complete observations);
  * reject groups smaller than `min_group_size`;
  * apply Benjamini–Hochberg FDR for multiple comparisons in the
    correlation matrix.

P2 additions
------------
* `anova_robust`: classical F + Welch's F + Kruskal-Wallis side-by-side, with
  Shapiro-Wilk (normality) and Levene (homoscedasticity) diagnostics. Lets the
  reader choose the right test instead of pretending assumptions hold.
* `pairwise_hedges_g`: bias-corrected effect sizes with 95 % CI for pairwise
  comparisons of an independent factor.
* `summary_by_sector` / `summary_by_environment`: descriptive tables aligned
  with the new sector classifier (P1) — explicit `n`, missing rate, mean,
  median, IQR per metric.
"""
from __future__ import annotations
import math
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.multitest import multipletests


# ---------------------------------------------------------------------------
# ANOVA + Tukey HSD by an INDEPENDENT categorical factor
# ---------------------------------------------------------------------------
def anova_by_factor(df: pd.DataFrame, factor_col: str,
                    response_col: str = "rsrp_dbm",
                    min_group_size: int = 5) -> dict:
    """
    One-way ANOVA of `response_col` across levels of `factor_col`.
    Drops rows with NaN in either column. Skips groups with < min_group_size.
    Reports F, p, eta², group summaries, and Tukey HSD pairwise.
    """
    if factor_col not in df.columns or response_col not in df.columns:
        return {"error": f"Coluna ausente: {factor_col} ou {response_col}"}

    sub = df[[factor_col, response_col]].dropna()
    sub = sub[sub[factor_col].astype(str).str.len() > 0]
    levels = sub.groupby(factor_col)[response_col].apply(list)
    eligible = {k: np.array(v) for k, v in levels.items() if len(v) >= min_group_size}

    if len(eligible) < 2:
        return {
            "factor": factor_col,
            "response": response_col,
            "error": "Menos de 2 grupos com tamanho mínimo.",
            "min_group_size": min_group_size,
            "groups_seen": {k: int(len(v)) for k, v in levels.items()},
        }

    groups = list(eligible.values())
    f, p = stats.f_oneway(*groups)

    # Effect size: η²  =  SSB / SST
    grand = np.concatenate(groups)
    grand_mean = grand.mean()
    ssb = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in groups)
    sst = ((grand - grand_mean) ** 2).sum()
    eta2 = float(ssb / sst) if sst > 0 else float("nan")

    # Group summary
    summary = []
    for k, g in eligible.items():
        summary.append({
            "level": str(k),
            "n": int(len(g)),
            "mean": float(np.mean(g)),
            "std":  float(np.std(g, ddof=1)) if len(g) > 1 else 0.0,
            "median": float(np.median(g)),
            "p25": float(np.percentile(g, 25)),
            "p75": float(np.percentile(g, 75)),
        })

    # Tukey HSD
    tukey_rows: list[dict] = []
    if len(eligible) >= 2:
        tk = pairwise_tukeyhsd(
            endog=sub[sub[factor_col].isin(eligible.keys())][response_col].values,
            groups=sub[sub[factor_col].isin(eligible.keys())][factor_col].values,
            alpha=0.05,
        )
        for row in tk._results_table.data[1:]:
            tukey_rows.append({
                "group1": str(row[0]), "group2": str(row[1]),
                "meandiff": float(row[2]), "p_adj": float(row[3]),
                "lower": float(row[4]), "upper": float(row[5]),
                "reject": bool(row[6]),
            })

    return {
        "factor": factor_col,
        "response": response_col,
        "F": float(f), "p": float(p), "eta_squared": eta2,
        "n_total": int(sum(len(g) for g in groups)),
        "n_groups": int(len(eligible)),
        "min_group_size": min_group_size,
        "groups": summary,
        "tukey_hsd": tukey_rows,
    }


# ---------------------------------------------------------------------------
# Pearson with pairwise-complete observations and FDR control
# ---------------------------------------------------------------------------
def pearson_with_fdr(df: pd.DataFrame, cols: Iterable[str],
                     alpha: float = 0.05) -> dict:
    cols = [c for c in cols if c in df.columns]
    n = len(cols)
    r_mat = np.full((n, n), np.nan)
    p_mat = np.full((n, n), np.nan)
    n_mat = np.zeros((n, n), dtype=int)

    pvals: list[float] = []
    pidx:  list[tuple[int, int]] = []

    for i in range(n):
        for j in range(n):
            if i == j:
                r_mat[i, j], p_mat[i, j] = 1.0, 0.0
                n_mat[i, j] = int(df[cols[i]].notna().sum())
                continue
            sub = df[[cols[i], cols[j]]].dropna()
            n_mat[i, j] = len(sub)
            if len(sub) < 4:
                continue
            r, p = stats.pearsonr(sub[cols[i]], sub[cols[j]])
            r_mat[i, j] = r
            p_mat[i, j] = p
            if i < j:           # collect upper triangle once for FDR
                pvals.append(p)
                pidx.append((i, j))

    p_adj_mat = np.full((n, n), np.nan)
    rejected_mat = np.zeros((n, n), dtype=bool)
    if pvals:
        rej, p_adj, _, _ = multipletests(pvals, alpha=alpha, method="fdr_bh")
        for k, (i, j) in enumerate(pidx):
            p_adj_mat[i, j] = p_adj_mat[j, i] = p_adj[k]
            rejected_mat[i, j] = rejected_mat[j, i] = rej[k]

    def _fmt(M, fmt="{:.4f}"):
        return [
            [None if (isinstance(x, float) and math.isnan(x)) else fmt.format(x)
             for x in row]
            for row in M
        ]

    return {
        "cols": cols,
        "n_pairwise": n_mat.tolist(),
        "r":   _fmt(r_mat),
        "p":   _fmt(p_mat),
        "p_adj_fdr_bh": _fmt(p_adj_mat),
        "rejected_fdr": rejected_mat.tolist(),
        "alpha": alpha,
        "method": "Benjamini-Hochberg (FDR)",
    }


# ---------------------------------------------------------------------------
# Stratified summary by tech (5G vs 4G)
# ---------------------------------------------------------------------------
def stratified_summary(df: pd.DataFrame, by: str = "network_tech",
                       cols: Iterable[str] = ("rsrp_dbm", "sinr_db", "cqi",
                                              "ping_avg_ms", "ping_stdev_ms",
                                              "test_dl_max_kbps")) -> dict:
    out: dict[str, dict] = {}
    if by not in df.columns:
        return {"error": f"Coluna {by} ausente"}
    for level, grp in df.groupby(by, dropna=False):
        col_stats = {}
        for c in cols:
            if c not in df.columns:
                continue
            s = grp[c].dropna()
            if len(s) == 0:
                col_stats[c] = {"n": 0}
                continue
            col_stats[c] = {
                "n": int(len(s)),
                "mean": float(s.mean()),
                "std":  float(s.std(ddof=1)) if len(s) > 1 else 0.0,
                "median": float(s.median()),
                "min": float(s.min()), "max": float(s.max()),
            }
        out[str(level)] = {"n_rows": int(len(grp)), "metrics": col_stats}
    return out


# ===========================================================================
# P2 — Robust ANOVA, effect sizes, sector/environment summaries
# ===========================================================================
DEFAULT_METRICS = (
    "rsrp_dbm", "rsrq_db", "sinr_db", "cqi", "ltersssi_dbm",
    "ping_avg_ms", "ping_stdev_ms", "ping_loss_pct",
    "test_dl_max_kbps", "test_ul_max_kbps",
    "dl_bitrate_kbps", "ul_bitrate_kbps",
)


def _omega_squared(groups: list[np.ndarray]) -> float:
    """ω² — less biased than η² for small samples. Returns NaN if undefined."""
    k = len(groups)
    n_total = sum(len(g) for g in groups)
    if k < 2 or n_total <= k:
        return float("nan")
    grand = np.concatenate(groups)
    grand_mean = grand.mean()
    ssb = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in groups)
    ssw = sum(((g - g.mean()) ** 2).sum() for g in groups)
    sst = ssb + ssw
    mse = ssw / (n_total - k)
    denom = sst + mse
    if denom <= 0:
        return float("nan")
    return float((ssb - (k - 1) * mse) / denom)


def _welch_anova(groups: list[np.ndarray]) -> tuple[float, float, float, float]:
    """Welch's one-way ANOVA. Returns (F, p, df1, df2). NaNs if undefined."""
    k = len(groups)
    if k < 2:
        return (float("nan"),) * 4
    n  = np.array([len(g) for g in groups], dtype=float)
    if np.any(n < 2):
        return (float("nan"),) * 4
    m  = np.array([g.mean() for g in groups], dtype=float)
    v  = np.array([g.var(ddof=1) for g in groups], dtype=float)
    if np.any(v <= 0):
        # Degenerate group(s); Welch is undefined, fall back to NaN.
        return (float("nan"),) * 4
    w  = n / v
    W  = w.sum()
    mw = (w * m).sum() / W
    num = (w * (m - mw) ** 2).sum() / (k - 1)
    inv_df_term = (((1.0 - w / W) ** 2) / (n - 1.0)).sum()
    den = 1.0 + (2.0 * (k - 2) / (k * k - 1.0)) * inv_df_term
    F   = num / den
    df1 = k - 1
    df2 = (k * k - 1.0) / (3.0 * inv_df_term)
    p   = float(stats.f.sf(F, df1, df2))
    return float(F), p, float(df1), float(df2)


def anova_robust(df: pd.DataFrame, factor_col: str,
                 response_col: str = "rsrp_dbm",
                 min_group_size: int = 5,
                 alpha_assumption: float = 0.05) -> dict:
    """
    Robust one-way ANOVA workflow:
      1. Drop rows with NaN in factor or response.
      2. Drop groups smaller than `min_group_size`.
      3. Run Shapiro-Wilk per group (normality).
      4. Run Levene's test (homoscedasticity).
      5. Report classical F (`scipy.stats.f_oneway`),
         Welch's F (variance-aware) and Kruskal-Wallis H (non-parametric)
         simultaneously — the reader picks based on the assumption diagnostics.
      6. Report η² and ω² as effect sizes.

    The function does NOT silently choose a test. It emits a `recommendation`
    string explaining which result is most defensible given the diagnostics.
    """
    if factor_col not in df.columns or response_col not in df.columns:
        return {"error": f"Coluna ausente: {factor_col} ou {response_col}"}

    sub = df[[factor_col, response_col]].dropna()
    sub = sub[sub[factor_col].astype(str).str.len() > 0]
    levels = sub.groupby(factor_col)[response_col].apply(list)
    eligible = {k: np.array(v) for k, v in levels.items() if len(v) >= min_group_size}

    if len(eligible) < 2:
        return {
            "factor": factor_col, "response": response_col,
            "error": "Menos de 2 grupos com tamanho mínimo.",
            "min_group_size": min_group_size,
            "groups_seen": {str(k): int(len(v)) for k, v in levels.items()},
        }

    groups = list(eligible.values())
    keys   = [str(k) for k in eligible.keys()]

    # Diagnostics
    shapiro_rows = []
    any_non_normal = False
    for k, g in zip(keys, groups):
        if 3 <= len(g) <= 5000:
            W, p_w = stats.shapiro(g)
            non_norm = bool(p_w < alpha_assumption)
            any_non_normal = any_non_normal or non_norm
            shapiro_rows.append({
                "level": k, "n": int(len(g)),
                "W": float(W), "p": float(p_w),
                "non_normal_at_alpha": non_norm,
            })
        else:
            shapiro_rows.append({"level": k, "n": int(len(g)),
                                 "W": None, "p": None,
                                 "non_normal_at_alpha": None})

    levene_W, levene_p = stats.levene(*groups, center="median")
    heteroscedastic = bool(levene_p < alpha_assumption)

    # Tests
    F_classic, p_classic = stats.f_oneway(*groups)
    F_welch, p_welch, df1_w, df2_w = _welch_anova(groups)
    H_kw, p_kw = stats.kruskal(*groups)

    # Effect sizes
    grand = np.concatenate(groups)
    grand_mean = grand.mean()
    ssb = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in groups)
    sst = ((grand - grand_mean) ** 2).sum()
    eta2  = float(ssb / sst) if sst > 0 else float("nan")
    omega2 = _omega_squared(groups)

    # Group summary
    summary = []
    for k, g in zip(keys, groups):
        ci_low, ci_high = (float("nan"), float("nan"))
        if len(g) > 1:
            sem = stats.sem(g)
            ci_low, ci_high = stats.t.interval(0.95, len(g) - 1,
                                                loc=g.mean(), scale=sem)
        summary.append({
            "level": k, "n": int(len(g)),
            "mean": float(g.mean()),
            "std":  float(g.std(ddof=1)) if len(g) > 1 else 0.0,
            "median": float(np.median(g)),
            "p25": float(np.percentile(g, 25)),
            "p75": float(np.percentile(g, 75)),
            "ci95_low":  None if math.isnan(ci_low)  else float(ci_low),
            "ci95_high": None if math.isnan(ci_high) else float(ci_high),
        })

    if heteroscedastic and any_non_normal:
        recommendation = ("Variâncias desiguais (Levene rejeita H0) e ao menos "
                          "um grupo não-normal (Shapiro): use Kruskal-Wallis.")
    elif heteroscedastic:
        recommendation = ("Variâncias desiguais (Levene rejeita H0): use F de "
                          "Welch em vez do F clássico.")
    elif any_non_normal and min(len(g) for g in groups) < 30:
        recommendation = ("Não-normalidade com amostras pequenas: prefira "
                          "Kruskal-Wallis. Para n grande (≥30) o F clássico é "
                          "robusto via TLC.")
    else:
        recommendation = ("Hipóteses razoáveis: o F clássico é apropriado; "
                          "Welch e Kruskal devem concordar.")

    return {
        "factor": factor_col, "response": response_col,
        "min_group_size": min_group_size,
        "n_total": int(grand.size), "n_groups": int(len(groups)),
        "groups": summary,
        "diagnostics": {
            "shapiro_per_group": shapiro_rows,
            "levene_W": float(levene_W), "levene_p": float(levene_p),
            "heteroscedastic_at_alpha": heteroscedastic,
            "alpha": alpha_assumption,
        },
        "tests": {
            "anova_classic": {"F": float(F_classic), "p": float(p_classic),
                              "df1": int(len(groups) - 1),
                              "df2": int(grand.size - len(groups))},
            "anova_welch":   {"F": F_welch, "p": p_welch,
                              "df1": df1_w, "df2": df2_w},
            "kruskal_wallis": {"H": float(H_kw), "p": float(p_kw),
                               "df": int(len(groups) - 1)},
        },
        "effect_size": {"eta_squared": eta2, "omega_squared": omega2},
        "recommendation": recommendation,
    }


def pairwise_hedges_g(df: pd.DataFrame, factor_col: str,
                      response_col: str = "rsrp_dbm",
                      min_group_size: int = 5,
                      alpha: float = 0.05) -> dict:
    """
    Pairwise Hedges' g (bias-corrected Cohen's d) with 95 % CI for every
    distinct pair of factor levels. Effect-size magnitudes (Cohen, 1988):
      |g| < 0.2  → negligible
      |g| < 0.5  → small
      |g| < 0.8  → medium
      |g| ≥ 0.8  → large
    """
    if factor_col not in df.columns or response_col not in df.columns:
        return {"error": f"Coluna ausente: {factor_col} ou {response_col}"}

    sub = df[[factor_col, response_col]].dropna()
    levels = sub.groupby(factor_col)[response_col].apply(list)
    eligible = {str(k): np.array(v) for k, v in levels.items()
                if len(v) >= min_group_size}
    keys = list(eligible.keys())
    if len(keys) < 2:
        return {"factor": factor_col, "response": response_col,
                "error": "Menos de 2 grupos com tamanho mínimo.",
                "min_group_size": min_group_size}

    z = stats.norm.ppf(1 - alpha / 2)
    pairs = []
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a, b = eligible[keys[i]], eligible[keys[j]]
            n1, n2 = len(a), len(b)
            s1, s2 = a.std(ddof=1), b.std(ddof=1)
            sp = math.sqrt(((n1 - 1) * s1 ** 2 + (n2 - 1) * s2 ** 2)
                           / (n1 + n2 - 2)) if (n1 + n2) > 2 else float("nan")
            if sp <= 0 or math.isnan(sp):
                pairs.append({"group1": keys[i], "group2": keys[j],
                              "g": None, "ci95_low": None, "ci95_high": None,
                              "magnitude": "indeterminado",
                              "n1": n1, "n2": n2})
                continue
            d  = (a.mean() - b.mean()) / sp
            J  = 1.0 - 3.0 / (4.0 * (n1 + n2) - 9.0)
            g  = J * d
            se = math.sqrt((n1 + n2) / (n1 * n2) + g ** 2 / (2 * (n1 + n2)))
            mag = ("desprezível" if abs(g) < 0.2 else
                   "pequeno"      if abs(g) < 0.5 else
                   "médio"        if abs(g) < 0.8 else "grande")
            pairs.append({
                "group1": keys[i], "group2": keys[j],
                "n1": n1, "n2": n2,
                "mean1": float(a.mean()), "mean2": float(b.mean()),
                "g": float(g),
                "se":   float(se),
                "ci95_low":  float(g - z * se),
                "ci95_high": float(g + z * se),
                "magnitude": mag,
            })
    return {
        "factor": factor_col, "response": response_col,
        "alpha": alpha, "n_groups": len(keys),
        "pairs": pairs,
        "interpretation": ("Hedges' g é bias-corrigido para n pequeno. CI 95 % "
                           "que cruza zero ⇒ diferença não distinguível."),
    }


def _metric_block(s: pd.Series) -> dict:
    """Descriptive stats for one metric in one group."""
    valid = s.dropna()
    if len(valid) == 0:
        return {"n_valid": 0, "n_total": int(len(s)), "missing_pct": 100.0}
    return {
        "n_valid": int(len(valid)),
        "n_total": int(len(s)),
        "missing_pct": float(round(100.0 * (1 - len(valid) / len(s)), 2)),
        "mean":   float(valid.mean()),
        "std":    float(valid.std(ddof=1)) if len(valid) > 1 else 0.0,
        "median": float(valid.median()),
        "p25":    float(valid.quantile(0.25)),
        "p75":    float(valid.quantile(0.75)),
        "min":    float(valid.min()),
        "max":    float(valid.max()),
    }


def summary_by_sector(df: pd.DataFrame,
                      metrics: Iterable[str] = DEFAULT_METRICS) -> dict:
    """
    Per-sector descriptive table. Prefer 'sector_code_effective'
    (manual > buffer > strict) when available — captures muito mais pontos
    que 'sector_code' puro (que so cobre quem caiu DENTRO do poligono).
    """
    sector_col = ("sector_code_effective" if "sector_code_effective" in df.columns
                  else "sector_code" if "sector_code" in df.columns
                  else None)
    if sector_col is None:
        return {"error": "Nenhuma coluna de setor disponivel — execute /api/sectors/reclassify."}
    name_col = ("sector_name_effective" if "sector_name_effective" in df.columns
                else "sector_name" if "sector_name" in df.columns else None)
    env_col  = ("environment_class_effective" if "environment_class_effective" in df.columns
                else "environment_class" if "environment_class" in df.columns else None)

    cols = [c for c in metrics if c in df.columns]
    out = []
    for level, grp in df.groupby(df[sector_col].fillna("unclassified"),
                                 dropna=False):
        env = (grp[env_col].dropna().unique().tolist()
               if env_col else [])
        name = (grp[name_col].dropna().unique().tolist()
                if name_col else [])
        out.append({
            "sector_code": str(level),
            "sector_name": name[0] if name else None,
            "environment_class": env[0] if env else None,
            "n_rows": int(len(grp)),
            "metrics": {c: _metric_block(grp[c]) for c in cols},
        })
    out.sort(key=lambda x: x["sector_code"])
    return {
        "source_column": sector_col,
        "n_sectors_with_data": len([x for x in out if x["sector_code"] != "unclassified"]),
        "sectors": out,
        "policy": ("Agrupa pelo setor 'efetivo' (manual > buffer > strict). "
                   "Linhas totalmente sem classificacao aparecem como 'unclassified'."),
    }


def minimum_detectable_effect(df: pd.DataFrame, factor_col: str,
                              response_col: str = "rsrp_dbm",
                              alpha: float = 0.05, power: float = 0.80,
                              min_group_size: int = 5) -> dict:
    """
    Post-hoc minimum detectable effect (MDE) per group.

    For each level of `factor_col` with n_i ≥ min_group_size, reports the
    smallest pairwise Cohen's d that the current sample size would let you
    detect with the chosen alpha and power, against the largest other group
    (worst-case companion). This answers the banca's "you only have n=12 — can
    you actually conclude anything?" question without forcing the student to
    plan more measurements they cannot collect.

    Interpretation:
        small  : d ≥ 0.2
        medium : d ≥ 0.5
        large  : d ≥ 0.8
    A reported MDE of d=0.85 means *only* large effects are distinguishable.
    """
    from statsmodels.stats.power import TTestIndPower

    if factor_col not in df.columns or response_col not in df.columns:
        return {"error": f"Coluna ausente: {factor_col} ou {response_col}"}

    sub = df[[factor_col, response_col]].dropna()
    counts = sub.groupby(factor_col).size()
    eligible = {str(k): int(v) for k, v in counts.items() if v >= min_group_size}
    if len(eligible) < 2:
        return {"factor": factor_col, "response": response_col,
                "error": "Menos de 2 grupos com tamanho mínimo.",
                "min_group_size": min_group_size}

    pwr = TTestIndPower()
    rows = []
    for level, n in eligible.items():
        # Worst-case companion: the LARGEST other group (we have most info
        # against it, so MDE depends on this group's n). The asymmetric
        # ratio is handled by `ratio = n_other / n`.
        n_other = max((v for k, v in eligible.items() if k != level), default=n)
        try:
            d = pwr.solve_power(effect_size=None, nobs1=n, alpha=alpha,
                                power=power, ratio=n_other / n,
                                alternative="two-sided")
        except Exception:                                  # noqa: BLE001
            d = float("nan")
        mag = ("desprezível"     if d < 0.2 else
               "pequeno"         if d < 0.5 else
               "médio"           if d < 0.8 else "grande")
        rows.append({
            "level": level, "n": n, "n_companion": n_other,
            "min_detectable_d": None if (d != d) else float(d),
            "magnitude_floor": mag,
        })
    rows.sort(key=lambda r: (r["min_detectable_d"] is None,
                              r["min_detectable_d"] or 0))
    return {
        "factor": factor_col, "response": response_col,
        "alpha": alpha, "power": power,
        "min_group_size": min_group_size,
        "rows": rows,
        "interpretation": ("MDE = menor efeito (d de Cohen) detectável com "
                           "esta amostra. Efeitos abaixo do MDE não são "
                           "distinguíveis do ruído com o n disponível."),
    }


def summary_by_environment(df: pd.DataFrame,
                           metrics: Iterable[str] = DEFAULT_METRICS) -> dict:
    """Per environment_class table (edificado / aberto / arborizado / null).
    Prefere environment_class_effective (que herda da LEGEND quando o
    aluno declarou manual_sector). Captura muito mais pontos."""
    env_col = ("environment_class_effective" if "environment_class_effective" in df.columns
               else "environment_class" if "environment_class" in df.columns
               else None)
    if env_col is None:
        return {"error": "Nenhuma coluna de classe ambiental disponivel."}
    sector_col = ("sector_code_effective" if "sector_code_effective" in df.columns
                  else "sector_code" if "sector_code" in df.columns else None)
    cols = [c for c in metrics if c in df.columns]
    out = []
    grouper = df[env_col].fillna("indefinido")
    for level, grp in df.groupby(grouper, dropna=False):
        out.append({
            "environment_class": str(level),
            "n_rows": int(len(grp)),
            "n_distinct_sectors": int(grp[sector_col].dropna().nunique())
                                  if sector_col else None,
            "metrics": {c: _metric_block(grp[c]) for c in cols},
        })
    out.sort(key=lambda x: x["environment_class"])
    return {"source_column": env_col, "environments": out}
