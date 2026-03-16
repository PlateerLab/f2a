"""Statistical hypothesis tests module.

Provides Levene, Kruskal-Wallis, Mann-Whitney, Chi-Square goodness-of-fit,
Grubbs outlier test, and Augmented Dickey-Fuller stationarity test.

**Enhancements over v1**:

* **Kruskal-Wallis** now uses categorical columns as grouping variables
  so each test compares one numeric column across groups of a factor — the
  semantically correct usage.
* **Benjamini-Hochberg FDR** correction is applied to all pairwise /
  multi-test batteries (Levene, Mann-Whitney, Kruskal-Wallis).
* **Effect sizes** are reported alongside every test:
  - rank-biserial *r* for Mann-Whitney U
  - η² (eta-squared) for Kruskal-Wallis H
  - Cohen's *d* proxy for Levene (log-variance difference)
  - Cramér's *V* for Chi-Square

References:
    - Levene (1960) — equality of variances
    - Kruskal & Wallis (1952) — non-parametric one-way ANOVA
    - Mann & Whitney (1947) — two-sample rank test
    - Grubbs (1950) — single-outlier test
    - Dickey & Fuller (1979) — stationarity test
    - Benjamini & Hochberg (1995) — FDR control
    - Rosenthal (1991) — rank-biserial correlation
    - Cohen (1988) — effect size conventions
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from f2a.core.schema import DataSchema
from f2a.utils.logging import get_logger

logger = get_logger(__name__)

# ── Utility: Benjamini-Hochberg FDR correction ───────────


def _bh_adjust(p_values: list[float]) -> list[float]:
    """Return Benjamini-Hochberg adjusted p-values.

    Args:
        p_values: Raw p-values (same order as rows).

    Returns:
        Adjusted p-values clipped to [0, 1].
    """
    m = len(p_values)
    if m == 0:
        return []
    arr = np.asarray(p_values, dtype=float)
    order = np.argsort(arr)
    ranked = np.empty_like(arr)
    ranked[order] = np.arange(1, m + 1)

    adjusted = arr * m / ranked
    # enforce monotonicity (descending by rank order)
    sorted_idx = np.argsort(ranked)[::-1]
    cum_min = np.minimum.accumulate(adjusted[sorted_idx])
    adjusted[sorted_idx] = cum_min
    return np.clip(adjusted, 0.0, 1.0).tolist()


def _significance_stars(p: float) -> str:
    """Return significance star annotation."""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    if p < 0.1:
        return "†"
    return "ns"


class StatisticalTests:
    """Perform various statistical hypothesis tests.

    Args:
        df: Target DataFrame.
        schema: Data schema.
    """

    _MAX_PAIRWISE = 15
    _MAX_CATEGORIES = 20
    _MIN_GROUP_SIZE = 5

    def __init__(self, df: pd.DataFrame, schema: DataSchema) -> None:
        self._df = df
        self._schema = schema

    # ── Levene's test (homogeneity of variances) ──────────

    def levene_test(self) -> pd.DataFrame:
        """Levene's test for equality of variances across numeric columns.

        Tests whether pairs of numeric columns have equal variances.
        Results include BH-adjusted p-values and a log-variance-ratio
        effect size proxy.

        Returns:
            DataFrame with pairwise Levene test results.
        """
        cols = self._schema.numeric_columns
        if len(cols) < 2:
            return pd.DataFrame()

        cols = cols[: self._MAX_PAIRWISE]
        rows: list[dict] = []

        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                a = self._df[cols[i]].dropna().values
                b = self._df[cols[j]].dropna().values
                if len(a) < 3 or len(b) < 3:
                    continue
                try:
                    stat, p = sp_stats.levene(a, b)
                    # Effect size: absolute log-variance ratio
                    var_a = float(np.var(a, ddof=1)) if len(a) > 1 else 1e-12
                    var_b = float(np.var(b, ddof=1)) if len(b) > 1 else 1e-12
                    log_var_ratio = abs(
                        float(np.log(max(var_a, 1e-12) / max(var_b, 1e-12)))
                    )
                    rows.append({
                        "col_a": cols[i],
                        "col_b": cols[j],
                        "levene_stat": round(float(stat), 4),
                        "p_value": round(float(p), 6),
                        "log_var_ratio": round(log_var_ratio, 4),
                    })
                except Exception:
                    continue

        if not rows:
            return pd.DataFrame()

        # BH-adjusted p-values
        raw_p = [r["p_value"] for r in rows]
        adj_p = _bh_adjust(raw_p)
        for r, ap in zip(rows, adj_p):
            r["adjusted_p"] = round(ap, 6)
            r["significant_0.05"] = ap < 0.05
            r["stars"] = _significance_stars(ap)

        return pd.DataFrame(rows)

    # ── Kruskal-Wallis test ───────────────────────────────

    def kruskal_wallis(self) -> pd.DataFrame:
        """Kruskal-Wallis H-test: numeric column grouped by categorical factor.

        For each (categorical, numeric) pair the test checks whether the
        numeric distribution differs across the levels of the factor.
        Reports η² (eta-squared) effect size and BH-adjusted p-values.

        Returns:
            DataFrame with one row per (grouping_col, numeric_col) pair.
        """
        num_cols = self._schema.numeric_columns
        cat_cols = self._schema.categorical_columns

        if not num_cols or not cat_cols:
            return pd.DataFrame()

        # Limit to manageable size
        cat_cols = cat_cols[:10]
        num_cols = num_cols[:15]

        rows: list[dict] = []

        for cat in cat_cols:
            groups_series = self._df[cat]
            unique_vals = groups_series.dropna().unique()
            # skip useless groupings (1 group, or >50 levels)
            if len(unique_vals) < 2 or len(unique_vals) > 50:
                continue

            for num in num_cols:
                sub = self._df[[cat, num]].dropna()
                grouped = [
                    grp[num].values
                    for _, grp in sub.groupby(cat)
                    if len(grp) >= self._MIN_GROUP_SIZE
                ]
                if len(grouped) < 2:
                    continue

                try:
                    stat, p = sp_stats.kruskal(*grouped)
                    n_total = sum(len(g) for g in grouped)
                    k = len(grouped)
                    # η² = (H - k + 1) / (n - k)
                    eta_sq = max(
                        0.0, (float(stat) - k + 1) / (n_total - k)
                    ) if n_total > k else 0.0
                    rows.append({
                        "grouping_col": cat,
                        "numeric_col": num,
                        "n_groups": k,
                        "h_statistic": round(float(stat), 4),
                        "p_value": round(float(p), 6),
                        "eta_squared": round(eta_sq, 4),
                        "effect_magnitude": (
                            "large" if eta_sq >= 0.14
                            else "medium" if eta_sq >= 0.06
                            else "small"
                        ),
                    })
                except Exception:
                    continue

        if not rows:
            return pd.DataFrame()

        # BH correction
        raw_p = [r["p_value"] for r in rows]
        adj_p = _bh_adjust(raw_p)
        for r, ap in zip(rows, adj_p):
            r["adjusted_p"] = round(ap, 6)
            r["reject_h0_0.05"] = ap < 0.05
            r["stars"] = _significance_stars(ap)
            r["interpretation"] = (
                f"Significant (η²={r['eta_squared']}, {r['effect_magnitude']})"
                if ap < 0.05
                else "No significant difference"
            )

        return pd.DataFrame(rows)

    # ── Mann-Whitney U test ───────────────────────────────

    def mann_whitney(self) -> pd.DataFrame:
        """Pairwise Mann-Whitney U tests between numeric columns.

        Reports rank-biserial *r* effect size (Rosenthal, 1991) and
        BH-adjusted p-values.

        Returns:
            DataFrame with col_a, col_b, U-stat, p-value, effect size.
        """
        cols = self._schema.numeric_columns
        if len(cols) < 2:
            return pd.DataFrame()

        cols = cols[: self._MAX_PAIRWISE]
        rows: list[dict] = []

        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                a = self._df[cols[i]].dropna().values
                b = self._df[cols[j]].dropna().values
                if len(a) < self._MIN_GROUP_SIZE or len(b) < self._MIN_GROUP_SIZE:
                    continue
                try:
                    stat, p = sp_stats.mannwhitneyu(a, b, alternative="two-sided")
                    n1, n2 = len(a), len(b)
                    # rank-biserial r = 1 - 2U / (n1 * n2)
                    r_rb = 1.0 - 2.0 * float(stat) / (n1 * n2)
                    rows.append({
                        "col_a": cols[i],
                        "col_b": cols[j],
                        "u_statistic": round(float(stat), 2),
                        "p_value": round(float(p), 6),
                        "rank_biserial_r": round(r_rb, 4),
                        "effect_magnitude": (
                            "large" if abs(r_rb) >= 0.5
                            else "medium" if abs(r_rb) >= 0.3
                            else "small"
                        ),
                    })
                except Exception:
                    continue

        if not rows:
            return pd.DataFrame()

        # BH correction
        raw_p = [r["p_value"] for r in rows]
        adj_p = _bh_adjust(raw_p)
        for r, ap in zip(rows, adj_p):
            r["adjusted_p"] = round(ap, 6)
            r["significant_0.05"] = ap < 0.05
            r["stars"] = _significance_stars(ap)

        return pd.DataFrame(rows)

    # ── Chi-square goodness-of-fit ────────────────────────

    def chi_square_goodness(self) -> pd.DataFrame:
        """Chi-square goodness-of-fit test for categorical columns.

        Tests whether observed frequencies differ from expected uniform.
        Reports Cramér's *V* effect size.

        Returns:
            DataFrame with test results per categorical column.
        """
        cols = self._schema.categorical_columns
        if not cols:
            return pd.DataFrame()

        rows: list[dict] = []
        for col in cols[: self._MAX_CATEGORIES]:
            vc = self._df[col].value_counts()
            if len(vc) < 2 or len(vc) > 100:
                continue

            observed = vc.values.astype(float)
            expected = np.full_like(observed, observed.mean())
            n_obs = float(observed.sum())
            k = len(vc)

            try:
                stat, p = sp_stats.chisquare(observed, f_exp=expected)
                # Cramér's V for goodness-of-fit: sqrt(chi2 / (n*(k-1)))
                cramers_v = float(np.sqrt(stat / (n_obs * (k - 1)))) if k > 1 else 0.0
                rows.append({
                    "column": col,
                    "n_categories": k,
                    "chi2_stat": round(float(stat), 4),
                    "p_value": round(float(p), 6),
                    "cramers_v": round(cramers_v, 4),
                    "effect_magnitude": (
                        "large" if cramers_v >= 0.5
                        else "medium" if cramers_v >= 0.3
                        else "small"
                    ),
                    "uniform_0.05": float(p) > 0.05,
                    "interpretation": (
                        "Approximately uniform"
                        if float(p) > 0.05
                        else "Non-uniform distribution"
                    ),
                })
            except Exception:
                continue

        return pd.DataFrame(rows).set_index("column") if rows else pd.DataFrame()

    # ── Grubbs' outlier test ──────────────────────────────

    def grubbs_test(self, alpha: float = 0.05) -> pd.DataFrame:
        """Grubbs' test for a single outlier in each numeric column.

        Tests whether the maximum or minimum value is an outlier
        assuming normal distribution.

        Args:
            alpha: Significance level.

        Returns:
            DataFrame with test results per column.
        """
        cols = self._schema.numeric_columns
        if not cols:
            return pd.DataFrame()

        rows: list[dict] = []
        for col in cols:
            series = self._df[col].dropna()
            n = len(series)
            if n < 7:
                continue

            mean = float(series.mean())
            std = float(series.std())
            if std == 0:
                continue

            # Test statistic = max(|x_i - mean|) / std
            max_diff_idx = (series - mean).abs().idxmax()
            max_val = float(series.loc[max_diff_idx])
            g_stat = abs(max_val - mean) / std

            # Critical value (t-distribution)
            t_crit = float(sp_stats.t.ppf(1 - alpha / (2 * n), n - 2))
            g_crit = (n - 1) / np.sqrt(n) * np.sqrt(t_crit**2 / (n - 2 + t_crit**2))

            is_outlier = g_stat > g_crit

            rows.append({
                "column": col,
                "suspect_value": round(max_val, 4),
                "grubbs_statistic": round(float(g_stat), 4),
                "critical_value": round(float(g_crit), 4),
                "is_outlier": is_outlier,
                "n": n,
            })

        return pd.DataFrame(rows).set_index("column") if rows else pd.DataFrame()

    # ── Augmented Dickey-Fuller (stationarity) ────────────

    def adf_test(self) -> pd.DataFrame:
        """Augmented Dickey-Fuller test for stationarity.

        Tests whether a numeric time-series is stationary.
        H0: The series has a unit root (non-stationary).

        Returns:
            DataFrame with ADF results per numeric column.
        """
        cols = self._schema.numeric_columns
        if not cols:
            return pd.DataFrame()

        try:
            from statsmodels.tsa.stattools import adfuller
        except ImportError:
            logger.info("statsmodels not available; skipping ADF test.")
            return pd.DataFrame()

        rows: list[dict] = []
        for col in cols:
            series = self._df[col].dropna()
            if len(series) < 20:
                continue
            try:
                result = adfuller(series, autolag="AIC")
                adf_stat, p_val, used_lag, nobs, critical_values, ic_best = result
                rows.append({
                    "column": col,
                    "adf_statistic": round(float(adf_stat), 4),
                    "p_value": round(float(p_val), 6),
                    "used_lag": int(used_lag),
                    "n_observations": int(nobs),
                    "critical_1%": round(float(critical_values["1%"]), 4),
                    "critical_5%": round(float(critical_values["5%"]), 4),
                    "critical_10%": round(float(critical_values["10%"]), 4),
                    "is_stationary_0.05": float(p_val) < 0.05,
                })
            except Exception:
                continue

        return pd.DataFrame(rows).set_index("column") if rows else pd.DataFrame()

    # ── Summary ───────────────────────────────────────────

    def summary(self) -> dict[str, Any]:
        """Return combined statistical test results."""
        result: dict[str, Any] = {}

        try:
            lev = self.levene_test()
            if not lev.empty:
                result["levene"] = lev
        except Exception as exc:
            logger.debug("Levene test skipped: %s", exc)

        try:
            kw = self.kruskal_wallis()
            if not kw.empty:
                result["kruskal_wallis"] = kw
        except Exception as exc:
            logger.debug("Kruskal-Wallis skipped: %s", exc)

        try:
            mw = self.mann_whitney()
            if not mw.empty:
                result["mann_whitney"] = mw
        except Exception as exc:
            logger.debug("Mann-Whitney skipped: %s", exc)

        try:
            csq = self.chi_square_goodness()
            if not csq.empty:
                result["chi_square_goodness"] = csq
        except Exception as exc:
            logger.debug("Chi-square goodness skipped: %s", exc)

        try:
            grb = self.grubbs_test()
            if not grb.empty:
                result["grubbs"] = grb
        except Exception as exc:
            logger.debug("Grubbs test skipped: %s", exc)

        try:
            adf = self.adf_test()
            if not adf.empty:
                result["adf"] = adf
        except Exception as exc:
            logger.debug("ADF test skipped: %s", exc)

        return result
