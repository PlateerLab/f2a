"""Statistical hypothesis tests module.

Provides Levene, Kruskal-Wallis, Mann-Whitney, Chi-Square goodness-of-fit,
Grubbs outlier test, and Augmented Dickey-Fuller stationarity test.

References:
    - Levene (1960) — equality of variances
    - Kruskal & Wallis (1952) — non-parametric one-way ANOVA
    - Mann & Whitney (1947) — two-sample rank test
    - Grubbs (1950) — single-outlier test
    - Dickey & Fuller (1979) — stationarity test
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from f2a.core.schema import DataSchema
from f2a.utils.logging import get_logger

logger = get_logger(__name__)


class StatisticalTests:
    """Perform various statistical hypothesis tests.

    Args:
        df: Target DataFrame.
        schema: Data schema.
    """

    def __init__(self, df: pd.DataFrame, schema: DataSchema) -> None:
        self._df = df
        self._schema = schema

    # ── Levene's test (homogeneity of variances) ──────────

    def levene_test(self) -> pd.DataFrame:
        """Levene's test for equality of variances across numeric columns.

        Tests whether columns have equal variances — useful before ANOVA.

        Returns:
            DataFrame with pairwise Levene test results.
        """
        cols = self._schema.numeric_columns
        if len(cols) < 2:
            return pd.DataFrame()

        cols = cols[:15]
        rows: list[dict] = []

        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                a = self._df[cols[i]].dropna().values
                b = self._df[cols[j]].dropna().values
                if len(a) < 3 or len(b) < 3:
                    continue
                try:
                    stat, p = sp_stats.levene(a, b)
                    rows.append({
                        "col_a": cols[i],
                        "col_b": cols[j],
                        "levene_stat": round(float(stat), 4),
                        "p_value": round(float(p), 6),
                        "equal_variance_0.05": float(p) > 0.05,
                    })
                except Exception:
                    continue

        return pd.DataFrame(rows) if rows else pd.DataFrame()

    # ── Kruskal-Wallis test ───────────────────────────────

    def kruskal_wallis(self) -> pd.DataFrame:
        """Kruskal-Wallis H-test across numeric columns.

        Non-parametric alternative to one-way ANOVA. Tests whether
        the distributions of all numeric columns come from the same population.

        Returns:
            DataFrame with test results for column groups.
        """
        cols = self._schema.numeric_columns
        if len(cols) < 2:
            return pd.DataFrame()

        cols = cols[:20]
        groups = [self._df[c].dropna().values for c in cols if len(self._df[c].dropna()) >= 5]

        if len(groups) < 2:
            return pd.DataFrame()

        try:
            stat, p = sp_stats.kruskal(*groups)
            return pd.DataFrame([{
                "test": "Kruskal-Wallis",
                "n_groups": len(groups),
                "h_statistic": round(float(stat), 4),
                "p_value": round(float(p), 6),
                "reject_h0_0.05": float(p) < 0.05,
                "interpretation": (
                    "At least one group differs"
                    if float(p) < 0.05
                    else "No significant difference"
                ),
            }]).set_index("test")
        except Exception:
            return pd.DataFrame()

    # ── Mann-Whitney U test ───────────────────────────────

    def mann_whitney(self) -> pd.DataFrame:
        """Pairwise Mann-Whitney U tests between numeric columns.

        Non-parametric test for equal medians between two samples.

        Returns:
            DataFrame with col_a, col_b, U_stat, p_value.
        """
        cols = self._schema.numeric_columns
        if len(cols) < 2:
            return pd.DataFrame()

        cols = cols[:15]
        rows: list[dict] = []

        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                a = self._df[cols[i]].dropna().values
                b = self._df[cols[j]].dropna().values
                if len(a) < 5 or len(b) < 5:
                    continue
                try:
                    stat, p = sp_stats.mannwhitneyu(a, b, alternative="two-sided")
                    rows.append({
                        "col_a": cols[i],
                        "col_b": cols[j],
                        "u_statistic": round(float(stat), 2),
                        "p_value": round(float(p), 6),
                        "significant_0.05": float(p) < 0.05,
                    })
                except Exception:
                    continue

        return pd.DataFrame(rows) if rows else pd.DataFrame()

    # ── Chi-square goodness-of-fit ────────────────────────

    def chi_square_goodness(self) -> pd.DataFrame:
        """Chi-square goodness-of-fit test for categorical columns.

        Tests whether observed frequencies differ from expected uniform.

        Returns:
            DataFrame with test results per categorical column.
        """
        cols = self._schema.categorical_columns
        if not cols:
            return pd.DataFrame()

        rows: list[dict] = []
        for col in cols[:20]:
            vc = self._df[col].value_counts()
            if len(vc) < 2 or len(vc) > 100:
                continue

            observed = vc.values.astype(float)
            expected = np.full_like(observed, observed.mean())

            try:
                stat, p = sp_stats.chisquare(observed, f_exp=expected)
                rows.append({
                    "column": col,
                    "n_categories": len(vc),
                    "chi2_stat": round(float(stat), 4),
                    "p_value": round(float(p), 6),
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
            # Try scipy-only alternative (simplified)
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
