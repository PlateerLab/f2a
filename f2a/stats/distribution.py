"""Distribution analysis module."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from f2a.core.schema import DataSchema


class DistributionStats:
    """Analyze distribution characteristics of numeric columns.

    Args:
        df: Target DataFrame to analyze.
        schema: Data schema.
    """

    def __init__(self, df: pd.DataFrame, schema: DataSchema) -> None:
        self._df = df
        self._schema = schema

    def analyze(self) -> pd.DataFrame:
        """Return distribution information for numeric columns.

        Returns:
            DataFrame containing skewness, kurtosis, and normality test results.
        """
        cols = self._schema.numeric_columns
        if not cols:
            return pd.DataFrame()

        rows: list[dict] = []
        for col in cols:
            series = self._df[col].dropna()
            if len(series) < 3:
                continue
            rows.append(self._analyze_column(col, series))

        return pd.DataFrame(rows).set_index("column") if rows else pd.DataFrame()

    def quantile_table(self, quantiles: list[float] | None = None) -> pd.DataFrame:
        """Return quantile table for numeric columns.

        Args:
            quantiles: List of quantiles to compute. Defaults to
                ``[0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]``.

        Returns:
            Quantile DataFrame.
        """
        if quantiles is None:
            quantiles = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]

        cols = self._schema.numeric_columns
        if not cols:
            return pd.DataFrame()

        return self._df[cols].quantile(quantiles)

    @staticmethod
    def _analyze_column(col: str, series: pd.Series) -> dict:
        """Analyze the distribution of a single numeric column."""
        skew = float(series.skew())
        kurt = float(series.kurtosis())

        n = len(series)

        # ── Normality tests ──────────────────────────────
        shapiro_p: float | None = None
        dagostino_p: float | None = None
        ks_p: float | None = None
        anderson_stat: float | None = None
        anderson_critical: float | None = None

        # Shapiro-Wilk (best for n <= 5000)
        if 3 <= n <= 5000:
            try:
                _, shapiro_p = sp_stats.shapiro(series)
            except Exception:
                pass

        # D'Agostino-Pearson (good for n > 20)
        if n > 20:
            try:
                _, dagostino_p = sp_stats.normaltest(series)
            except Exception:
                pass

        # Kolmogorov-Smirnov
        if n >= 5:
            try:
                mean, std = series.mean(), series.std()
                if std > 0:
                    _, ks_p = sp_stats.kstest(series, "norm", args=(mean, std))
            except Exception:
                pass

        # Anderson-Darling
        if n >= 8:
            try:
                ad = sp_stats.anderson(series, "norm")
                anderson_stat = float(ad.statistic)
                # Use the 5% significance level critical value
                anderson_critical = float(ad.critical_values[2])  # index 2 = 5%
            except Exception:
                pass

        # Primary normality verdict (prefer Shapiro for small, D'Agostino for large)
        primary_p: float | None = None
        primary_test: str = "n/a"
        if shapiro_p is not None:
            primary_p = shapiro_p
            primary_test = "shapiro"
        elif dagostino_p is not None:
            primary_p = dagostino_p
            primary_test = "dagostino"

        # Skewness interpretation
        if abs(skew) < 0.5:
            skew_label = "symmetric"
        elif abs(skew) < 1.0:
            skew_label = "moderate skew"
        else:
            skew_label = "high skew"

        # Kurtosis interpretation (excess kurtosis: 0 = normal)
        if abs(kurt) < 0.5:
            kurt_label = "mesokurtic"
        elif kurt > 0:
            kurt_label = "leptokurtic"
        else:
            kurt_label = "platykurtic"

        return {
            "column": col,
            "n": n,
            "skewness": round(skew, 4),
            "skew_type": skew_label,
            "kurtosis": round(kurt, 4),
            "kurt_type": kurt_label,
            "normality_test": primary_test,
            "normality_p": round(primary_p, 6) if primary_p is not None else None,
            "is_normal_0.05": primary_p > 0.05 if primary_p is not None else None,
            "shapiro_p": round(shapiro_p, 6) if shapiro_p is not None else None,
            "dagostino_p": round(dagostino_p, 6) if dagostino_p is not None else None,
            "ks_p": round(ks_p, 6) if ks_p is not None else None,
            "anderson_stat": round(anderson_stat, 4) if anderson_stat is not None else None,
            "anderson_5pct_cv": round(anderson_critical, 4) if anderson_critical is not None else None,
        }
