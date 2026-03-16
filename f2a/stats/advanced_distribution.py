"""Advanced distribution analysis module.

Provides best-fit distribution testing, power-transform recommendation,
Jarque-Bera normality test, ECDF computation, and KDE bandwidth analysis.

References:
    - Box & Cox (1964) — power transform
    - Jarque & Bera (1987) — normality test
    - Silverman (1986) — KDE bandwidth selection
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from f2a.core.schema import DataSchema
from f2a.utils.logging import get_logger

logger = get_logger(__name__)

# Candidate distributions for best-fit analysis
_CANDIDATE_DISTRIBUTIONS = [
    ("norm", sp_stats.norm),
    ("lognorm", sp_stats.lognorm),
    ("expon", sp_stats.expon),
    ("gamma", sp_stats.gamma),
    ("beta", sp_stats.beta),
    ("weibull_min", sp_stats.weibull_min),
    ("uniform", sp_stats.uniform),
]


class AdvancedDistributionStats:
    """Advanced distribution analysis for numeric columns.

    Args:
        df: Target DataFrame.
        schema: Data schema.
        n_fits: Number of candidate distributions to fit.
        max_sample: Max rows to sample for expensive operations.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        schema: DataSchema,
        n_fits: int = 7,
        max_sample: int = 5000,
    ) -> None:
        self._df = df
        self._schema = schema
        self._n_fits = min(n_fits, len(_CANDIDATE_DISTRIBUTIONS))
        self._max_sample = max_sample

    # ── Best-fit distribution ─────────────────────────────

    def best_fit(self) -> pd.DataFrame:
        """Fit candidate distributions and rank by AIC/BIC.

        For each numeric column, fits up to ``n_fits`` scipy distributions,
        computes AIC and BIC, and returns the best match.

        Returns:
            DataFrame with columns: column, best_dist, aic, bic, ks_stat, ks_p,
            params (per-column best).
        """
        cols = self._schema.numeric_columns
        if not cols:
            return pd.DataFrame()

        rows: list[dict] = []
        candidates = _CANDIDATE_DISTRIBUTIONS[: self._n_fits]

        for col in cols:
            series = self._df[col].dropna()
            if len(series) < 20:
                continue

            sample = (
                series.sample(self._max_sample, random_state=42)
                if len(series) > self._max_sample
                else series
            )
            data = sample.values

            best: dict[str, Any] | None = None

            for name, dist in candidates:
                try:
                    params = dist.fit(data)
                    # Log-likelihood
                    ll = np.sum(dist.logpdf(data, *params))
                    if not np.isfinite(ll):
                        continue
                    k = len(params)
                    n = len(data)
                    aic = 2 * k - 2 * ll
                    bic = k * np.log(n) - 2 * ll

                    ks_stat, ks_p = sp_stats.kstest(data, name, args=params)

                    entry = {
                        "dist_name": name,
                        "aic": float(aic),
                        "bic": float(bic),
                        "ks_stat": float(ks_stat),
                        "ks_p": float(ks_p),
                        "params": params,
                    }
                    if best is None or aic < best["aic"]:
                        best = entry
                except Exception:
                    continue

            if best is not None:
                rows.append({
                    "column": col,
                    "best_distribution": best["dist_name"],
                    "aic": round(best["aic"], 2),
                    "bic": round(best["bic"], 2),
                    "ks_statistic": round(best["ks_stat"], 4),
                    "ks_p_value": round(best["ks_p"], 6),
                    "fit_quality": (
                        "good" if best["ks_p"] > 0.05
                        else "moderate" if best["ks_p"] > 0.01
                        else "poor"
                    ),
                })

        return pd.DataFrame(rows).set_index("column") if rows else pd.DataFrame()

    # ── Jarque-Bera normality test ────────────────────────

    def jarque_bera(self) -> pd.DataFrame:
        """Perform Jarque-Bera test for normality on each numeric column.

        The JB test jointly tests whether skewness and kurtosis
        match a normal distribution.  H0: data is normally distributed.

        Returns:
            DataFrame with jb_stat, p_value, is_normal columns.
        """
        cols = self._schema.numeric_columns
        if not cols:
            return pd.DataFrame()

        rows: list[dict] = []
        for col in cols:
            series = self._df[col].dropna()
            if len(series) < 8:
                continue
            try:
                jb_stat, p_val = sp_stats.jarque_bera(series)
                rows.append({
                    "column": col,
                    "jb_statistic": round(float(jb_stat), 4),
                    "p_value": round(float(p_val), 6),
                    "is_normal_0.05": float(p_val) > 0.05,
                    "skewness": round(float(series.skew()), 4),
                    "kurtosis": round(float(series.kurtosis()), 4),
                })
            except Exception:
                continue

        return pd.DataFrame(rows).set_index("column") if rows else pd.DataFrame()

    # ── Power transform recommendation ────────────────────

    def power_transform_recommendation(self) -> pd.DataFrame:
        """Recommend power transformations (Box-Cox / Yeo-Johnson).

        Box-Cox requires strictly positive data; Yeo-Johnson works for any data.
        Reports the optimal lambda and post-transform skewness.

        Returns:
            DataFrame with method, lambda, original_skew, transformed_skew.
        """
        cols = self._schema.numeric_columns
        if not cols:
            return pd.DataFrame()

        rows: list[dict] = []
        for col in cols:
            series = self._df[col].dropna()
            if len(series) < 10:
                continue

            original_skew = float(series.skew())
            data = series.values

            # Try Box-Cox (positive data only)
            bc_lambda = None
            bc_skew = None
            if (data > 0).all():
                try:
                    transformed, lmbda = sp_stats.boxcox(data)
                    bc_lambda = round(float(lmbda), 4)
                    bc_skew = round(float(pd.Series(transformed).skew()), 4)
                except Exception:
                    pass

            # Yeo-Johnson (any data)
            yj_lambda = None
            yj_skew = None
            try:
                transformed, lmbda = sp_stats.yeojohnson(data)
                yj_lambda = round(float(lmbda), 4)
                yj_skew = round(float(pd.Series(transformed).skew()), 4)
            except Exception:
                pass

            # Recommendation
            if bc_skew is not None and abs(bc_skew) < (abs(yj_skew) if yj_skew is not None else float("inf")):
                recommended = "box-cox"
                rec_lambda = bc_lambda
                rec_skew = bc_skew
            elif yj_skew is not None:
                recommended = "yeo-johnson"
                rec_lambda = yj_lambda
                rec_skew = yj_skew
            else:
                recommended = "none"
                rec_lambda = None
                rec_skew = None

            needs_transform = abs(original_skew) > 0.5

            rows.append({
                "column": col,
                "original_skewness": round(original_skew, 4),
                "recommended_method": recommended,
                "optimal_lambda": rec_lambda,
                "transformed_skewness": rec_skew,
                "needs_transform": needs_transform,
                "improvement": (
                    round(abs(original_skew) - abs(rec_skew), 4)
                    if rec_skew is not None
                    else None
                ),
            })

        return pd.DataFrame(rows).set_index("column") if rows else pd.DataFrame()

    # ── ECDF data ─────────────────────────────────────────

    def ecdf(self) -> dict[str, pd.DataFrame]:
        """Compute Empirical Cumulative Distribution Function for each column.

        Returns:
            Dictionary mapping column name to DataFrame with x, ecdf columns.
        """
        cols = self._schema.numeric_columns
        result: dict[str, pd.DataFrame] = {}
        for col in cols:
            series = self._df[col].dropna().sort_values()
            if len(series) < 2:
                continue
            n = len(series)
            # Subsample for very large data
            if n > self._max_sample:
                indices = np.linspace(0, n - 1, self._max_sample, dtype=int)
                series = series.iloc[indices]
                n = len(series)
            result[col] = pd.DataFrame({
                "x": series.values,
                "ecdf": np.arange(1, n + 1) / n,
            })
        return result

    # ── KDE bandwidth analysis ────────────────────────────

    def kde_analysis(self) -> pd.DataFrame:
        """Compute optimal KDE bandwidth using Silverman's rule of thumb.

        Returns:
            DataFrame with column, silverman_bw, scotts_bw, n.
        """
        cols = self._schema.numeric_columns
        if not cols:
            return pd.DataFrame()

        rows: list[dict] = []
        for col in cols:
            series = self._df[col].dropna()
            n = len(series)
            if n < 5:
                continue

            std = float(series.std())
            iqr = float(series.quantile(0.75) - series.quantile(0.25))

            # Silverman's rule: h = 0.9 * min(std, IQR/1.34) * n^(-1/5)
            spread = min(std, iqr / 1.34) if iqr > 0 else std
            silverman = 0.9 * spread * (n ** (-0.2)) if spread > 0 else None

            # Scott's rule: h = 3.49 * std * n^(-1/3)
            scotts = 3.49 * std * (n ** (-1 / 3)) if std > 0 else None

            rows.append({
                "column": col,
                "n": n,
                "std": round(std, 4),
                "iqr": round(iqr, 4),
                "silverman_bandwidth": round(silverman, 4) if silverman else None,
                "scotts_bandwidth": round(scotts, 4) if scotts else None,
            })

        return pd.DataFrame(rows).set_index("column") if rows else pd.DataFrame()

    # ── Combined summary ──────────────────────────────────

    def summary(self) -> dict[str, Any]:
        """Return a combined advanced distribution analysis summary."""
        result: dict[str, Any] = {}

        try:
            bf = self.best_fit()
            if not bf.empty:
                result["best_fit"] = bf
        except Exception as exc:
            logger.debug("Best-fit analysis skipped: %s", exc)

        try:
            jb = self.jarque_bera()
            if not jb.empty:
                result["jarque_bera"] = jb
        except Exception as exc:
            logger.debug("Jarque-Bera test skipped: %s", exc)

        try:
            pt = self.power_transform_recommendation()
            if not pt.empty:
                result["power_transform"] = pt
        except Exception as exc:
            logger.debug("Power transform analysis skipped: %s", exc)

        try:
            kde = self.kde_analysis()
            if not kde.empty:
                result["kde_bandwidth"] = kde
        except Exception as exc:
            logger.debug("KDE analysis skipped: %s", exc)

        return result
