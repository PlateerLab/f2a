"""Outlier detection module.

Provides IQR-based and Z-score-based outlier detection for numeric columns.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from f2a.core.schema import DataSchema


class OutlierStats:
    """Detect and summarise outliers in numeric columns.

    Args:
        df: Target DataFrame.
        schema: Data schema.
    """

    def __init__(self, df: pd.DataFrame, schema: DataSchema) -> None:
        self._df = df
        self._schema = schema

    # ── IQR method ────────────────────────────────────────

    def iqr_summary(self, multiplier: float = 1.5) -> pd.DataFrame:
        """Detect outliers using the IQR fence method.

        Args:
            multiplier: IQR multiplier (default 1.5 for moderate outliers,
                3.0 for extreme outliers).

        Returns:
            Per-column outlier summary DataFrame.
        """
        cols = self._schema.numeric_columns
        if not cols:
            return pd.DataFrame()

        rows: list[dict] = []
        for col in cols:
            series = self._df[col].dropna()
            if len(series) == 0:
                continue

            q1 = float(series.quantile(0.25))
            q3 = float(series.quantile(0.75))
            iqr = q3 - q1
            lower = q1 - multiplier * iqr
            upper = q3 + multiplier * iqr

            outlier_mask = (series < lower) | (series > upper)
            outliers = series[outlier_mask]
            n_outliers = len(outliers)

            rows.append({
                "column": col,
                "q1": round(q1, 4),
                "q3": round(q3, 4),
                "iqr": round(iqr, 4),
                "lower_bound": round(lower, 4),
                "upper_bound": round(upper, 4),
                "outlier_count": n_outliers,
                "outlier_%": round(n_outliers / len(series) * 100, 2),
                "min_outlier": round(float(outliers.min()), 4) if n_outliers > 0 else None,
                "max_outlier": round(float(outliers.max()), 4) if n_outliers > 0 else None,
            })

        return pd.DataFrame(rows).set_index("column") if rows else pd.DataFrame()

    # ── Z-score method ────────────────────────────────────

    def zscore_summary(self, threshold: float = 3.0) -> pd.DataFrame:
        """Detect outliers using the Z-score method.

        Args:
            threshold: Z-score absolute threshold (default 3.0).

        Returns:
            Per-column outlier summary DataFrame.
        """
        cols = self._schema.numeric_columns
        if not cols:
            return pd.DataFrame()

        rows: list[dict] = []
        for col in cols:
            series = self._df[col].dropna()
            if len(series) < 3:
                continue

            mean = float(series.mean())
            std = float(series.std())
            if std == 0:
                continue

            z = np.abs((series - mean) / std)
            n_outliers = int((z > threshold).sum())

            rows.append({
                "column": col,
                "mean": round(mean, 4),
                "std": round(std, 4),
                "threshold": threshold,
                "outlier_count": n_outliers,
                "outlier_%": round(n_outliers / len(series) * 100, 2),
                "max_zscore": round(float(z.max()), 4),
            })

        return pd.DataFrame(rows).set_index("column") if rows else pd.DataFrame()

    # ── Convenience ───────────────────────────────────────

    def summary(self, method: str = "iqr", **kwargs) -> pd.DataFrame:
        """Return outlier summary using the specified *method*.

        Args:
            method: ``"iqr"`` (default) or ``"zscore"``.
            **kwargs: Passed to the underlying method.
        """
        if method == "zscore":
            return self.zscore_summary(**kwargs)
        return self.iqr_summary(**kwargs)

    def outlier_mask(self, method: str = "iqr", **kwargs) -> pd.DataFrame:
        """Return a boolean DataFrame where ``True`` marks an outlier.

        Useful for downstream visualisation.
        """
        cols = self._schema.numeric_columns
        if not cols:
            return pd.DataFrame()

        mask = pd.DataFrame(False, index=self._df.index, columns=cols)

        if method == "zscore":
            threshold = kwargs.get("threshold", 3.0)
            for col in cols:
                series = self._df[col].dropna()
                if len(series) < 3 or series.std() == 0:
                    continue
                z = np.abs((series - series.mean()) / series.std())
                mask.loc[z.index, col] = z > threshold
        else:
            multiplier = kwargs.get("multiplier", 1.5)
            for col in cols:
                series = self._df[col].dropna()
                if len(series) == 0:
                    continue
                q1 = series.quantile(0.25)
                q3 = series.quantile(0.75)
                iqr = q3 - q1
                mask.loc[series.index, col] = (series < q1 - multiplier * iqr) | (
                    series > q3 + multiplier * iqr
                )

        return mask
