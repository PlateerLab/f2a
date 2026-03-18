"""Descriptive statistics analysis module."""

from __future__ import annotations

import numpy as np
import pandas as pd

from f2a.core.schema import DataSchema
from f2a.utils.type_inference import ColumnType


class DescriptiveStats:
    """Compute descriptive statistics.

    Args:
        df: Target DataFrame to analyze.
        schema: Data schema.
    """

    def __init__(self, df: pd.DataFrame, schema: DataSchema) -> None:
        self._df = df
        self._schema = schema

    def summary(self) -> pd.DataFrame:
        """Return overall summary statistics.

        Generates a unified summary table covering both numeric and categorical columns.

        Returns:
            Summary statistics DataFrame.
        """
        rows: list[dict] = []
        for col_info in self._schema.columns:
            series = self._df[col_info.name]
            row: dict = {
                "column": col_info.name,
                "type": col_info.inferred_type.value,
                "count": int(series.count()),
                "missing": col_info.n_missing,
                "missing_%": round(col_info.missing_ratio * 100, 2),
                "unique": col_info.n_unique,
            }

            if col_info.inferred_type == ColumnType.NUMERIC:
                row.update(self._numeric_stats(series))
            elif col_info.inferred_type in (ColumnType.CATEGORICAL, ColumnType.BOOLEAN):
                row.update(self._categorical_stats(series))

            rows.append(row)

        return pd.DataFrame(rows).set_index("column")

    def numeric_summary(self) -> pd.DataFrame:
        """Return summary statistics for numeric columns only."""
        cols = self._schema.numeric_columns
        if not cols:
            return pd.DataFrame()
        return self._df[cols].describe().T

    def categorical_summary(self) -> pd.DataFrame:
        """Return summary statistics for categorical columns only."""
        cols = self._schema.categorical_columns
        if not cols:
            return pd.DataFrame()

        rows: list[dict] = []
        for col in cols:
            series = self._df[col]
            top_val = series.mode().iloc[0] if not series.mode().empty else None
            rows.append(
                {
                    "column": col,
                    "count": int(series.count()),
                    "unique": int(series.nunique()),
                    "top": top_val,
                    "freq": int(series.value_counts().iloc[0]) if top_val is not None else 0,
                }
            )
        return pd.DataFrame(rows).set_index("column")

    # ── Internal helpers ────────────────────────────────

    @staticmethod
    def _numeric_stats(series: pd.Series) -> dict:
        """Return numeric column statistics as a dictionary."""
        desc = series.describe()
        q1 = float(desc.get("25%", np.nan))
        q3 = float(desc.get("75%", np.nan))
        mean = float(series.mean())
        std = float(series.std())
        count = int(series.count())
        skew_val = float(series.skew()) if count >= 3 else np.nan
        kurt_val = float(series.kurtosis()) if count >= 4 else np.nan
        se = std / np.sqrt(count) if count > 0 else np.nan
        cv = abs(std / mean) if mean != 0 else np.nan
        mad = float((series - series.median()).abs().median())

        return {
            "mean": round(mean, 4),
            "median": round(float(series.median()), 4),
            "std": round(std, 4),
            "se": round(float(se), 4),
            "cv": round(float(cv), 4) if not np.isnan(cv) else None,
            "mad": round(mad, 4),
            "min": float(series.min()),
            "max": float(series.max()),
            "range": round(float(series.max() - series.min()), 4),
            "p5": round(float(series.quantile(0.05)), 4),
            "q1": round(q1, 4),
            "q3": round(q3, 4),
            "p95": round(float(series.quantile(0.95)), 4),
            "iqr": round(q3 - q1, 4),
            "skewness": round(skew_val, 4) if not np.isnan(skew_val) else None,
            "kurtosis": round(kurt_val, 4) if not np.isnan(kurt_val) else None,
        }

    @staticmethod
    def _categorical_stats(series: pd.Series) -> dict:
        """Return categorical column statistics as a dictionary."""
        vc = series.value_counts()
        top_val = vc.index[0] if len(vc) > 0 else None
        return {
            "top": top_val,
            "freq": int(vc.iloc[0]) if len(vc) > 0 else 0,
        }
