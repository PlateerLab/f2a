"""Data quality scoring module.

Computes per-column and overall quality scores across four dimensions:
completeness, uniqueness, consistency, and validity.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from f2a.core.schema import DataSchema


class QualityStats:
    """Compute data quality scores.

    Args:
        df: Target DataFrame.
        schema: Data schema.
    """

    def __init__(self, df: pd.DataFrame, schema: DataSchema) -> None:
        self._df = df
        self._schema = schema

    # ── Dimension scores ──────────────────────────────────

    def completeness(self) -> float:
        """Proportion of non-missing cells."""
        total = self._df.shape[0] * self._df.shape[1]
        if total == 0:
            return 1.0
        return round(1.0 - float(self._df.isna().sum().sum() / total), 4)

    def uniqueness(self) -> float:
        """Proportion of non-duplicate rows."""
        n = len(self._df)
        if n == 0:
            return 1.0
        return round(1.0 - float(self._df.duplicated().sum() / n), 4)

    def consistency(self) -> float:
        """Type-consistency score — fraction of columns with uniform types."""
        ncol = len(self._df.columns)
        if ncol == 0:
            return 1.0

        consistent = 0
        for col in self._df.columns:
            non_null = self._df[col].dropna()
            if len(non_null) == 0:
                consistent += 1
                continue
            if non_null.apply(type).nunique() <= 1:
                consistent += 1

        return round(consistent / ncol, 4)

    def validity(self) -> float:
        """Proportion of finite numeric values (excludes ``inf`` / ``-inf``)."""
        num_cols = self._schema.numeric_columns
        if not num_cols:
            return 1.0

        total = 0
        valid = 0
        for col in num_cols:
            series = self._df[col].dropna()
            total += len(series)
            valid += int(np.isfinite(series).sum())

        return round(valid / total, 4) if total > 0 else 1.0

    def overall_score(self) -> float:
        """Weighted average of all quality dimensions.

        Weights: completeness 35 %, uniqueness 25 %, consistency 20 %,
        validity 20 %.
        """
        return round(
            0.35 * self.completeness()
            + 0.25 * self.uniqueness()
            + 0.20 * self.consistency()
            + 0.20 * self.validity(),
            4,
        )

    # ── Summaries ─────────────────────────────────────────

    def summary(self) -> dict[str, Any]:
        """Return all quality dimension scores."""
        return {
            "completeness": self.completeness(),
            "uniqueness": self.uniqueness(),
            "consistency": self.consistency(),
            "validity": self.validity(),
            "overall": self.overall_score(),
        }

    def column_quality(self) -> pd.DataFrame:
        """Return per-column quality scores.

        Returns:
            DataFrame indexed by column name with completeness, uniqueness,
            type, and composite quality_score.
        """
        rows: list[dict] = []
        for col_info in self._schema.columns:
            col = col_info.name
            series = self._df[col]
            compl = 1.0 - col_info.missing_ratio

            n_total = int(series.count())
            n_unique = int(series.nunique())
            uniqueness = n_unique / n_total if n_total > 0 else 1.0

            rows.append({
                "column": col,
                "completeness": round(compl, 4),
                "uniqueness": round(min(uniqueness, 1.0), 4),
                "type": col_info.inferred_type.value,
                "quality_score": round((compl + min(uniqueness, 1.0)) / 2, 4),
            })

        return pd.DataFrame(rows).set_index("column") if rows else pd.DataFrame()
