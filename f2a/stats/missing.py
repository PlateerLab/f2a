"""Missing data analysis module."""

from __future__ import annotations

import pandas as pd

from f2a.core.schema import DataSchema


class MissingStats:
    """Analyze missing data patterns.

    Args:
        df: Target DataFrame to analyze.
        schema: Data schema.
    """

    def __init__(self, df: pd.DataFrame, schema: DataSchema) -> None:
        self._df = df
        self._schema = schema

    def column_summary(self) -> pd.DataFrame:
        """Return per-column missing data summary.

        Returns:
            DataFrame with missing count, ratio, and dtype per column.
        """
        rows: list[dict] = []
        for col_info in self._schema.columns:
            rows.append(
                {
                    "column": col_info.name,
                    "missing_count": col_info.n_missing,
                    "missing_ratio": col_info.missing_ratio,
                    "missing_%": round(col_info.missing_ratio * 100, 2),
                    "dtype": col_info.dtype,
                }
            )

        result = pd.DataFrame(rows).set_index("column")
        return result.sort_values("missing_count", ascending=False)

    def row_missing_distribution(self) -> pd.DataFrame:
        """Return per-row missing count distribution.

        Returns:
            Frequency table of missing counts per row.
        """
        row_missing = self._df.isna().sum(axis=1)
        dist = row_missing.value_counts().sort_index()
        return pd.DataFrame(
            {
                "missing_per_row": dist.index,
                "row_count": dist.values,
                "row_%": (dist.values / len(self._df) * 100).round(2),
            }
        )

    def missing_matrix(self) -> pd.DataFrame:
        """Return missing data matrix (boolean).

        Boolean matrix used for visualizing missing data patterns.

        Returns:
            Boolean DataFrame where True indicates missing.
        """
        return self._df.isna()

    def total_missing_ratio(self) -> float:
        """Return the overall missing data ratio."""
        total_cells = self._df.shape[0] * self._df.shape[1]
        if total_cells == 0:
            return 0.0
        return round(float(self._df.isna().sum().sum() / total_cells), 4)
