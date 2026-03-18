"""Duplicate detection module."""

from __future__ import annotations

from typing import Any

import pandas as pd

from f2a.core.schema import DataSchema


class DuplicateStats:
    """Detect and analyse duplicate rows and column uniqueness.

    Args:
        df: Target DataFrame.
        schema: Data schema.
    """

    def __init__(self, df: pd.DataFrame, schema: DataSchema) -> None:
        self._df = df
        self._schema = schema

    def exact_duplicates(self) -> dict[str, Any]:
        """Count exact duplicate rows.

        Returns:
            Dictionary with ``total_rows``, ``duplicate_rows``,
            ``unique_rows``, ``duplicate_ratio``.
        """
        n = len(self._df)
        n_dup = int(self._df.duplicated().sum())
        return {
            "total_rows": n,
            "duplicate_rows": n_dup,
            "unique_rows": n - n_dup,
            "duplicate_ratio": round(n_dup / n, 4) if n > 0 else 0.0,
        }

    def column_uniqueness(self) -> pd.DataFrame:
        """Return per-column uniqueness statistics.

        Returns:
            DataFrame indexed by column name with uniqueness metrics.
        """
        rows: list[dict] = []
        for col in self._df.columns:
            n_unique = int(self._df[col].nunique())
            n_total = int(self._df[col].count())
            rows.append({
                "column": col,
                "unique_values": n_unique,
                "total_non_null": n_total,
                "uniqueness_ratio": round(n_unique / n_total, 4) if n_total > 0 else 0.0,
                "is_unique_key": n_unique == n_total > 0,
            })
        return pd.DataFrame(rows).set_index("column") if rows else pd.DataFrame()

    def summary(self) -> dict[str, Any]:
        """Return concise duplicate summary."""
        return self.exact_duplicates()
