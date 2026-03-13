"""Categorical data analysis module.

Computes entropy, chi-square independence tests, and frequency analytics
for categorical columns.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

from f2a.core.schema import DataSchema
from f2a.utils.logging import get_logger

logger = get_logger(__name__)


class CategoricalStats:
    """Analyse categorical columns in depth.

    Args:
        df: Target DataFrame.
        schema: Data schema.
    """

    def __init__(self, df: pd.DataFrame, schema: DataSchema) -> None:
        self._df = df
        self._schema = schema

    # ── Frequency ─────────────────────────────────────────

    def frequency_table(self, column: str, top_n: int = 20) -> pd.DataFrame:
        """Return a frequency table for a single column.

        Args:
            column: Column name.
            top_n: Max categories to show.
        """
        series = self._df[column]
        vc = series.value_counts()
        total = int(series.count())

        df = pd.DataFrame({
            "value": vc.index[:top_n],
            "count": vc.values[:top_n],
            "percentage": (vc.values[:top_n] / total * 100).round(2) if total > 0 else 0,
        })
        if len(vc) > top_n:
            other_count = int(vc.values[top_n:].sum())
            other_row = pd.DataFrame([{
                "value": f"(other {len(vc) - top_n} categories)",
                "count": other_count,
                "percentage": round(other_count / total * 100, 2) if total > 0 else 0,
            }])
            df = pd.concat([df, other_row], ignore_index=True)
        return df

    # ── Entropy ───────────────────────────────────────────

    def entropy_summary(self) -> pd.DataFrame:
        """Compute Shannon entropy for each categorical column."""
        cols = self._schema.categorical_columns
        if not cols:
            return pd.DataFrame()

        rows: list[dict] = []
        for col in cols:
            vc = self._df[col].value_counts(normalize=True)
            entropy = float(-np.sum(vc * np.log2(vc + 1e-15)))
            max_entropy = float(np.log2(len(vc))) if len(vc) > 1 else 0.0
            rows.append({
                "column": col,
                "unique_values": int(self._df[col].nunique()),
                "entropy": round(entropy, 4),
                "max_entropy": round(max_entropy, 4),
                "normalized_entropy": round(entropy / max_entropy, 4) if max_entropy > 0 else 0.0,
            })

        return pd.DataFrame(rows).set_index("column") if rows else pd.DataFrame()

    # ── Chi-square independence ───────────────────────────

    def chi_square_matrix(self) -> pd.DataFrame:
        """Chi-square independence test p-values between categorical pairs.

        Returns:
            Square DataFrame of p-values.  A low p-value (<0.05) signals
            a statistically significant association between two columns.
        """
        cols = self._schema.categorical_columns
        if len(cols) < 2:
            return pd.DataFrame()

        # Limit to prevent combinatorial explosion
        cols = cols[:15]
        n = len(cols)
        matrix = pd.DataFrame(np.ones((n, n)), index=cols, columns=cols)

        for i in range(n):
            for j in range(i + 1, n):
                try:
                    ct = pd.crosstab(self._df[cols[i]], self._df[cols[j]])
                    if ct.size > 0 and ct.sum().sum() > 0:
                        _, p, _, _ = chi2_contingency(ct)
                        matrix.iloc[i, j] = round(p, 6)
                        matrix.iloc[j, i] = round(p, 6)
                except Exception:
                    matrix.iloc[i, j] = np.nan
                    matrix.iloc[j, i] = np.nan

        return matrix

    # ── Combined summary ──────────────────────────────────

    def summary(self) -> pd.DataFrame:
        """Return a combined categorical analysis summary table."""
        cols = self._schema.categorical_columns
        if not cols:
            return pd.DataFrame()

        rows: list[dict] = []
        for col in cols:
            series = self._df[col]
            vc = series.value_counts()
            top = vc.index[0] if len(vc) > 0 else None
            top_freq = int(vc.iloc[0]) if len(vc) > 0 else 0
            count = int(series.count())

            # Entropy
            vc_norm = series.value_counts(normalize=True)
            entropy = float(-np.sum(vc_norm * np.log2(vc_norm + 1e-15)))
            max_entropy = float(np.log2(len(vc_norm))) if len(vc_norm) > 1 else 0.0

            rows.append({
                "column": col,
                "count": count,
                "unique": int(series.nunique()),
                "top_value": str(top)[:50] if top is not None else None,
                "top_frequency": top_freq,
                "top_%": round(top_freq / count * 100, 2) if count > 0 else 0.0,
                "entropy": round(entropy, 4),
                "norm_entropy": round(entropy / max_entropy, 4) if max_entropy > 0 else 0.0,
            })

        return pd.DataFrame(rows).set_index("column") if rows else pd.DataFrame()
