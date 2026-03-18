"""Feature importance analysis module.

Ranks features by variance, correlation, and mutual information.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from f2a.core.schema import DataSchema
from f2a.utils.logging import get_logger

logger = get_logger(__name__)


class FeatureImportanceStats:
    """Compute feature importance rankings for numeric columns.

    Args:
        df: Target DataFrame.
        schema: Data schema.
    """

    def __init__(self, df: pd.DataFrame, schema: DataSchema) -> None:
        self._df = df
        self._schema = schema

    # ── Variance-based ranking ────────────────────────────

    def variance_ranking(self) -> pd.DataFrame:
        """Rank numeric features by normalised variance (coefficient of variation).

        Returns:
            DataFrame sorted by variance (descending).
        """
        cols = self._schema.numeric_columns
        if not cols:
            return pd.DataFrame()

        rows: list[dict] = []
        for col in cols:
            series = self._df[col].dropna()
            if len(series) < 2:
                continue
            mean = float(series.mean())
            std = float(series.std())
            cv = abs(std / mean) if mean != 0 else None
            rows.append({
                "column": col,
                "variance": round(float(series.var()), 4),
                "std": round(std, 4),
                "cv": round(cv, 4) if cv is not None else None,
                "range": round(float(series.max() - series.min()), 4),
            })

        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows).sort_values("variance", ascending=False).set_index("column")

    # ── Correlation-with-all ranking ──────────────────────

    def mean_abs_correlation(self) -> pd.DataFrame:
        """Rank features by mean absolute correlation with all other features.

        Columns with higher mean |r| are more *connected* to the rest of the
        dataset and may be more informative (or redundant).
        """
        cols = self._schema.numeric_columns
        if len(cols) < 2:
            return pd.DataFrame()

        corr = self._df[cols].corr(method="pearson").abs()
        # Exclude self-correlation
        np.fill_diagonal(corr.values, 0)
        mean_corr = corr.mean()

        df = pd.DataFrame({
            "column": mean_corr.index,
            "mean_abs_corr": mean_corr.values.round(4),
        }).sort_values("mean_abs_corr", ascending=False).set_index("column")

        return df

    # ── Mutual information ────────────────────────────────

    def mutual_information(self) -> pd.DataFrame:
        """Compute average mutual-information score per numeric feature.

        Requires ``scikit-learn``.  Returns an empty DataFrame if unavailable.
        """
        cols = self._schema.numeric_columns
        if len(cols) < 2:
            return pd.DataFrame()

        try:
            from sklearn.feature_selection import mutual_info_regression
        except ImportError:
            logger.info("scikit-learn not installed; skipping mutual-information analysis.")
            return pd.DataFrame()

        cols = cols[:15]  # limit to avoid expensive computation
        df_clean = self._df[cols].dropna()
        if len(df_clean) < 30:
            return pd.DataFrame()

        # For each column, compute MI against all others, then average
        mi_scores: dict[str, float] = {col: 0.0 for col in cols}
        n_pairs = 0
        for col in cols:
            X = df_clean.drop(columns=[col])
            y = df_clean[col]
            try:
                mi = mutual_info_regression(X, y, random_state=42, n_neighbors=5)
                for other_col, mi_val in zip(X.columns, mi):
                    mi_scores[other_col] += float(mi_val)
                    mi_scores[col] += float(mi_val)
                n_pairs += len(X.columns)
            except Exception:
                continue

        if n_pairs == 0:
            return pd.DataFrame()

        # Average
        for col in mi_scores:
            mi_scores[col] /= max(1, len(cols) - 1)

        df_result = pd.DataFrame({
            "column": list(mi_scores.keys()),
            "avg_mutual_info": [round(v, 4) for v in mi_scores.values()],
        }).sort_values("avg_mutual_info", ascending=False).set_index("column")

        return df_result

    # ── Combined summary ──────────────────────────────────

    def summary(self) -> pd.DataFrame:
        """Return a combined feature-importance summary (variance-based)."""
        return self.variance_ranking()
