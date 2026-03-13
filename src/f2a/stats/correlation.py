"""Correlation analysis module."""

from __future__ import annotations

import numpy as np
import pandas as pd

from f2a.core.schema import DataSchema
from f2a.utils.logging import get_logger

logger = get_logger(__name__)


class CorrelationStats:
    """Analyze correlations between columns.

    Args:
        df: Target DataFrame to analyze.
        schema: Data schema.
    """

    def __init__(self, df: pd.DataFrame, schema: DataSchema) -> None:
        self._df = df
        self._schema = schema

    def pearson(self) -> pd.DataFrame:
        """Return the Pearson correlation matrix."""
        cols = self._schema.numeric_columns
        if len(cols) < 2:
            return pd.DataFrame()
        return self._df[cols].corr(method="pearson")

    def spearman(self) -> pd.DataFrame:
        """Return the Spearman rank correlation matrix."""
        cols = self._schema.numeric_columns
        if len(cols) < 2:
            return pd.DataFrame()
        return self._df[cols].corr(method="spearman")

    def kendall(self) -> pd.DataFrame:
        """Return the Kendall tau correlation matrix."""
        cols = self._schema.numeric_columns
        if len(cols) < 2:
            return pd.DataFrame()
        # Kendall is expensive — limit columns
        cols = cols[:15]
        return self._df[cols].corr(method="kendall")

    def cramers_v_matrix(self) -> pd.DataFrame:
        """Return the Cramer's V matrix for categorical columns."""
        cols = self._schema.categorical_columns
        if len(cols) < 2:
            return pd.DataFrame()

        cols = cols[:15]
        n = len(cols)
        matrix = pd.DataFrame(np.ones((n, n)), index=cols, columns=cols)

        for i in range(n):
            for j in range(i + 1, n):
                v = self._cramers_v(self._df[cols[i]], self._df[cols[j]])
                matrix.iloc[i, j] = v
                matrix.iloc[j, i] = v

        return matrix

    def vif(self) -> pd.DataFrame:
        """Compute Variance Inflation Factor for numeric columns.

        VIF > 5 suggests moderate multicollinearity;
        VIF > 10 suggests severe multicollinearity.

        Uses the inverse-correlation-matrix diagonal method.
        """
        cols = self._schema.numeric_columns
        if len(cols) < 2:
            return pd.DataFrame()

        df_clean = self._df[cols].dropna()
        if len(df_clean) < len(cols) + 1:
            return pd.DataFrame()

        corr = df_clean.corr()
        try:
            corr_inv = np.linalg.inv(corr.values)
            vif_values = np.diag(corr_inv)
        except np.linalg.LinAlgError:
            logger.warning("Singular correlation matrix; VIF cannot be computed.")
            return pd.DataFrame()

        rows: list[dict] = []
        for col, vif_val in zip(cols, vif_values):
            severity = (
                "severe" if vif_val > 10
                else "moderate" if vif_val > 5
                else "low"
            )
            rows.append({
                "column": col,
                "VIF": round(float(vif_val), 2),
                "multicollinearity": severity,
            })

        return (
            pd.DataFrame(rows)
            .set_index("column")
            .sort_values("VIF", ascending=False)
        )

    def high_correlations(self, threshold: float = 0.9) -> list[tuple[str, str, float]]:
        """Return pairs with high correlation.

        Args:
            threshold: Absolute correlation coefficient threshold.

        Returns:
            List of ``(col_a, col_b, correlation)`` tuples.
        """
        corr = self.pearson()
        if corr.empty:
            return []

        pairs: list[tuple[str, str, float]] = []
        cols = corr.columns
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                val = corr.iloc[i, j]
                if abs(val) >= threshold:
                    pairs.append((cols[i], cols[j], round(float(val), 4)))

        if pairs:
            logger.warning(
                "Multicollinearity warning: %d column pairs have |r| >= %.2f.",
                len(pairs),
                threshold,
            )

        return pairs

    # ── Internal helpers ────────────────────────────────

    @staticmethod
    def _cramers_v(x: pd.Series, y: pd.Series) -> float:
        """Compute Cramer's V between two categorical variables."""
        confusion = pd.crosstab(x, y)
        n = confusion.sum().sum()
        if n == 0:
            return 0.0

        from scipy.stats import chi2_contingency

        chi2, _, _, _ = chi2_contingency(confusion)
        min_dim = min(confusion.shape) - 1
        if min_dim == 0:
            return 0.0

        return float(np.sqrt(chi2 / (n * min_dim)))
