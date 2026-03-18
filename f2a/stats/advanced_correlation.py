"""Advanced correlation analysis module.

Provides partial correlation matrix, mutual information matrix,
bootstrap correlation confidence intervals, and correlation network data.

References:
    - Székely et al. (2007) — distance correlation
    - Reshef et al. (2011) — mutual information concepts
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from f2a.core.schema import DataSchema
from f2a.utils.logging import get_logger

logger = get_logger(__name__)


class AdvancedCorrelationStats:
    """Advanced correlation analysis for numeric columns.

    Args:
        df: Target DataFrame.
        schema: Data schema.
        bootstrap_iterations: Number of bootstrap resamples for CI.
        max_sample: Max rows to sample for expensive operations.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        schema: DataSchema,
        bootstrap_iterations: int = 1000,
        max_sample: int = 5000,
    ) -> None:
        self._df = df
        self._schema = schema
        self._bootstrap_n = bootstrap_iterations
        self._max_sample = max_sample

    # ── Partial correlation ───────────────────────────────

    def partial_correlation_matrix(self) -> pd.DataFrame:
        """Compute the partial correlation matrix.

        Partial correlation measures the linear relationship between two
        variables after removing the effect of all other variables.
        Computed via the inverse of the correlation matrix.

        Returns:
            Square DataFrame of partial correlations.
        """
        cols = self._schema.numeric_columns
        if len(cols) < 3:
            return pd.DataFrame()

        cols = cols[:30]  # limit
        df_clean = self._df[cols].dropna()
        if len(df_clean) < len(cols) + 2:
            return pd.DataFrame()

        corr = df_clean.corr()
        if corr.isna().any().any():
            logger.warning("NaN in correlation matrix (zero-variance columns); skipping partial correlation.")
            return pd.DataFrame()
        try:
            precision = np.linalg.inv(corr.values)
        except np.linalg.LinAlgError:
            logger.warning("Singular correlation matrix; partial correlation unavailable.")
            return pd.DataFrame()

        # Partial corr: -P_ij / sqrt(P_ii * P_jj)
        d = np.sqrt(np.abs(np.diag(precision)))  # abs to handle numerical noise
        d[d == 0] = 1e-15  # avoid division by zero
        partial = -precision / np.outer(d, d)
        np.fill_diagonal(partial, 1.0)

        return pd.DataFrame(
            np.round(partial, 4),
            index=cols,
            columns=cols,
        )

    # ── Mutual information matrix ─────────────────────────

    def mutual_information_matrix(self) -> pd.DataFrame:
        """Compute pairwise mutual information between numeric columns.

        Uses sklearn's ``mutual_info_regression`` to estimate MI for
        each pair of columns.

        Returns:
            Square DataFrame of MI values.
        """
        cols = self._schema.numeric_columns
        if len(cols) < 2:
            return pd.DataFrame()

        try:
            from sklearn.feature_selection import mutual_info_regression
        except ImportError:
            logger.info("scikit-learn not available for MI computation.")
            return pd.DataFrame()

        cols = cols[:30]  # limit
        df_clean = self._df[cols].dropna()
        if len(df_clean) < 30:
            return pd.DataFrame()

        # Sample for speed
        if len(df_clean) > self._max_sample:
            df_clean = df_clean.sample(self._max_sample, random_state=42)

        n = len(cols)
        mi_matrix = np.zeros((n, n))

        for i, col in enumerate(cols):
            X = df_clean.drop(columns=[col]).values
            y = df_clean[col].values
            try:
                mi = mutual_info_regression(X, y, random_state=42, n_neighbors=5)
                other_cols = [c for c in cols if c != col]
                for j, other in enumerate(other_cols):
                    idx = cols.index(other)
                    mi_matrix[i, idx] = float(mi[j])
            except Exception:
                continue

        # Symmetrize
        mi_matrix = (mi_matrix + mi_matrix.T) / 2
        np.fill_diagonal(mi_matrix, 0.0)

        return pd.DataFrame(
            np.round(mi_matrix, 4),
            index=cols,
            columns=cols,
        )

    # ── Bootstrap correlation CI ──────────────────────────

    def bootstrap_correlation_ci(
        self,
        alpha: float = 0.05,
    ) -> pd.DataFrame:
        """Compute bootstrap confidence intervals for Pearson correlations.

        For each column pair, resamples ``bootstrap_iterations`` times
        and reports the ``alpha/2`` and ``1 - alpha/2`` percentile bounds.

        Args:
            alpha: Significance level (default 0.05 → 95% CI).

        Returns:
            DataFrame with col_a, col_b, r, ci_lower, ci_upper, ci_width.
        """
        cols = self._schema.numeric_columns
        if len(cols) < 2:
            return pd.DataFrame()

        cols = cols[:15]  # limit pairs
        df_clean = self._df[cols].dropna()
        n = len(df_clean)
        if n < 20:
            return pd.DataFrame()

        # Sample for speed
        if n > self._max_sample:
            df_clean = df_clean.sample(self._max_sample, random_state=42)
            n = len(df_clean)

        rng = np.random.default_rng(42)
        rows: list[dict] = []

        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                x = df_clean[cols[i]].values
                y = df_clean[cols[j]].values

                # Point estimate
                r_point = float(np.corrcoef(x, y)[0, 1])

                # Bootstrap
                boot_corrs = np.empty(self._bootstrap_n)
                for b in range(self._bootstrap_n):
                    idx = rng.integers(0, n, size=n)
                    bx, by = x[idx], y[idx]
                    std_x, std_y = bx.std(), by.std()
                    if std_x == 0 or std_y == 0:
                        boot_corrs[b] = 0.0
                    else:
                        boot_corrs[b] = float(np.corrcoef(bx, by)[0, 1])

                lower = float(np.percentile(boot_corrs, 100 * alpha / 2))
                upper = float(np.percentile(boot_corrs, 100 * (1 - alpha / 2)))

                rows.append({
                    "col_a": cols[i],
                    "col_b": cols[j],
                    "pearson_r": round(r_point, 4),
                    "ci_lower": round(lower, 4),
                    "ci_upper": round(upper, 4),
                    "ci_width": round(upper - lower, 4),
                    "significant": not (lower <= 0 <= upper),
                })

        return pd.DataFrame(rows) if rows else pd.DataFrame()

    # ── Correlation network data ──────────────────────────

    def correlation_network(self, threshold: float = 0.5) -> dict[str, Any]:
        """Build correlation network data for visualization.

        Nodes are columns; edges exist where |r| >= threshold.

        Args:
            threshold: Minimum absolute correlation for an edge.

        Returns:
            Dictionary with ``nodes`` (list of names) and ``edges``
            (list of {source, target, weight} dicts).
        """
        cols = self._schema.numeric_columns
        if len(cols) < 2:
            return {"nodes": [], "edges": []}

        cols = cols[:30]
        corr = self._df[cols].dropna().corr()

        edges: list[dict[str, Any]] = []
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                r = float(corr.iloc[i, j])
                if abs(r) >= threshold:
                    edges.append({
                        "source": cols[i],
                        "target": cols[j],
                        "weight": round(r, 4),
                        "abs_weight": round(abs(r), 4),
                    })

        # Only include nodes that have at least one edge
        connected = set()
        for e in edges:
            connected.add(e["source"])
            connected.add(e["target"])

        return {
            "nodes": sorted(connected),
            "edges": edges,
            "threshold": threshold,
            "n_edges": len(edges),
        }

    # ── Distance correlation ──────────────────────────────

    def distance_correlation_matrix(self) -> pd.DataFrame:
        """Compute pairwise distance correlations (Székely et al., 2007).

        Distance correlation can detect non-linear dependencies
        that Pearson correlation misses.

        Returns:
            Square DataFrame of distance correlations.
        """
        cols = self._schema.numeric_columns
        if len(cols) < 2:
            return pd.DataFrame()

        cols = cols[:15]  # expensive O(n^2) per pair
        df_clean = self._df[cols].dropna()
        if len(df_clean) < 10:
            return pd.DataFrame()

        # Sample for speed
        if len(df_clean) > min(self._max_sample, 2000):
            df_clean = df_clean.sample(min(self._max_sample, 2000), random_state=42)

        n = len(cols)
        matrix = np.eye(n)

        for i in range(n):
            for j in range(i + 1, n):
                dc = self._dcor(df_clean[cols[i]].values, df_clean[cols[j]].values)
                matrix[i, j] = dc
                matrix[j, i] = dc

        return pd.DataFrame(
            np.round(matrix, 4),
            index=cols,
            columns=cols,
        )

    @staticmethod
    def _dcor(x: np.ndarray, y: np.ndarray) -> float:
        """Compute distance correlation between two 1-D arrays."""
        n = len(x)
        if n < 4:
            return 0.0

        a = np.abs(x[:, None] - x[None, :])
        b = np.abs(y[:, None] - y[None, :])

        # Double centering
        a_row = a.mean(axis=1, keepdims=True)
        a_col = a.mean(axis=0, keepdims=True)
        a_grand = a.mean()
        A = a - a_row - a_col + a_grand

        b_row = b.mean(axis=1, keepdims=True)
        b_col = b.mean(axis=0, keepdims=True)
        b_grand = b.mean()
        B = b - b_row - b_col + b_grand

        dcov2 = (A * B).mean()
        dvar_x = (A * A).mean()
        dvar_y = (B * B).mean()

        if dvar_x <= 0 or dvar_y <= 0:
            return 0.0

        return float(np.sqrt(max(dcov2, 0) / np.sqrt(dvar_x * dvar_y)))

    # ── Summary ───────────────────────────────────────────

    def summary(self) -> dict[str, Any]:
        """Return combined advanced correlation results."""
        result: dict[str, Any] = {}

        try:
            pcm = self.partial_correlation_matrix()
            if not pcm.empty:
                result["partial_correlation"] = pcm
        except Exception as exc:
            logger.debug("Partial correlation skipped: %s", exc)

        try:
            mi = self.mutual_information_matrix()
            if not mi.empty:
                result["mutual_information"] = mi
        except Exception as exc:
            logger.debug("MI matrix skipped: %s", exc)

        try:
            bci = self.bootstrap_correlation_ci()
            if not bci.empty:
                result["bootstrap_ci"] = bci
        except Exception as exc:
            logger.debug("Bootstrap CI skipped: %s", exc)

        try:
            net = self.correlation_network()
            if net.get("edges"):
                result["network"] = net
        except Exception as exc:
            logger.debug("Correlation network skipped: %s", exc)

        try:
            dc = self.distance_correlation_matrix()
            if not dc.empty:
                result["distance_correlation"] = dc
        except Exception as exc:
            logger.debug("Distance correlation skipped: %s", exc)

        return result
