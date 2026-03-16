"""PCA (Principal Component Analysis) module.

Computes variance explained, loadings, and transformed coordinates
for numeric columns.  Requires ``scikit-learn``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from f2a.core.schema import DataSchema
from f2a.utils.logging import get_logger

logger = get_logger(__name__)


class PCAStats:
    """Perform PCA on numeric columns.

    Args:
        df: Target DataFrame.
        schema: Data schema.
        max_components: Maximum number of components to compute.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        schema: DataSchema,
        max_components: int = 10,
    ) -> None:
        self._df = df
        self._schema = schema
        self._max_components = max_components

        self._fitted = False
        self._pca: Any = None
        self._X_scaled: np.ndarray | None = None
        self._feature_names: list[str] = []
        self._n_components = 0

    # ── Lazy fitting ──────────────────────────────────────

    def _fit(self) -> bool:
        """Fit PCA model.  Returns ``True`` on success."""
        if self._fitted:
            return self._pca is not None

        self._fitted = True

        cols = self._schema.numeric_columns
        if len(cols) < 2:
            return False

        try:
            from sklearn.decomposition import PCA
            from sklearn.preprocessing import StandardScaler
        except ImportError:
            logger.info("scikit-learn not installed; skipping PCA analysis.")
            return False

        df_clean = self._df[cols].dropna()
        if len(df_clean) < max(10, len(cols)):
            return False

        try:
            scaler = StandardScaler()
            X = scaler.fit_transform(df_clean)

            self._n_components = min(self._max_components, len(cols), len(df_clean) - 1)
            if self._n_components < 1:
                return False

            self._pca = PCA(n_components=self._n_components)
            self._pca.fit(X)
            self._X_scaled = X
            self._feature_names = list(cols)
            return True
        except Exception as exc:
            logger.warning("PCA failed: %s", exc)
            return False

    # ── Variance explained ────────────────────────────────

    def variance_explained(self) -> pd.DataFrame:
        """Return variance explained by each principal component.

        Returns:
            DataFrame with variance ratio, cumulative ratio, and eigenvalue
            per component.
        """
        if not self._fit():
            return pd.DataFrame()

        rows: list[dict] = []
        cum = np.cumsum(self._pca.explained_variance_ratio_)
        for i in range(self._n_components):
            rows.append({
                "component": f"PC{i + 1}",
                "variance_ratio": round(float(self._pca.explained_variance_ratio_[i]), 4),
                "cumulative_ratio": round(float(cum[i]), 4),
                "eigenvalue": round(float(self._pca.explained_variance_[i]), 4),
            })

        return pd.DataFrame(rows).set_index("component")

    # ── Loadings ──────────────────────────────────────────

    def loadings(self) -> pd.DataFrame:
        """Return PCA loadings (feature weights per component).

        Returns:
            DataFrame with features as rows and ``PC1 .. PCn`` as columns.
        """
        if not self._fit():
            return pd.DataFrame()

        n_show = min(5, self._n_components)
        cols = [f"PC{i + 1}" for i in range(n_show)]
        return pd.DataFrame(
            self._pca.components_[:n_show].T,
            index=self._feature_names,
            columns=cols,
        ).round(4)

    # ── Transformed coordinates ───────────────────────────

    def transformed(self, n_components: int = 2) -> pd.DataFrame:
        """Return data projected onto the first *n_components* principal components."""
        if not self._fit() or self._X_scaled is None:
            return pd.DataFrame()

        n = min(n_components, self._n_components)
        coords = self._pca.transform(self._X_scaled)[:, :n]
        cols = [f"PC{i + 1}" for i in range(n)]
        return pd.DataFrame(coords, columns=cols)

    # ── Summary ───────────────────────────────────────────

    def summary(self) -> dict[str, Any]:
        """Return a concise PCA summary."""
        ve = self.variance_explained()
        if ve.empty:
            return {}

        # Number of components to reach 90 % variance
        cum = ve["cumulative_ratio"]
        above_90 = cum[cum >= 0.90]
        n_for_90 = int(above_90.index[0].replace("PC", "")) if len(above_90) > 0 else len(cum)

        return {
            "n_components": len(ve),
            "total_variance_explained": round(float(cum.iloc[-1]), 4),
            "components_for_90pct": n_for_90,
            "top_component_variance": round(float(ve["variance_ratio"].iloc[0]), 4),
        }
