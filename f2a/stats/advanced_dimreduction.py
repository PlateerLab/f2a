"""Advanced dimensionality reduction module.

Provides t-SNE, UMAP (optional), and Factor Analysis for
non-linear dimensionality reduction and latent factor discovery.

References:
    - van der Maaten & Hinton (2008) — t-SNE
    - McInnes et al. (2018) — UMAP
    - Spearman (1904) — Factor Analysis
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from f2a.core.schema import DataSchema
from f2a.utils.logging import get_logger

logger = get_logger(__name__)


class AdvancedDimReductionStats:
    """Advanced dimensionality reduction analysis.

    Args:
        df: Target DataFrame.
        schema: Data schema.
        tsne_perplexity: t-SNE perplexity parameter.
        max_sample: Max rows to sample.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        schema: DataSchema,
        tsne_perplexity: float = 30.0,
        max_sample: int = 5000,
    ) -> None:
        self._df = df
        self._schema = schema
        self._tsne_perplexity = tsne_perplexity
        self._max_sample = max_sample

    def _prepare_data(self) -> tuple[np.ndarray, pd.DataFrame, list[str]] | None:
        """Scale and sample numeric data."""
        cols = self._schema.numeric_columns
        if len(cols) < 3:
            return None

        try:
            from sklearn.preprocessing import StandardScaler
        except ImportError:
            return None

        df_clean = self._df[cols].dropna()
        if len(df_clean) < 20:
            return None

        if len(df_clean) > self._max_sample:
            df_clean = df_clean.sample(self._max_sample, random_state=42)

        scaler = StandardScaler()
        X = scaler.fit_transform(df_clean)
        return X, df_clean, cols

    # ── t-SNE ─────────────────────────────────────────────

    def tsne_2d(self) -> dict[str, Any]:
        """Compute t-SNE 2D embedding.

        t-SNE (t-distributed Stochastic Neighbor Embedding) is excellent
        for visualizing high-dimensional data in 2D.

        Returns:
            Dictionary with embedding coordinates, parameters.
        """
        prepared = self._prepare_data()
        if prepared is None:
            return {}

        X, df_clean, cols = prepared

        try:
            from sklearn.manifold import TSNE
        except ImportError:
            return {}

        perplexity = min(self._tsne_perplexity, max(5, len(X) / 4))

        try:
            tsne = TSNE(
                n_components=2,
                perplexity=perplexity,
                random_state=42,
                max_iter=1000,
                learning_rate="auto",
                init="pca",
            )
            embedding = tsne.fit_transform(X)

            return {
                "method": "t-SNE",
                "embedding": pd.DataFrame(
                    embedding,
                    columns=["tsne_1", "tsne_2"],
                ),
                "perplexity": perplexity,
                "kl_divergence": round(float(tsne.kl_divergence_), 4),
                "n_samples": len(X),
                "n_features": X.shape[1],
            }
        except Exception as exc:
            logger.debug("t-SNE failed: %s", exc)
            return {}

    # ── UMAP ──────────────────────────────────────────────

    def umap_2d(self) -> dict[str, Any]:
        """Compute UMAP 2D embedding (if umap-learn is installed).

        UMAP (Uniform Manifold Approximation and Projection) preserves
        both local and global structure better than t-SNE.

        Returns:
            Dictionary with embedding coordinates, parameters.
        """
        prepared = self._prepare_data()
        if prepared is None:
            return {}

        X, df_clean, cols = prepared

        try:
            from umap import UMAP
        except ImportError:
            logger.info("umap-learn not installed; UMAP analysis skipped.")
            return {}

        try:
            n_neighbors = min(15, max(2, len(X) // 10))
            reducer = UMAP(
                n_components=2,
                n_neighbors=n_neighbors,
                min_dist=0.1,
                random_state=42,
            )
            embedding = reducer.fit_transform(X)

            return {
                "method": "UMAP",
                "embedding": pd.DataFrame(
                    embedding,
                    columns=["umap_1", "umap_2"],
                ),
                "n_neighbors": n_neighbors,
                "min_dist": 0.1,
                "n_samples": len(X),
                "n_features": X.shape[1],
            }
        except Exception as exc:
            logger.debug("UMAP failed: %s", exc)
            return {}

    # ── Factor Analysis ───────────────────────────────────

    def factor_analysis(self, n_factors: int | None = None) -> dict[str, Any]:
        """Perform Factor Analysis to discover latent factors.

        Factor Analysis models observed variables as linear combinations
        of unobserved latent factors plus error terms.

        Args:
            n_factors: Number of factors. Auto-detected if None.

        Returns:
            Dictionary with loadings, variance explained, factor scores.
        """
        prepared = self._prepare_data()
        if prepared is None:
            return {}

        X, df_clean, cols = prepared

        try:
            from sklearn.decomposition import FactorAnalysis
        except ImportError:
            return {}

        # Auto-detect n_factors using eigenvalue > 1 rule (Kaiser criterion)
        if n_factors is None:
            cov = np.cov(X.T)
            eigenvalues = np.linalg.eigvalsh(cov)[::-1]
            n_factors = max(1, int((eigenvalues > 1).sum()))
            n_factors = min(n_factors, len(cols) - 1, 10)

        if n_factors < 1:
            return {}

        try:
            fa = FactorAnalysis(n_components=n_factors, random_state=42)
            fa.fit(X)

            loadings = pd.DataFrame(
                fa.components_.T,
                index=cols,
                columns=[f"factor_{i + 1}" for i in range(n_factors)],
            ).round(4)

            # Variance explained by each factor (approximate)
            factor_var = np.sum(fa.components_ ** 2, axis=1)
            total_var = np.sum(np.var(X, axis=0))
            var_explained = factor_var / total_var

            noise_variance = pd.DataFrame({
                "column": cols,
                "noise_variance": np.round(fa.noise_variance_, 4),
                "communality": np.round(
                    1 - fa.noise_variance_ / np.maximum(np.var(X, axis=0), 1e-15), 4
                ),
            }).set_index("column")

            return {
                "method": "Factor Analysis",
                "n_factors": n_factors,
                "loadings": loadings,
                "variance_explained": [round(float(v), 4) for v in var_explained],
                "total_variance_explained": round(float(var_explained.sum()), 4),
                "noise_variance": noise_variance,
                "n_samples": len(X),
                "n_features": len(cols),
            }
        except Exception as exc:
            logger.debug("Factor Analysis failed: %s", exc)
            return {}

    # ── Feature contributions ─────────────────────────────

    def feature_contribution(self) -> pd.DataFrame:
        """Analyze feature contributions across dimensionality reduction.

        Computes how much each feature contributes to the variance
        captured by PCA components.

        Returns:
            DataFrame with feature importance across components.
        """
        cols = self._schema.numeric_columns
        if len(cols) < 3:
            return pd.DataFrame()

        try:
            from sklearn.decomposition import PCA
            from sklearn.preprocessing import StandardScaler
        except ImportError:
            return pd.DataFrame()

        df_clean = self._df[cols].dropna()
        if len(df_clean) < len(cols) + 1:
            return pd.DataFrame()

        if len(df_clean) > self._max_sample:
            df_clean = df_clean.sample(self._max_sample, random_state=42)

        scaler = StandardScaler()
        X = scaler.fit_transform(df_clean)

        n_comp = min(5, len(cols), len(df_clean) - 1)
        pca = PCA(n_components=n_comp)
        pca.fit(X)

        # Weighted contribution: |loading| * variance_explained
        contributions = np.zeros(len(cols))
        for i in range(n_comp):
            contributions += np.abs(pca.components_[i]) * pca.explained_variance_ratio_[i]

        result = pd.DataFrame({
            "column": cols,
            "contribution_score": np.round(contributions, 4),
            "rank": np.argsort(-contributions) + 1,
        }).sort_values("contribution_score", ascending=False).set_index("column")

        return result

    # ── Summary ───────────────────────────────────────────

    def summary(self) -> dict[str, Any]:
        """Return combined dimensionality reduction results."""
        result: dict[str, Any] = {}

        try:
            tsne = self.tsne_2d()
            if tsne:
                # Store summary without large embedding
                result["tsne"] = {
                    k: v for k, v in tsne.items() if k != "embedding"
                }
                if "embedding" in tsne:
                    result["tsne_embedding"] = tsne["embedding"]
        except Exception as exc:
            logger.debug("t-SNE skipped: %s", exc)

        try:
            umap_res = self.umap_2d()
            if umap_res:
                result["umap"] = {
                    k: v for k, v in umap_res.items() if k != "embedding"
                }
                if "embedding" in umap_res:
                    result["umap_embedding"] = umap_res["embedding"]
        except Exception as exc:
            logger.debug("UMAP skipped: %s", exc)

        try:
            fa = self.factor_analysis()
            if fa:
                result["factor_analysis"] = {
                    k: v for k, v in fa.items()
                    if k not in ("loadings", "noise_variance")
                }
                if "loadings" in fa:
                    result["factor_loadings"] = fa["loadings"]
                if "noise_variance" in fa:
                    result["factor_noise"] = fa["noise_variance"]
        except Exception as exc:
            logger.debug("Factor Analysis skipped: %s", exc)

        try:
            fc = self.feature_contribution()
            if not fc.empty:
                result["feature_contribution"] = fc
        except Exception as exc:
            logger.debug("Feature contribution skipped: %s", exc)

        return result
