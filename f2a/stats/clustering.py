"""Clustering analysis module.

Provides K-Means (with elbow + silhouette), DBSCAN (auto-eps),
hierarchical clustering, and cluster profiling.

References:
    - MacQueen (1967) — K-Means
    - Ester et al. (1996) — DBSCAN
    - Rousseeuw (1987) — silhouette score
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from f2a.core.schema import DataSchema
from f2a.utils.logging import get_logger

logger = get_logger(__name__)


class ClusteringStats:
    """Clustering analysis for numeric columns.

    Args:
        df: Target DataFrame.
        schema: Data schema.
        max_k: Maximum k for K-Means elbow search.
        max_sample: Max rows to sample.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        schema: DataSchema,
        max_k: int = 10,
        max_sample: int = 5000,
    ) -> None:
        self._df = df
        self._schema = schema
        self._max_k = max_k
        self._max_sample = max_sample

    def _prepare_data(self) -> tuple[np.ndarray, list[str]] | None:
        """Scale and sample numeric data for clustering."""
        cols = self._schema.numeric_columns
        if len(cols) < 2:
            return None

        try:
            from sklearn.preprocessing import StandardScaler
        except ImportError:
            logger.info("scikit-learn not available for clustering.")
            return None

        df_clean = self._df[cols].dropna()
        if len(df_clean) < 10:
            return None

        if len(df_clean) > self._max_sample:
            df_clean = df_clean.sample(self._max_sample, random_state=42)

        scaler = StandardScaler()
        X = scaler.fit_transform(df_clean)
        return X, cols

    # ── K-Means with elbow & silhouette ───────────────────

    def kmeans_analysis(self) -> dict[str, Any]:
        """Perform K-Means clustering with elbow and silhouette analysis.

        Returns:
            Dictionary with:
            - ``elbow_data``: DataFrame (k, inertia, silhouette)
            - ``optimal_k``: best k by silhouette score
            - ``labels``: cluster labels for optimal k
            - ``cluster_sizes``: dict of cluster → count
        """
        prepared = self._prepare_data()
        if prepared is None:
            return {}

        X, cols = prepared

        try:
            from sklearn.cluster import KMeans
            from sklearn.metrics import silhouette_score
        except ImportError:
            return {}

        max_k = min(self._max_k, len(X) - 1)
        if max_k < 2:
            return {}

        rows: list[dict] = []
        best_score = -1.0
        best_k = 2
        best_labels: np.ndarray | None = None

        for k in range(2, max_k + 1):
            try:
                km = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300)
                labels = km.fit_predict(X)
                inertia = float(km.inertia_)
                sil = float(silhouette_score(X, labels))

                rows.append({
                    "k": k,
                    "inertia": round(inertia, 2),
                    "silhouette_score": round(sil, 4),
                })

                if sil > best_score:
                    best_score = sil
                    best_k = k
                    best_labels = labels
            except Exception:
                continue

        if not rows:
            return {}

        elbow_df = pd.DataFrame(rows).set_index("k")

        # Cluster sizes
        sizes: dict[str, int] = {}
        if best_labels is not None:
            unique, counts = np.unique(best_labels, return_counts=True)
            sizes = {f"cluster_{int(u)}": int(c) for u, c in zip(unique, counts)}

        return {
            "elbow_data": elbow_df,
            "optimal_k": best_k,
            "best_silhouette": round(best_score, 4),
            "cluster_sizes": sizes,
            "n_samples": len(X),
        }

    # ── DBSCAN with auto-eps ──────────────────────────────

    def dbscan_analysis(self) -> dict[str, Any]:
        """Perform DBSCAN clustering with automated eps selection.

        Uses the k-distance graph method to estimate eps.

        Returns:
            Dictionary with labels, n_clusters, n_noise, cluster_sizes.
        """
        prepared = self._prepare_data()
        if prepared is None:
            return {}

        X, cols = prepared

        try:
            from sklearn.cluster import DBSCAN
            from sklearn.neighbors import NearestNeighbors
        except ImportError:
            return {}

        # Auto-eps via k-distance graph (k = min_samples)
        min_samples = max(2, min(5, len(X) // 20))
        try:
            nn = NearestNeighbors(n_neighbors=min_samples)
            nn.fit(X)
            distances, _ = nn.kneighbors(X)
            k_distances = np.sort(distances[:, -1])

            # Estimate eps at the "elbow" using maximum curvature
            n = len(k_distances)
            if n < 10:
                return {}

            # Simple elbow: point of maximum second derivative
            second_deriv = np.diff(k_distances, n=2)
            elbow_idx = int(np.argmax(second_deriv)) + 1
            eps = float(k_distances[elbow_idx])
            eps = max(eps, 0.1)  # minimum eps

            db = DBSCAN(eps=eps, min_samples=min_samples)
            labels = db.fit_predict(X)

            n_clusters = len(set(labels) - {-1})
            n_noise = int((labels == -1).sum())

            sizes: dict[str, int] = {}
            unique, counts = np.unique(labels, return_counts=True)
            for u, c in zip(unique, counts):
                lbl = "noise" if u == -1 else f"cluster_{int(u)}"
                sizes[lbl] = int(c)

            return {
                "eps": round(eps, 4),
                "min_samples": min_samples,
                "n_clusters": n_clusters,
                "n_noise": n_noise,
                "noise_ratio": round(n_noise / len(X), 4),
                "cluster_sizes": sizes,
                "n_samples": len(X),
            }
        except Exception as exc:
            logger.debug("DBSCAN failed: %s", exc)
            return {}

    # ── Hierarchical clustering ───────────────────────────

    def hierarchical_analysis(self) -> dict[str, Any]:
        """Perform hierarchical (agglomerative) clustering.

        Returns:
            Dictionary with n_clusters (auto), labels, linkage method,
            and dendrogram data.
        """
        prepared = self._prepare_data()
        if prepared is None:
            return {}

        X, cols = prepared

        try:
            from sklearn.cluster import AgglomerativeClustering
            from sklearn.metrics import silhouette_score
            from scipy.cluster.hierarchy import linkage
        except ImportError:
            return {}

        # Try different n_clusters, pick best silhouette
        best_k = 2
        best_score = -1.0

        max_k = min(self._max_k, len(X) - 1)
        for k in range(2, max_k + 1):
            try:
                agg = AgglomerativeClustering(n_clusters=k, linkage="ward")
                labels = agg.fit_predict(X)
                score = float(silhouette_score(X, labels))
                if score > best_score:
                    best_score = score
                    best_k = k
            except Exception:
                continue

        # Final fit with best k
        try:
            agg = AgglomerativeClustering(n_clusters=best_k, linkage="ward")
            labels = agg.fit_predict(X)

            sizes: dict[str, int] = {}
            unique, counts = np.unique(labels, return_counts=True)
            for u, c in zip(unique, counts):
                sizes[f"cluster_{int(u)}"] = int(c)

            # Linkage matrix for dendrogram
            Z = linkage(X[:min(500, len(X))], method="ward")

            return {
                "optimal_k": best_k,
                "silhouette_score": round(best_score, 4),
                "linkage_method": "ward",
                "cluster_sizes": sizes,
                "linkage_matrix": Z,
                "n_samples": len(X),
            }
        except Exception as exc:
            logger.debug("Hierarchical clustering failed: %s", exc)
            return {}

    # ── Cluster profiling ─────────────────────────────────

    def cluster_profiles(self, kmeans_result: dict[str, Any] | None = None) -> pd.DataFrame:
        """Profile clusters by computing per-cluster mean of each feature.

        Uses the optimal K-Means clustering result.

        Args:
            kmeans_result: Pre-computed K-Means result (avoids re-running).

        Returns:
            DataFrame with cluster labels as index, feature means as columns.
        """
        if kmeans_result is None:
            kmeans_result = self.kmeans_analysis()
        if not kmeans_result:
            return pd.DataFrame()

        prepared = self._prepare_data()
        if prepared is None:
            return pd.DataFrame()

        X, cols = prepared
        optimal_k = kmeans_result["optimal_k"]

        try:
            from sklearn.cluster import KMeans
        except ImportError:
            return pd.DataFrame()

        try:
            km = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
            labels = km.fit_predict(X)

            # Build profiles using original (unscaled) data
            df_clean = self._df[cols].dropna()
            if len(df_clean) > self._max_sample:
                df_clean = df_clean.sample(self._max_sample, random_state=42)

            df_clean = df_clean.copy()
            df_clean["cluster"] = labels[: len(df_clean)]

            profiles = df_clean.groupby("cluster").mean().round(4)
            profiles.index = [f"cluster_{i}" for i in profiles.index]

            return profiles
        except Exception:
            return pd.DataFrame()

    # ── Summary ───────────────────────────────────────────

    def summary(self) -> dict[str, Any]:
        """Return combined clustering analysis results."""
        result: dict[str, Any] = {}

        try:
            km = self.kmeans_analysis()
            if km:
                result["kmeans"] = km
        except Exception as exc:
            logger.debug("K-Means analysis skipped: %s", exc)

        try:
            db = self.dbscan_analysis()
            if db:
                result["dbscan"] = db
        except Exception as exc:
            logger.debug("DBSCAN analysis skipped: %s", exc)

        try:
            hc = self.hierarchical_analysis()
            if hc:
                result["hierarchical"] = hc
        except Exception as exc:
            logger.debug("Hierarchical analysis skipped: %s", exc)

        try:
            cp = self.cluster_profiles(kmeans_result=km if "kmeans" in result else None)
            if not cp.empty:
                result["profiles"] = cp
        except Exception as exc:
            logger.debug("Cluster profiling skipped: %s", exc)

        return result
