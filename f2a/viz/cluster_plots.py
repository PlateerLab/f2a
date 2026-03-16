"""Clustering visualization module.

Provides elbow/silhouette plots, cluster scatter (2D PCA), DBSCAN results,
dendrogram, and cluster profile heatmap.
"""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

from f2a.viz.theme import DEFAULT_THEME, F2ATheme


class ClusterPlotter:
    """Visualise clustering analysis results.

    Args:
        theme: Visualisation theme.
    """

    def __init__(self, theme: F2ATheme | None = None) -> None:
        self._theme = theme or DEFAULT_THEME
        self._theme.apply()

    def elbow_silhouette(self, kmeans_result: dict[str, Any]) -> plt.Figure:
        """Plot elbow curve and silhouette scores.

        Args:
            kmeans_result: Dictionary from ClusteringStats.kmeans_analysis().
        """
        elbow_df = kmeans_result.get("elbow_data")
        if elbow_df is None or elbow_df.empty:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "No K-Means data", ha="center", va="center")
            return fig

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        ks = elbow_df.index.tolist()
        inertias = elbow_df["inertia"].values
        sil_scores = elbow_df["silhouette_score"].values
        optimal_k = kmeans_result.get("optimal_k", 0)

        # Elbow curve
        ax1.plot(ks, inertias, "bo-", linewidth=2, markersize=6)
        if optimal_k in ks:
            idx = ks.index(optimal_k)
            ax1.axvline(x=optimal_k, color="#e74c3c", linestyle="--",
                        alpha=0.7, label=f"Optimal k={optimal_k}")
            ax1.plot(optimal_k, inertias[idx], "r*", markersize=15, zorder=5)
        ax1.set_xlabel("k (number of clusters)")
        ax1.set_ylabel("Inertia")
        ax1.set_title("Elbow Method", fontsize=self._theme.title_size)
        ax1.legend()

        # Silhouette scores
        colors = ["#e74c3c" if k == optimal_k else "#3498db" for k in ks]
        ax2.bar(ks, sil_scores, color=colors, width=0.6)
        ax2.set_xlabel("k (number of clusters)")
        ax2.set_ylabel("Silhouette Score")
        ax2.set_title("Silhouette Score per k", fontsize=self._theme.title_size)

        for k, s in zip(ks, sil_scores):
            ax2.text(k, s + 0.01, f"{s:.3f}", ha="center", fontsize=8)

        fig.suptitle("K-Means Clustering Analysis",
                     fontsize=self._theme.title_size + 2, y=1.02)
        fig.tight_layout()
        return fig

    def cluster_scatter_2d(
        self,
        df: pd.DataFrame,
        numeric_cols: list[str],
        kmeans_result: dict[str, Any],
        max_sample: int = 2000,
    ) -> plt.Figure:
        """Scatter plot of clusters in 2D PCA space.

        Args:
            df: Original DataFrame.
            numeric_cols: Numeric column names.
            kmeans_result: K-Means result dict.
            max_sample: Max points to plot.
        """
        if not kmeans_result or len(numeric_cols) < 2:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "No cluster data", ha="center", va="center")
            return fig

        try:
            from sklearn.cluster import KMeans
            from sklearn.decomposition import PCA
            from sklearn.preprocessing import StandardScaler
        except ImportError:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "scikit-learn required", ha="center", va="center")
            return fig

        df_clean = df[numeric_cols].dropna()
        if len(df_clean) < 10:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center")
            return fig

        if len(df_clean) > max_sample:
            df_clean = df_clean.sample(max_sample, random_state=42)

        scaler = StandardScaler()
        X = scaler.fit_transform(df_clean)

        # K-Means fit
        optimal_k = kmeans_result.get("optimal_k", 3)
        km = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        labels = km.fit_predict(X)

        # PCA to 2D
        pca = PCA(n_components=2)
        X_2d = pca.fit_transform(X)

        fig, ax = plt.subplots(figsize=(10, 8))
        palette = sns.color_palette("husl", optimal_k)

        for cluster_id in range(optimal_k):
            mask = labels == cluster_id
            ax.scatter(
                X_2d[mask, 0], X_2d[mask, 1],
                c=[palette[cluster_id]], label=f"Cluster {cluster_id}",
                alpha=0.6, s=20,
            )

        # Centroids
        centroids_2d = pca.transform(km.cluster_centers_)
        ax.scatter(
            centroids_2d[:, 0], centroids_2d[:, 1],
            c="black", marker="X", s=150, zorder=5,
            label="Centroids",
        )

        var_explained = pca.explained_variance_ratio_
        ax.set_xlabel(f"PC1 ({var_explained[0] * 100:.1f}% var)")
        ax.set_ylabel(f"PC2 ({var_explained[1] * 100:.1f}% var)")
        ax.set_title(
            f"K-Means Clusters (k={optimal_k}) in PCA Space",
            fontsize=self._theme.title_size,
        )
        ax.legend(fontsize=8)
        fig.tight_layout()
        return fig

    def dendrogram(
        self,
        hierarchical_result: dict[str, Any],
        max_leaf: int = 30,
    ) -> plt.Figure:
        """Draw a dendrogram from hierarchical clustering.

        Args:
            hierarchical_result: Dictionary from hierarchical_analysis().
            max_leaf: Max leaf nodes shown.
        """
        Z = hierarchical_result.get("linkage_matrix")
        if Z is None:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "No hierarchical data", ha="center", va="center")
            return fig

        try:
            from scipy.cluster.hierarchy import dendrogram as scipy_dendro
        except ImportError:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "scipy required", ha="center", va="center")
            return fig

        fig, ax = plt.subplots(figsize=(12, 6))
        scipy_dendro(
            Z,
            ax=ax,
            truncate_mode="lastp" if len(Z) > max_leaf else None,
            p=max_leaf,
            leaf_rotation=90,
            leaf_font_size=8,
            color_threshold=0,
        )
        ax.set_title("Hierarchical Clustering Dendrogram (Ward)",
                      fontsize=self._theme.title_size)
        ax.set_xlabel("Sample index or cluster size")
        ax.set_ylabel("Distance")
        fig.tight_layout()
        return fig

    def cluster_profile_heatmap(
        self,
        profiles_df: pd.DataFrame,
        **kwargs: Any,
    ) -> plt.Figure:
        """Heatmap of cluster profiles (mean feature values per cluster).

        Args:
            profiles_df: DataFrame from ClusteringStats.cluster_profiles().
        """
        if profiles_df.empty:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "No cluster profile data", ha="center", va="center")
            return fig

        # Normalize for heatmap (z-score per column)
        normed = (profiles_df - profiles_df.mean()) / (profiles_df.std() + 1e-15)
        n_clusters = len(normed)
        n_features = len(normed.columns)

        fig, ax = plt.subplots(
            figsize=(max(8, n_features * 0.6), max(4, n_clusters * 1.2))
        )
        kwargs.setdefault("cmap", "RdYlGn")
        kwargs.setdefault("center", 0)
        kwargs.setdefault("annot", True)
        kwargs.setdefault("fmt", ".2f")
        kwargs.setdefault("linewidths", 0.5)

        sns.heatmap(normed, ax=ax, **kwargs)
        ax.set_title("Cluster Profiles (z-scored)", fontsize=self._theme.title_size)
        ax.set_ylabel("Cluster")
        fig.tight_layout()
        return fig
