"""Advanced anomaly detection visualization module.

Provides anomaly scatter plot, Mahalanobis distance histogram,
consensus comparison chart, and t-SNE / UMAP anomaly overlay.
"""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

from f2a.viz.theme import DEFAULT_THEME, F2ATheme


class AdvancedAnomalyPlotter:
    """Visualise advanced anomaly detection results.

    Args:
        theme: Visualisation theme.
    """

    def __init__(self, theme: F2ATheme | None = None) -> None:
        self._theme = theme or DEFAULT_THEME
        self._theme.apply()

    def anomaly_scatter_2d(
        self,
        df: pd.DataFrame,
        numeric_cols: list[str],
        anomaly_result: dict[str, Any],
        max_sample: int = 2000,
    ) -> plt.Figure:
        """Scatter plot of anomalies in 2D PCA space.

        Args:
            df: Original DataFrame.
            numeric_cols: Numeric column names.
            anomaly_result: Result dict with 'labels' key (-1 = anomaly).
            max_sample: Max points to plot.
        """
        labels = anomaly_result.get("labels")
        if labels is None or len(numeric_cols) < 2:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "No anomaly data", ha="center", va="center")
            return fig

        labels = np.asarray(labels)  # ensure ndarray for element-wise comparison

        try:
            from sklearn.decomposition import PCA
            from sklearn.preprocessing import StandardScaler
        except ImportError:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "scikit-learn required", ha="center", va="center")
            return fig

        df_clean = df[numeric_cols].dropna()
        # Align labels with df_clean by truncating to min length first, then sample
        n = min(len(df_clean), len(labels))
        df_clean = df_clean.iloc[:n]
        labels = labels[:n]
        if n > max_sample:
            rng = np.random.RandomState(42)
            idx = rng.choice(n, size=max_sample, replace=False)
            df_clean = df_clean.iloc[idx]
            labels = labels[idx]

        scaler = StandardScaler()
        X = scaler.fit_transform(df_clean)
        pca = PCA(n_components=2)
        X_2d = pca.fit_transform(X)

        fig, ax = plt.subplots(figsize=(10, 8))

        normal_mask = labels == 1
        anomaly_mask = labels == -1

        ax.scatter(X_2d[normal_mask, 0], X_2d[normal_mask, 1],
                   c="#3498db", alpha=0.3, s=15, label="Normal")
        ax.scatter(X_2d[anomaly_mask, 0], X_2d[anomaly_mask, 1],
                   c="#e74c3c", alpha=0.8, s=40, marker="x",
                   label=f"Anomaly ({anomaly_mask.sum()})")

        method = anomaly_result.get("method", "Unknown")
        var_explained = pca.explained_variance_ratio_
        ax.set_xlabel(f"PC1 ({var_explained[0] * 100:.1f}%)")
        ax.set_ylabel(f"PC2 ({var_explained[1] * 100:.1f}%)")
        ax.set_title(f"Anomaly Detection: {method}", fontsize=self._theme.title_size)
        ax.legend()
        fig.tight_layout()
        return fig

    def mahalanobis_histogram(
        self,
        maha_result: dict[str, Any],
    ) -> plt.Figure:
        """Histogram of Mahalanobis distances with threshold line.

        Args:
            maha_result: Result dict from mahalanobis_distance().
        """
        distances = maha_result.get("distances")
        if distances is None:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "No Mahalanobis data", ha="center", va="center")
            return fig

        threshold = maha_result.get("threshold", 0)

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist(distances, bins=50, color="#3498db", alpha=0.7, edgecolor="#2980b9")
        ax.axvline(x=threshold, color="#e74c3c", linestyle="--", linewidth=2,
                    label=f"Threshold ({threshold:.2f})")

        n_anomaly = maha_result.get("anomaly_count", 0)
        ax.set_xlabel("Mahalanobis Distance")
        ax.set_ylabel("Frequency")
        ax.set_title(
            f"Mahalanobis Distance Distribution ({n_anomaly} anomalies)",
            fontsize=self._theme.title_size,
        )
        ax.legend()
        fig.tight_layout()
        return fig

    def consensus_comparison(
        self,
        consensus_result: dict[str, Any],
    ) -> plt.Figure:
        """Bar chart comparing anomaly counts across methods.

        Args:
            consensus_result: Result dict from consensus_anomaly().
        """
        per_method = consensus_result.get("per_method_counts", {})
        if not per_method:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "No consensus data", ha="center", va="center")
            return fig

        # Add consensus count
        counts = dict(per_method)
        counts["consensus"] = consensus_result.get("consensus_count", 0)

        labels = list(counts.keys())
        values = list(counts.values())
        colors = ["#3498db"] * len(per_method) + ["#e74c3c"]

        fig, ax = plt.subplots(figsize=(8, 5))
        bars = ax.bar(labels, values, color=colors, width=0.6)

        for bar, v in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    str(v), ha="center", fontsize=10, fontweight="bold")

        n_samples = consensus_result.get("n_samples", 0)
        ax.set_ylabel("Anomaly Count")
        ax.set_title(
            f"Anomaly Detection Method Comparison (n={n_samples})",
            fontsize=self._theme.title_size,
        )
        plt.xticks(rotation=15, ha="right")
        fig.tight_layout()
        return fig

    def tsne_anomaly_overlay(
        self,
        embedding: pd.DataFrame,
        anomaly_labels: np.ndarray,
    ) -> plt.Figure:
        """Overlay anomaly labels on t-SNE / UMAP embedding.

        Args:
            embedding: DataFrame with 2 columns (e.g. tsne_1, tsne_2).
            anomaly_labels: Array of -1 (anomaly) or 1 (normal).
        """
        if embedding.empty or anomaly_labels is None:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "No embedding data", ha="center", va="center")
            return fig

        anomaly_labels = np.asarray(anomaly_labels)  # ensure ndarray
        if embedding.shape[1] < 2:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "Need ≥ 2 embedding dims", ha="center", va="center")
            return fig

        n = min(len(embedding), len(anomaly_labels))
        X = embedding.values[:n]
        labels = anomaly_labels[:n]

        fig, ax = plt.subplots(figsize=(10, 8))

        normal = labels == 1
        anomaly = labels == -1

        ax.scatter(X[normal, 0], X[normal, 1], c="#3498db", alpha=0.3, s=15, label="Normal")
        ax.scatter(X[anomaly, 0], X[anomaly, 1], c="#e74c3c", alpha=0.8, s=40,
                   marker="x", label=f"Anomaly ({anomaly.sum()})")

        col_names = embedding.columns.tolist()
        ax.set_xlabel(col_names[0])
        ax.set_ylabel(col_names[1])
        ax.set_title("Anomaly Overlay on Embedding", fontsize=self._theme.title_size)
        ax.legend()
        fig.tight_layout()
        return fig
