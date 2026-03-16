"""PCA visualization module.

Scree plot (variance explained) and loadings heatmap.
"""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

from f2a.viz.theme import DEFAULT_THEME, F2ATheme


class PCAPlotter:
    """Visualise PCA results.

    Args:
        theme: Visualisation theme.
    """

    def __init__(self, theme: F2ATheme | None = None) -> None:
        self._theme = theme or DEFAULT_THEME
        self._theme.apply()

    def scree_plot(self, variance_df: pd.DataFrame, **kwargs: Any) -> plt.Figure:
        """Draw a scree plot (variance explained per component).

        Args:
            variance_df: DataFrame from :meth:`PCAStats.variance_explained`.
        """
        if variance_df.empty:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "PCA not available", ha="center", va="center")
            return fig

        components = variance_df.index.tolist()
        var_ratio = variance_df["variance_ratio"].values
        cum_ratio = variance_df["cumulative_ratio"].values
        n = len(components)

        fig, ax1 = plt.subplots(figsize=(max(8, n * 0.8), 5))
        color1 = "#3498DB"
        color2 = "#E74C3C"

        # Bar — individual variance
        bars = ax1.bar(range(n), var_ratio * 100, color=color1, alpha=0.7, label="Individual")
        ax1.set_xlabel("Principal Component")
        ax1.set_ylabel("Variance Explained (%)", color=color1)
        ax1.set_xticks(range(n))
        ax1.set_xticklabels(components, rotation=45 if n > 6 else 0)
        ax1.tick_params(axis="y", labelcolor=color1)

        # Line — cumulative variance
        ax2 = ax1.twinx()
        ax2.plot(range(n), cum_ratio * 100, color=color2, marker="o",
                 linewidth=2, label="Cumulative")
        ax2.set_ylabel("Cumulative Variance (%)", color=color2)
        ax2.tick_params(axis="y", labelcolor=color2)
        ax2.set_ylim(0, 105)

        # 90% threshold line
        ax2.axhline(y=90, color="#95A5A6", linestyle="--", alpha=0.7, label="90% threshold")

        # Annotate bars
        for i, (bar, v) in enumerate(zip(bars, var_ratio)):
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                     f"{v * 100:.1f}%", ha="center", va="bottom", fontsize=8)

        # Combined legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right")

        fig.suptitle("PCA Scree Plot", fontsize=self._theme.title_size + 2)
        fig.tight_layout()
        return fig

    def loadings_heatmap(self, loadings_df: pd.DataFrame, **kwargs: Any) -> plt.Figure:
        """Draw a heatmap of PCA loadings.

        Args:
            loadings_df: DataFrame from :meth:`PCAStats.loadings`.
        """
        if loadings_df.empty:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "PCA not available", ha="center", va="center")
            return fig

        n_features = len(loadings_df)
        n_components = len(loadings_df.columns)

        fig, ax = plt.subplots(
            figsize=(max(7, n_components * 1.5), max(5, n_features * 0.4))
        )

        kwargs.setdefault("annot", True)
        kwargs.setdefault("fmt", ".3f")
        kwargs.setdefault("cmap", "RdBu_r")
        kwargs.setdefault("center", 0)
        kwargs.setdefault("vmin", -1)
        kwargs.setdefault("vmax", 1)

        sns.heatmap(loadings_df, ax=ax, **kwargs)
        ax.set_title("PCA Loadings", fontsize=self._theme.title_size)
        ax.set_ylabel("Feature")
        ax.set_xlabel("Component")
        fig.tight_layout()
        return fig
