"""Advanced correlation visualization module.

Provides partial correlation heatmap, MI heatmap, bootstrap CI plot,
correlation network graph, and distance correlation heatmap.
"""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

from f2a.viz.theme import DEFAULT_THEME, F2ATheme


class AdvancedCorrPlotter:
    """Visualise advanced correlation analysis results.

    Args:
        theme: Visualisation theme.
    """

    def __init__(self, theme: F2ATheme | None = None) -> None:
        self._theme = theme or DEFAULT_THEME
        self._theme.apply()

    def partial_correlation_heatmap(
        self,
        pcorr: pd.DataFrame,
        **kwargs: Any,
    ) -> plt.Figure:
        """Heatmap of partial correlations.

        Args:
            pcorr: Partial correlation matrix DataFrame.
        """
        if pcorr.empty:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "No partial correlation data", ha="center", va="center")
            return fig

        n = len(pcorr)
        fig, ax = plt.subplots(figsize=(max(8, n * 0.7), max(6, n * 0.6)))
        kwargs.setdefault("annot", n <= 15)
        kwargs.setdefault("fmt", ".2f")
        kwargs.setdefault("cmap", "RdBu_r")
        kwargs.setdefault("center", 0)
        kwargs.setdefault("vmin", -1)
        kwargs.setdefault("vmax", 1)
        kwargs.setdefault("square", True)

        sns.heatmap(pcorr, ax=ax, **kwargs)
        ax.set_title("Partial Correlation Matrix", fontsize=self._theme.title_size)
        fig.tight_layout()
        return fig

    def mi_heatmap(
        self,
        mi_matrix: pd.DataFrame,
        **kwargs: Any,
    ) -> plt.Figure:
        """Heatmap of mutual information values.

        Args:
            mi_matrix: MI matrix DataFrame.
        """
        if mi_matrix.empty:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "No MI data", ha="center", va="center")
            return fig

        n = len(mi_matrix)
        fig, ax = plt.subplots(figsize=(max(8, n * 0.7), max(6, n * 0.6)))
        kwargs.setdefault("annot", n <= 15)
        kwargs.setdefault("fmt", ".3f")
        kwargs.setdefault("cmap", "YlOrRd")
        kwargs.setdefault("square", True)

        sns.heatmap(mi_matrix, ax=ax, **kwargs)
        ax.set_title("Mutual Information Matrix", fontsize=self._theme.title_size)
        fig.tight_layout()
        return fig

    def bootstrap_ci_plot(
        self,
        ci_df: pd.DataFrame,
        max_pairs: int = 20,
    ) -> plt.Figure:
        """Plot bootstrap confidence intervals for correlations.

        Args:
            ci_df: DataFrame from bootstrap_correlation_ci().
            max_pairs: Max number of pairs to show.
        """
        if ci_df.empty:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "No bootstrap CI data", ha="center", va="center")
            return fig

        required_cols = {"col_a", "col_b", "pearson_r", "ci_lower", "ci_upper", "significant"}
        if not required_cols.issubset(ci_df.columns):
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "Incomplete CI data", ha="center", va="center")
            return fig

        df = ci_df.head(max_pairs).copy()
        df["label"] = df["col_a"] + " ↔ " + df["col_b"]
        df = df.sort_values("pearson_r")

        fig, ax = plt.subplots(figsize=(10, max(5, len(df) * 0.35)))

        y = range(len(df))
        colors = ["#27ae60" if sig else "#95a5a6" for sig in df["significant"]]

        ax.barh(y, df["pearson_r"], height=0.4, color=colors, alpha=0.7, label="Point estimate")

        for i, (_, row) in enumerate(df.iterrows()):
            ax.plot(
                [row["ci_lower"], row["ci_upper"]],
                [i, i],
                color="#2c3e50",
                linewidth=2,
                solid_capstyle="round",
            )

        ax.axvline(x=0, color="#e74c3c", linestyle="--", alpha=0.5)
        ax.set_yticks(y)
        ax.set_yticklabels(df["label"], fontsize=8)
        ax.set_xlabel("Pearson r")
        ax.set_title("Bootstrap 95% CI for Correlations", fontsize=self._theme.title_size)
        ax.legend(fontsize=8)
        fig.tight_layout()
        return fig

    def correlation_network(
        self,
        network_data: dict[str, Any],
    ) -> plt.Figure:
        """Draw a correlation network graph.

        Args:
            network_data: Dictionary from correlation_network().
        """
        nodes = network_data.get("nodes", [])
        edges = network_data.get("edges", [])

        if not nodes or not edges:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "No network data", ha="center", va="center")
            return fig

        fig, ax = plt.subplots(figsize=(10, 8))

        # Layout nodes in a circle
        n = len(nodes)
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        pos = {node: (np.cos(a), np.sin(a)) for node, a in zip(nodes, angles)}

        # Draw edges
        for edge in edges:
            src = edge.get("source")
            tgt = edge.get("target")
            if src not in pos or tgt not in pos:
                continue
            w = edge.get("abs_weight", 0.5)
            color = "#e74c3c" if edge.get("weight", 0) < 0 else "#27ae60"
            ax.plot(
                [pos[src][0], pos[tgt][0]],
                [pos[src][1], pos[tgt][1]],
                color=color,
                alpha=min(w, 1.0),
                linewidth=w * 3,
            )

        # Draw nodes
        for node in nodes:
            x, y = pos[node]
            ax.scatter(x, y, s=200, c="#3498db", zorder=5, edgecolors="#2c3e50")
            ax.annotate(
                node, (x, y),
                textcoords="offset points",
                xytext=(0, 12),
                ha="center",
                fontsize=8,
                fontweight="bold",
            )

        threshold = network_data.get("threshold", 0.5)
        ax.set_title(
            f"Correlation Network (|r| ≥ {threshold}, {len(edges)} edges)",
            fontsize=self._theme.title_size,
        )
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_aspect("equal")
        ax.axis("off")

        # Legend
        from matplotlib.lines import Line2D

        legend_elements = [
            Line2D([0], [0], color="#27ae60", linewidth=2, label="Positive"),
            Line2D([0], [0], color="#e74c3c", linewidth=2, label="Negative"),
        ]
        ax.legend(handles=legend_elements, loc="lower right", fontsize=8)

        fig.tight_layout()
        return fig

    def distance_correlation_heatmap(
        self,
        dcorr: pd.DataFrame,
        **kwargs: Any,
    ) -> plt.Figure:
        """Heatmap of distance correlations.

        Args:
            dcorr: Distance correlation matrix DataFrame.
        """
        if dcorr.empty:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "No distance correlation data", ha="center", va="center")
            return fig

        n = len(dcorr)
        fig, ax = plt.subplots(figsize=(max(8, n * 0.7), max(6, n * 0.6)))
        kwargs.setdefault("annot", n <= 15)
        kwargs.setdefault("fmt", ".3f")
        kwargs.setdefault("cmap", "YlOrRd")
        kwargs.setdefault("vmin", 0)
        kwargs.setdefault("vmax", 1)
        kwargs.setdefault("square", True)

        sns.heatmap(dcorr, ax=ax, **kwargs)
        ax.set_title("Distance Correlation Matrix", fontsize=self._theme.title_size)
        fig.tight_layout()
        return fig
