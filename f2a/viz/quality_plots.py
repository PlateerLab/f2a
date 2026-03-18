"""Data quality visualization module.

Bar chart and heatmap of data-quality scores.
"""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

from f2a.viz.theme import DEFAULT_THEME, F2ATheme


class QualityPlotter:
    """Visualise data-quality scores.

    Args:
        theme: Visualisation theme.
    """

    def __init__(self, theme: F2ATheme | None = None) -> None:
        self._theme = theme or DEFAULT_THEME
        self._theme.apply()

    def dimension_bar(self, scores: dict[str, float], **kwargs: Any) -> plt.Figure:
        """Bar chart of quality dimension scores.

        Args:
            scores: Dict from :meth:`QualityStats.summary` containing
                ``completeness``, ``uniqueness``, ``consistency``,
                ``validity``, ``overall``.
        """
        if not scores:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "No quality scores", ha="center", va="center")
            return fig

        dims = ["completeness", "uniqueness", "consistency", "validity", "overall"]
        labels = ["Completeness", "Uniqueness", "Consistency", "Validity", "Overall"]
        values = [scores.get(d, 0) * 100 for d in dims]

        fig, ax = plt.subplots(figsize=(8, 5))

        colors = []
        for v in values:
            if v >= 90:
                colors.append("#27AE60")
            elif v >= 70:
                colors.append("#F39C12")
            else:
                colors.append("#E74C3C")

        bars = ax.barh(range(len(labels)), values, color=colors, height=0.6)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels)
        ax.invert_yaxis()
        ax.set_xlim(0, 105)
        ax.set_xlabel("Score (%)")
        ax.set_title("Data Quality Scores", fontsize=self._theme.title_size)

        # Threshold lines
        ax.axvline(x=90, color="#27AE60", linestyle="--", alpha=0.4, label="Good (90%)")
        ax.axvline(x=70, color="#F39C12", linestyle="--", alpha=0.4, label="Fair (70%)")
        ax.legend(fontsize=8, loc="lower right")

        # Value labels
        for bar, v in zip(bars, values):
            ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                    f"{v:.1f}%", va="center", fontsize=10, fontweight="bold")

        fig.tight_layout()
        return fig

    def column_quality_heatmap(
        self,
        quality_df: pd.DataFrame,
        **kwargs: Any,
    ) -> plt.Figure:
        """Heatmap of per-column quality scores.

        Args:
            quality_df: DataFrame from :meth:`QualityStats.column_quality`.
        """
        if quality_df.empty:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "No column quality data", ha="center", va="center")
            return fig

        numeric_cols = [c for c in quality_df.columns if c in ("completeness", "uniqueness", "quality_score")]
        if not numeric_cols:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "No numeric quality columns", ha="center", va="center")
            return fig

        plot_df = quality_df[numeric_cols]

        # Limit rows for readability
        if len(plot_df) > 40:
            plot_df = plot_df.head(40)

        fig, ax = plt.subplots(figsize=(max(6, len(numeric_cols) * 2), max(5, len(plot_df) * 0.3)))

        kwargs.setdefault("annot", True)
        kwargs.setdefault("fmt", ".2f")
        kwargs.setdefault("cmap", "RdYlGn")
        kwargs.setdefault("vmin", 0)
        kwargs.setdefault("vmax", 1)

        sns.heatmap(plot_df, ax=ax, **kwargs)
        ax.set_title("Column Quality Scores", fontsize=self._theme.title_size)
        ax.set_ylabel("")
        fig.tight_layout()
        return fig

    def feature_importance_bar(
        self,
        importance_df: pd.DataFrame,
        title: str = "Feature Importance (Variance Ranking)",
        **kwargs: Any,
    ) -> plt.Figure:
        """Horizontal bar chart of feature importance scores.

        Args:
            importance_df: DataFrame with numeric importance values.
            title: Chart title.
        """
        if importance_df.empty:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "No feature importance data", ha="center", va="center")
            return fig

        # Use first numeric column as the value
        val_col = None
        for c in importance_df.columns:
            if importance_df[c].dtype in ("float64", "float32", "int64", "int32"):
                val_col = c
                break
        if val_col is None:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "No numeric column found", ha="center", va="center")
            return fig

        df = importance_df.head(20).copy()
        df = df.sort_values(val_col, ascending=True)

        fig, ax = plt.subplots(figsize=(8, max(4, len(df) * 0.35)))
        colors = sns.color_palette("viridis", len(df))
        ax.barh(range(len(df)), df[val_col].values, color=colors, height=0.7)
        ax.set_yticks(range(len(df)))
        ax.set_yticklabels(df.index)
        ax.set_xlabel(val_col)
        ax.set_title(title, fontsize=self._theme.title_size)

        fig.tight_layout()
        return fig
