"""Categorical visualization module.

Bar charts, pie/donut charts, and chi-square heatmaps for categorical columns.
"""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

from f2a.core.schema import DataSchema
from f2a.viz.theme import DEFAULT_THEME, F2ATheme


class CategoricalPlotter:
    """Visualise categorical column distributions.

    Args:
        df: Target DataFrame.
        schema: Data schema.
        theme: Visualisation theme.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        schema: DataSchema,
        theme: F2ATheme | None = None,
    ) -> None:
        self._df = df
        self._schema = schema
        self._theme = theme or DEFAULT_THEME
        self._theme.apply()

    def frequency_bars(
        self,
        columns: list[str] | None = None,
        top_n: int = 15,
        max_cols: int = 20,
        **kwargs: Any,
    ) -> plt.Figure:
        """Horizontal frequency bar charts for categorical columns.

        Args:
            columns: Columns to plot.  Defaults to all categorical columns.
            top_n: Max categories per column.
            max_cols: Max subplot count.
            **kwargs: Passed to ``seaborn.barplot``.
        """
        cols = (columns or self._schema.categorical_columns)[:max_cols]
        if not cols:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "No categorical columns", ha="center", va="center")
            return fig

        n = len(cols)
        ncols_grid = min(2, n)
        nrows = (n + ncols_grid - 1) // ncols_grid

        fig, axes = plt.subplots(nrows, ncols_grid, figsize=(7 * ncols_grid, 4 * nrows))
        if n == 1:
            axes = [axes]
        else:
            axes = list(axes.flat)

        palette = self._theme.get_colors(top_n)

        for idx, col in enumerate(cols):
            ax = axes[idx]
            vc = self._df[col].value_counts().head(top_n)
            colors = palette[:len(vc)]
            ax.barh(range(len(vc)), vc.values, color=colors)
            ax.set_yticks(range(len(vc)))
            ax.set_yticklabels([str(v)[:30] for v in vc.index], fontsize=9)
            ax.invert_yaxis()
            ax.set_xlabel("Frequency")
            ax.set_title(f"{col} (top {min(top_n, len(vc))})")

            # Annotate with count
            for i, v in enumerate(vc.values):
                ax.text(v + max(vc.values) * 0.01, i, str(v), va="center", fontsize=8)

        for idx in range(n, len(axes)):
            axes[idx].set_visible(False)

        fig.suptitle("Categorical Column Frequencies",
                     fontsize=self._theme.title_size + 2, y=1.02)
        fig.tight_layout()
        return fig

    def chi_square_heatmap(
        self,
        chi_sq_matrix: pd.DataFrame,
        **kwargs: Any,
    ) -> plt.Figure:
        """Render the chi-square p-value matrix as a heatmap.

        Args:
            chi_sq_matrix: Square DataFrame of p-values from
                :meth:`CategoricalStats.chi_square_matrix`.
        """
        if chi_sq_matrix.empty:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "Not enough categorical columns", ha="center", va="center")
            return fig

        n = len(chi_sq_matrix)
        fig, ax = plt.subplots(figsize=(max(7, n * 0.8), max(6, n * 0.7)))

        kwargs.setdefault("annot", True)
        kwargs.setdefault("fmt", ".4f")
        kwargs.setdefault("cmap", "YlOrRd_r")
        kwargs.setdefault("vmin", 0)
        kwargs.setdefault("vmax", 1)
        kwargs.setdefault("square", True)

        sns.heatmap(chi_sq_matrix, ax=ax, **kwargs)
        ax.set_title("Chi-Square Independence Test p-values\n(low = significant association)",
                      fontsize=self._theme.title_size)
        fig.tight_layout()
        return fig
