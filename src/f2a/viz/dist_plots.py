"""Distribution visualization module."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from f2a.core.schema import DataSchema
from f2a.viz.theme import DEFAULT_THEME, F2ATheme


class DistributionPlotter:
    """Generate distribution-related visualizations."""

    def __init__(
        self,
        df: pd.DataFrame,
        schema: DataSchema,
        theme: F2ATheme | None = None,
    ) -> None:
        self._df = df
        self._schema = schema
        self._theme = theme or DEFAULT_THEME

    def violin_plots(self, columns: list[str] | None = None, max_cols: int = 20, **kwargs: Any) -> plt.Figure:
        """Generate violin plots for numeric columns."""
        cols = (columns or self._schema.numeric_columns)[:max_cols]
        if not cols:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "No numeric columns found", ha="center", va="center")
            return fig

        n = len(cols)
        ncols = min(3, n)
        nrows = (n + ncols - 1) // ncols

        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
        if n == 1:
            axes = [axes]
        else:
            axes = list(axes.flat)

        for idx, col in enumerate(cols):
            ax = axes[idx]
            sns.violinplot(data=self._df, y=col, ax=ax, **kwargs)
            ax.set_title(col)

        for idx in range(n, len(axes)):
            axes[idx].set_visible(False)

        fig.suptitle("Violin Plots", fontsize=self._theme.title_size + 2, y=1.02)
        fig.tight_layout()
        return fig

    def kde_plots(self, columns: list[str] | None = None, max_cols: int = 20, **kwargs: Any) -> plt.Figure:
        """Generate KDE (Kernel Density Estimation) plots for numeric columns."""
        cols = (columns or self._schema.numeric_columns)[:max_cols]
        if not cols:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "No numeric columns found", ha="center", va="center")
            return fig

        n = len(cols)
        ncols = min(3, n)
        nrows = (n + ncols - 1) // ncols

        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
        if n == 1:
            axes = [axes]
        else:
            axes = list(axes.flat)

        for idx, col in enumerate(cols):
            ax = axes[idx]
            sns.kdeplot(data=self._df, x=col, ax=ax, fill=True, **kwargs)
            ax.set_title(col)

        for idx in range(n, len(axes)):
            axes[idx].set_visible(False)

        fig.suptitle("Kernel Density Estimation", fontsize=self._theme.title_size + 2, y=1.02)
        fig.tight_layout()
        return fig

    def qq_plots(self, columns: list[str] | None = None, max_cols: int = 20, **kwargs: Any) -> plt.Figure:
        """Generate Q-Q (Quantile-Quantile) plots for numeric columns.

        Points close to the diagonal indicate normal distribution.
        """
        cols = (columns or self._schema.numeric_columns)[:max_cols]
        if not cols:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "No numeric columns found", ha="center", va="center")
            return fig

        n = len(cols)
        ncols = min(3, n)
        nrows = (n + ncols - 1) // ncols

        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
        if n == 1:
            axes = [axes]
        else:
            axes = list(axes.flat)

        for idx, col in enumerate(cols):
            ax = axes[idx]
            series = self._df[col].dropna()
            if len(series) < 3:
                ax.text(0.5, 0.5, f"Not enough data\n(n={len(series)})",
                        ha="center", va="center", transform=ax.transAxes)
                ax.set_title(col)
                continue

            sp_stats.probplot(series, dist="norm", plot=ax)
            ax.set_title(col)
            ax.get_lines()[0].set_markersize(3)
            ax.get_lines()[0].set_alpha(0.5)

        for idx in range(n, len(axes)):
            axes[idx].set_visible(False)

        fig.suptitle("Q-Q Plots (Normal Distribution)", fontsize=self._theme.title_size + 2, y=1.02)
        fig.tight_layout()
        return fig
