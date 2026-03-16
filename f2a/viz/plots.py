"""Basic plots — histograms, boxplots, bar charts."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from f2a.core.schema import DataSchema
from f2a.viz.theme import DEFAULT_THEME, F2ATheme


class BasicPlotter:
    """Generate basic visualizations.

    Args:
        df: Target DataFrame for visualization.
        schema: Data schema.
        theme: Visualization theme (default: ``DEFAULT_THEME``).
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

    def histograms(self, columns: list[str] | None = None, **kwargs: Any) -> plt.Figure:
        """Generate histograms for numeric columns.

        Args:
            columns: Target column list. ``None`` for all numeric columns.
            **kwargs: Additional arguments passed to ``seaborn.histplot``.

        Returns:
            matplotlib Figure.
        """
        cols = columns or self._schema.numeric_columns
        if not cols:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "No numeric columns found", ha="center", va="center")
            return fig

        n = len(cols)
        ncols = min(3, n)
        nrows = (n + ncols - 1) // ncols

        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
        axes = axes.flat if n > 1 else [axes]

        for idx, col in enumerate(cols):
            ax = axes[idx]
            kwargs.setdefault("kde", True)
            sns.histplot(data=self._df, x=col, ax=ax, **kwargs)
            ax.set_title(col)

        # Hide empty subplots
        for idx in range(n, len(list(axes))):
            axes[idx].set_visible(False)

        fig.suptitle("Numeric Column Distributions", fontsize=self._theme.title_size + 2, y=1.02)
        fig.tight_layout()
        return fig

    def boxplots(self, columns: list[str] | None = None, **kwargs: Any) -> plt.Figure:
        """Generate boxplots for numeric columns.

        Args:
            columns: Target column list.
            **kwargs: Additional arguments passed to ``seaborn.boxplot``.

        Returns:
            matplotlib Figure.
        """
        cols = columns or self._schema.numeric_columns
        if not cols:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "No numeric columns found", ha="center", va="center")
            return fig

        n = len(cols)
        ncols = min(3, n)
        nrows = (n + ncols - 1) // ncols

        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
        axes = axes.flat if n > 1 else [axes]

        for idx, col in enumerate(cols):
            ax = axes[idx]
            sns.boxplot(data=self._df, y=col, ax=ax, **kwargs)
            ax.set_title(col)

        for idx in range(n, len(list(axes))):
            axes[idx].set_visible(False)

        fig.suptitle("Numeric Column Boxplots", fontsize=self._theme.title_size + 2, y=1.02)
        fig.tight_layout()
        return fig

    def bar_charts(self, columns: list[str] | None = None, top_n: int = 15, **kwargs: Any) -> plt.Figure:
        """Generate frequency bar charts for categorical columns.

        Args:
            columns: Target column list. ``None`` for all categorical columns.
            top_n: Maximum categories to display per column.
            **kwargs: Additional arguments passed to ``seaborn.barplot``.

        Returns:
            matplotlib Figure.
        """
        cols = columns or self._schema.categorical_columns
        if not cols:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "No categorical columns found", ha="center", va="center")
            return fig

        n = len(cols)
        ncols = min(2, n)
        nrows = (n + ncols - 1) // ncols

        fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows))
        axes = axes.flat if n > 1 else [axes]

        for idx, col in enumerate(cols):
            ax = axes[idx]
            vc = self._df[col].value_counts().head(top_n)
            sns.barplot(x=vc.values, y=vc.index, ax=ax, **kwargs)
            ax.set_title(f"{col} (top {min(top_n, len(vc))})")
            ax.set_xlabel("Frequency")

        for idx in range(n, len(list(axes))):
            axes[idx].set_visible(False)

        fig.suptitle("Categorical Column Frequencies", fontsize=self._theme.title_size + 2, y=1.02)
        fig.tight_layout()
        return fig
