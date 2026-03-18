"""Correlation visualization module."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from f2a.core.schema import DataSchema
from f2a.viz.theme import DEFAULT_THEME, F2ATheme


class CorrelationPlotter:
    """Generate correlation visualizations."""

    def __init__(
        self,
        df: pd.DataFrame,
        schema: DataSchema,
        theme: F2ATheme | None = None,
    ) -> None:
        self._df = df
        self._schema = schema
        self._theme = theme or DEFAULT_THEME

    def heatmap(self, method: str = "pearson", **kwargs: Any) -> plt.Figure:
        """Generate a correlation coefficient heatmap.

        Args:
            method: Correlation method (``"pearson"`` or ``"spearman"``).
            **kwargs: Additional arguments passed to ``seaborn.heatmap``.

        Returns:
            matplotlib Figure.
        """
        cols = self._schema.numeric_columns
        if len(cols) < 2:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "Not enough numeric columns for correlation analysis", ha="center", va="center")
            return fig

        corr = self._df[cols].corr(method=method)

        fig, ax = plt.subplots(figsize=(max(8, len(cols)), max(6, len(cols) * 0.8)))
        kwargs.setdefault("annot", True)
        kwargs.setdefault("fmt", ".2f")
        kwargs.setdefault("cmap", "coolwarm")
        kwargs.setdefault("center", 0)
        kwargs.setdefault("vmin", -1)
        kwargs.setdefault("vmax", 1)
        kwargs.setdefault("square", True)

        sns.heatmap(corr, ax=ax, **kwargs)
        ax.set_title(f"Correlation Heatmap ({method.title()})", fontsize=self._theme.title_size)
        fig.tight_layout()
        return fig

    def pairplot(self, columns: list[str] | None = None, max_cols: int = 6, **kwargs: Any) -> sns.PairGrid:
        """Generate pairplot for numeric columns.

        Args:
            columns: Target columns. ``None`` for top ``max_cols`` numeric columns.
            max_cols: Maximum number of columns.
            **kwargs: Additional arguments passed to ``seaborn.pairplot``.

        Returns:
            seaborn PairGrid.
        """
        cols = columns or self._schema.numeric_columns[:max_cols]
        if len(cols) < 2:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "Not enough columns for pairplot", ha="center", va="center")
            return fig

        kwargs.setdefault("diag_kind", "kde")
        return sns.pairplot(self._df[cols], **kwargs)
