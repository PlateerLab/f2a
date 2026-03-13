"""Outlier visualization module.

Provides boxplots with strip overlay to highlight outlier points.
"""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

from f2a.core.schema import DataSchema
from f2a.viz.theme import DEFAULT_THEME, F2ATheme


class OutlierPlotter:
    """Visualise outliers in numeric columns.

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

    def box_strip(
        self,
        columns: list[str] | None = None,
        max_cols: int = 20,
        multiplier: float = 1.5,
        **kwargs: Any,
    ) -> plt.Figure:
        """Box-and-strip plot with outlier points highlighted.

        Args:
            columns: Columns to plot.  Defaults to all numeric columns.
            max_cols: Maximum number of columns (avoids overly large figures).
            multiplier: IQR multiplier for outlier classification.
            **kwargs: Passed to ``seaborn.boxplot``.

        Returns:
            matplotlib Figure.
        """
        cols = (columns or self._schema.numeric_columns)[:max_cols]
        if not cols:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "No numeric columns", ha="center", va="center")
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

            # Boxplot
            sns.boxplot(data=self._df, y=col, ax=ax, width=0.4,
                        color="#AED6F1", **kwargs)

            # Overlay strip/scatter — highlight outliers
            if len(series) > 0:
                q1 = series.quantile(0.25)
                q3 = series.quantile(0.75)
                iqr = q3 - q1
                lower = q1 - multiplier * iqr
                upper = q3 + multiplier * iqr
                is_outlier = (series < lower) | (series > upper)

                normal = series[~is_outlier]
                outliers = series[is_outlier]

                # Normal points (light / small)
                if len(normal) > 0:
                    sample = normal.sample(min(100, len(normal)), random_state=42)
                    ax.scatter(
                        np.random.normal(0, 0.04, len(sample)),
                        sample,
                        alpha=0.3, s=8, color="#2980B9", zorder=3,
                    )

                # Outlier points (red / larger)
                if len(outliers) > 0:
                    ax.scatter(
                        np.random.normal(0, 0.04, len(outliers)),
                        outliers,
                        alpha=0.7, s=25, color="#E74C3C", zorder=4,
                        label=f"outliers ({len(outliers)})",
                    )
                    ax.legend(fontsize=8)

            ax.set_title(col)

        for idx in range(n, len(axes)):
            axes[idx].set_visible(False)

        fig.suptitle("Outlier Detection (IQR Method)",
                     fontsize=self._theme.title_size + 2, y=1.02)
        fig.tight_layout()
        return fig
