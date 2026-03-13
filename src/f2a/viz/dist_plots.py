"""분포 시각화 모듈."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from f2a.core.schema import DataSchema
from f2a.viz.theme import DEFAULT_THEME, F2ATheme


class DistributionPlotter:
    """분포 관련 시각화를 생성합니다."""

    def __init__(
        self,
        df: pd.DataFrame,
        schema: DataSchema,
        theme: F2ATheme | None = None,
    ) -> None:
        self._df = df
        self._schema = schema
        self._theme = theme or DEFAULT_THEME

    def violin_plots(self, columns: list[str] | None = None, **kwargs: Any) -> plt.Figure:
        """수치형 컬럼들의 바이올린 플롯을 생성합니다."""
        cols = columns or self._schema.numeric_columns
        if not cols:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "수치형 컬럼이 없습니다", ha="center", va="center")
            return fig

        n = len(cols)
        ncols = min(3, n)
        nrows = (n + ncols - 1) // ncols

        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
        axes = axes.flat if n > 1 else [axes]

        for idx, col in enumerate(cols):
            ax = axes[idx]
            sns.violinplot(data=self._df, y=col, ax=ax, **kwargs)
            ax.set_title(col)

        for idx in range(n, len(list(axes))):
            axes[idx].set_visible(False)

        fig.suptitle("바이올린 플롯", fontsize=self._theme.title_size + 2, y=1.02)
        fig.tight_layout()
        return fig

    def kde_plots(self, columns: list[str] | None = None, **kwargs: Any) -> plt.Figure:
        """수치형 컬럼들의 KDE(커널 밀도 추정) 플롯을 생성합니다."""
        cols = columns or self._schema.numeric_columns
        if not cols:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "수치형 컬럼이 없습니다", ha="center", va="center")
            return fig

        n = len(cols)
        ncols = min(3, n)
        nrows = (n + ncols - 1) // ncols

        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
        axes = axes.flat if n > 1 else [axes]

        for idx, col in enumerate(cols):
            ax = axes[idx]
            sns.kdeplot(data=self._df, x=col, ax=ax, fill=True, **kwargs)
            ax.set_title(col)

        for idx in range(n, len(list(axes))):
            axes[idx].set_visible(False)

        fig.suptitle("커널 밀도 추정", fontsize=self._theme.title_size + 2, y=1.02)
        fig.tight_layout()
        return fig
