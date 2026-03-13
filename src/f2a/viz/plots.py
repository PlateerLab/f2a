"""기본 플롯 — 히스토그램, 박스플롯, 바 차트."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from f2a.core.schema import DataSchema
from f2a.viz.theme import DEFAULT_THEME, F2ATheme


class BasicPlotter:
    """기본 시각화를 생성합니다.

    Args:
        df: 시각화 대상 DataFrame.
        schema: 데이터 스키마.
        theme: 시각화 테마 (기본: ``DEFAULT_THEME``).
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
        """수치형 컬럼들의 히스토그램을 생성합니다.

        Args:
            columns: 대상 컬럼 목록. ``None``이면 모든 수치형 컬럼.
            **kwargs: ``seaborn.histplot``에 전달할 추가 인자.

        Returns:
            matplotlib Figure.
        """
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
            kwargs.setdefault("kde", True)
            sns.histplot(data=self._df, x=col, ax=ax, **kwargs)
            ax.set_title(col)

        # 빈 서브플롯 숨김
        for idx in range(n, len(list(axes))):
            axes[idx].set_visible(False)

        fig.suptitle("수치형 컬럼 분포", fontsize=self._theme.title_size + 2, y=1.02)
        fig.tight_layout()
        return fig

    def boxplots(self, columns: list[str] | None = None, **kwargs: Any) -> plt.Figure:
        """수치형 컬럼들의 박스플롯을 생성합니다.

        Args:
            columns: 대상 컬럼 목록.
            **kwargs: ``seaborn.boxplot``에 전달할 추가 인자.

        Returns:
            matplotlib Figure.
        """
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
            sns.boxplot(data=self._df, y=col, ax=ax, **kwargs)
            ax.set_title(col)

        for idx in range(n, len(list(axes))):
            axes[idx].set_visible(False)

        fig.suptitle("수치형 컬럼 박스플롯", fontsize=self._theme.title_size + 2, y=1.02)
        fig.tight_layout()
        return fig

    def bar_charts(self, columns: list[str] | None = None, top_n: int = 15, **kwargs: Any) -> plt.Figure:
        """범주형 컬럼들의 빈도 바 차트를 생성합니다.

        Args:
            columns: 대상 컬럼 목록. ``None``이면 모든 범주형 컬럼.
            top_n: 각 컬럼에서 표시할 최대 카테고리 수.
            **kwargs: ``seaborn.barplot``에 전달할 추가 인자.

        Returns:
            matplotlib Figure.
        """
        cols = columns or self._schema.categorical_columns
        if not cols:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "범주형 컬럼이 없습니다", ha="center", va="center")
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
            ax.set_title(f"{col} (상위 {min(top_n, len(vc))}개)")
            ax.set_xlabel("빈도")

        for idx in range(n, len(list(axes))):
            axes[idx].set_visible(False)

        fig.suptitle("범주형 컬럼 빈도", fontsize=self._theme.title_size + 2, y=1.02)
        fig.tight_layout()
        return fig
