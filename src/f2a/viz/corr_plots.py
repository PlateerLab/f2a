"""상관관계 시각화 모듈."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from f2a.core.schema import DataSchema
from f2a.viz.theme import DEFAULT_THEME, F2ATheme


class CorrelationPlotter:
    """상관관계 시각화를 생성합니다."""

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
        """상관계수 히트맵을 생성합니다.

        Args:
            method: 상관계수 방법 (``"pearson"`` 또는 ``"spearman"``).
            **kwargs: ``seaborn.heatmap``에 전달할 추가 인자.

        Returns:
            matplotlib Figure.
        """
        cols = self._schema.numeric_columns
        if len(cols) < 2:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "상관 분석을 위한 수치형 컬럼이 부족합니다", ha="center", va="center")
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
        ax.set_title(f"상관계수 히트맵 ({method.title()})", fontsize=self._theme.title_size)
        fig.tight_layout()
        return fig

    def pairplot(self, columns: list[str] | None = None, max_cols: int = 6, **kwargs: Any) -> sns.PairGrid:
        """수치형 컬럼들의 페어플롯을 생성합니다.

        Args:
            columns: 대상 컬럼. ``None``이면 수치형 상위 ``max_cols`` 개.
            max_cols: 최대 컬럼 수.
            **kwargs: ``seaborn.pairplot``에 전달할 추가 인자.

        Returns:
            seaborn PairGrid.
        """
        cols = columns or self._schema.numeric_columns[:max_cols]
        if len(cols) < 2:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "페어플롯을 위한 컬럼이 부족합니다", ha="center", va="center")
            return fig

        kwargs.setdefault("diag_kind", "kde")
        return sns.pairplot(self._df[cols], **kwargs)
