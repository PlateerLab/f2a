"""결측치 시각화 모듈."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

from f2a.core.schema import DataSchema
from f2a.viz.theme import DEFAULT_THEME, F2ATheme


class MissingPlotter:
    """결측치 패턴 시각화를 생성합니다."""

    def __init__(
        self,
        df: pd.DataFrame,
        schema: DataSchema,
        theme: F2ATheme | None = None,
    ) -> None:
        self._df = df
        self._schema = schema
        self._theme = theme or DEFAULT_THEME

    def matrix(self, max_rows: int = 500, **kwargs: Any) -> plt.Figure:
        """결측치 매트릭스를 생성합니다.

        흰색 = 결측, 색상 = 존재.

        Args:
            max_rows: 표시할 최대 행 수 (샘플링).
            **kwargs: 추가 인자.

        Returns:
            matplotlib Figure.
        """
        df_sample = self._df.head(max_rows)
        missing = df_sample.isna()

        fig, ax = plt.subplots(figsize=(max(10, len(self._df.columns) * 0.5), 6))
        ax.imshow(
            ~missing.values,
            aspect="auto",
            cmap="RdYlGn",
            interpolation="nearest",
        )
        ax.set_xticks(range(len(missing.columns)))
        ax.set_xticklabels(missing.columns, rotation=45, ha="right")
        ax.set_ylabel("행 인덱스")
        ax.set_title("결측치 매트릭스 (녹색=존재, 빨강=결측)")
        fig.tight_layout()
        return fig

    def bar(self, **kwargs: Any) -> plt.Figure:
        """컬럼별 결측 비율 바 차트를 생성합니다."""
        missing_ratio = self._df.isna().mean().sort_values(ascending=False)
        missing_ratio = missing_ratio[missing_ratio > 0]

        if missing_ratio.empty:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "결측치가 없습니다!", ha="center", va="center", fontsize=14)
            return fig

        fig, ax = plt.subplots(figsize=(max(8, len(missing_ratio) * 0.5), 5))
        colors = ["#e74c3c" if v > 0.5 else "#f39c12" if v > 0.1 else "#2ecc71" for v in missing_ratio]
        ax.bar(range(len(missing_ratio)), missing_ratio.values * 100, color=colors)
        ax.set_xticks(range(len(missing_ratio)))
        ax.set_xticklabels(missing_ratio.index, rotation=45, ha="right")
        ax.set_ylabel("결측 비율 (%)")
        ax.set_title("컬럼별 결측 비율")
        ax.axhline(y=50, color="red", linestyle="--", alpha=0.5, label="50%")
        ax.legend()
        fig.tight_layout()
        return fig
