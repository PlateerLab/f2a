"""상관 분석 모듈."""

from __future__ import annotations

import numpy as np
import pandas as pd

from f2a.core.schema import DataSchema
from f2a.utils.logging import get_logger

logger = get_logger(__name__)


class CorrelationStats:
    """컬럼 간 상관관계를 분석합니다.

    Args:
        df: 분석 대상 DataFrame.
        schema: 데이터 스키마.
    """

    def __init__(self, df: pd.DataFrame, schema: DataSchema) -> None:
        self._df = df
        self._schema = schema

    def pearson(self) -> pd.DataFrame:
        """Pearson 상관계수 행렬을 반환합니다."""
        cols = self._schema.numeric_columns
        if len(cols) < 2:
            return pd.DataFrame()
        return self._df[cols].corr(method="pearson")

    def spearman(self) -> pd.DataFrame:
        """Spearman 순위 상관계수 행렬을 반환합니다."""
        cols = self._schema.numeric_columns
        if len(cols) < 2:
            return pd.DataFrame()
        return self._df[cols].corr(method="spearman")

    def cramers_v_matrix(self) -> pd.DataFrame:
        """범주형 컬럼 간 Cramér's V 행렬을 반환합니다."""
        cols = self._schema.categorical_columns
        if len(cols) < 2:
            return pd.DataFrame()

        n = len(cols)
        matrix = pd.DataFrame(np.ones((n, n)), index=cols, columns=cols)

        for i in range(n):
            for j in range(i + 1, n):
                v = self._cramers_v(self._df[cols[i]], self._df[cols[j]])
                matrix.iloc[i, j] = v
                matrix.iloc[j, i] = v

        return matrix

    def high_correlations(self, threshold: float = 0.9) -> list[tuple[str, str, float]]:
        """높은 상관관계 쌍을 반환합니다.

        Args:
            threshold: 상관계수 절대값 임계치.

        Returns:
            ``(col_a, col_b, correlation)`` 튜플 리스트.
        """
        corr = self.pearson()
        if corr.empty:
            return []

        pairs: list[tuple[str, str, float]] = []
        cols = corr.columns
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                val = corr.iloc[i, j]
                if abs(val) >= threshold:
                    pairs.append((cols[i], cols[j], round(float(val), 4)))

        if pairs:
            logger.warning(
                "다중공선성 경고: %d쌍의 컬럼이 |r| ≥ %.2f 입니다.",
                len(pairs),
                threshold,
            )

        return pairs

    # ── 내부 헬퍼 ───────────────────────────────────────

    @staticmethod
    def _cramers_v(x: pd.Series, y: pd.Series) -> float:
        """두 범주형 변수 간 Cramér's V를 계산합니다."""
        confusion = pd.crosstab(x, y)
        n = confusion.sum().sum()
        if n == 0:
            return 0.0

        from scipy.stats import chi2_contingency

        chi2, _, _, _ = chi2_contingency(confusion)
        min_dim = min(confusion.shape) - 1
        if min_dim == 0:
            return 0.0

        return float(np.sqrt(chi2 / (n * min_dim)))
