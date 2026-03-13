"""분포 분석 모듈."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from f2a.core.schema import DataSchema


class DistributionStats:
    """수치형 컬럼의 분포 특성을 분석합니다.

    Args:
        df: 분석 대상 DataFrame.
        schema: 데이터 스키마.
    """

    def __init__(self, df: pd.DataFrame, schema: DataSchema) -> None:
        self._df = df
        self._schema = schema

    def analyze(self) -> pd.DataFrame:
        """수치형 컬럼들의 분포 정보를 반환합니다.

        Returns:
            왜도, 첨도, 정규성 검정 결과를 포함한 DataFrame.
        """
        cols = self._schema.numeric_columns
        if not cols:
            return pd.DataFrame()

        rows: list[dict] = []
        for col in cols:
            series = self._df[col].dropna()
            if len(series) < 3:
                continue
            rows.append(self._analyze_column(col, series))

        return pd.DataFrame(rows).set_index("column") if rows else pd.DataFrame()

    def quantile_table(self, quantiles: list[float] | None = None) -> pd.DataFrame:
        """수치형 컬럼들의 분위수 테이블을 반환합니다.

        Args:
            quantiles: 계산할 분위수 리스트. 기본값은
                ``[0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]``.

        Returns:
            분위수 DataFrame.
        """
        if quantiles is None:
            quantiles = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]

        cols = self._schema.numeric_columns
        if not cols:
            return pd.DataFrame()

        return self._df[cols].quantile(quantiles)

    @staticmethod
    def _analyze_column(col: str, series: pd.Series) -> dict:
        """단일 수치형 컬럼의 분포를 분석합니다."""
        skew = float(series.skew())
        kurt = float(series.kurtosis())

        # 정규성 검정
        normality_p: float | None = None
        normality_test: str = "n/a"

        n = len(series)
        if 3 <= n <= 5000:
            _, normality_p = sp_stats.shapiro(series)
            normality_test = "shapiro"
        elif n > 5000:
            _, normality_p = sp_stats.normaltest(series)
            normality_test = "dagostino"

        return {
            "column": col,
            "skewness": round(skew, 4),
            "kurtosis": round(kurt, 4),
            "normality_test": normality_test,
            "normality_p": round(normality_p, 6) if normality_p is not None else None,
            "is_normal_0.05": normality_p > 0.05 if normality_p is not None else None,
        }
