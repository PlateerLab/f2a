"""데이터 타입 자동 추론 유틸리티."""

from __future__ import annotations

from enum import Enum

import pandas as pd


class ColumnType(str, Enum):
    """컬럼 타입 분류."""

    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    TEXT = "text"
    DATETIME = "datetime"
    BOOLEAN = "boolean"


# 카테고리로 간주할 유니크 값 비율 상한
_CATEGORICAL_RATIO_THRESHOLD = 0.05  # 5%
# 카테고리로 간주할 절대 유니크 수 상한
_CATEGORICAL_UNIQUE_THRESHOLD = 50
# 텍스트로 간주할 평균 문자열 길이 하한
_TEXT_LENGTH_THRESHOLD = 50


def infer_column_type(series: pd.Series) -> ColumnType:
    """단일 컬럼의 의미론적 타입을 추론합니다.

    Args:
        series: 분석 대상 pandas Series.

    Returns:
        추론된 :class:`ColumnType`.
    """
    # boolean 체크
    if series.dtype == "bool" or set(series.dropna().unique()) <= {True, False, 0, 1}:
        return ColumnType.BOOLEAN

    # datetime 체크
    if pd.api.types.is_datetime64_any_dtype(series):
        return ColumnType.DATETIME

    # 수치형 체크
    if pd.api.types.is_numeric_dtype(series):
        n_unique = series.nunique()
        n_total = len(series)
        # 유니크 값이 매우 적으면 카테고리로 간주
        if n_unique <= 10 and n_total > 100:
            return ColumnType.CATEGORICAL
        return ColumnType.NUMERIC

    # 문자열 계열
    if pd.api.types.is_string_dtype(series) or pd.api.types.is_object_dtype(series):
        n_unique = series.nunique()
        n_total = len(series.dropna())

        if n_total == 0:
            return ColumnType.TEXT

        # datetime 파싱 시도
        try:
            pd.to_datetime(series.dropna().head(20))
            return ColumnType.DATETIME
        except (ValueError, TypeError):
            pass

        # 유니크 비율 및 문자열 길이로 텍스트 vs 카테고리 결정
        ratio = n_unique / n_total if n_total > 0 else 1.0
        avg_len = series.dropna().astype(str).str.len().mean()

        if avg_len > _TEXT_LENGTH_THRESHOLD:
            return ColumnType.TEXT
        if n_unique <= _CATEGORICAL_UNIQUE_THRESHOLD or ratio <= _CATEGORICAL_RATIO_THRESHOLD:
            return ColumnType.CATEGORICAL
        return ColumnType.TEXT

    return ColumnType.TEXT


def infer_all_types(df: pd.DataFrame) -> dict[str, ColumnType]:
    """DataFrame의 모든 컬럼 타입을 추론합니다.

    Args:
        df: 분석 대상 DataFrame.

    Returns:
        컬럼명 → :class:`ColumnType` 매핑.
    """
    return {col: infer_column_type(df[col]) for col in df.columns}
