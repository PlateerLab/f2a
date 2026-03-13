"""기술 통계 분석 모듈."""

from __future__ import annotations

import numpy as np
import pandas as pd

from f2a.core.schema import DataSchema
from f2a.utils.type_inference import ColumnType


class DescriptiveStats:
    """기술 통계량을 계산합니다.

    Args:
        df: 분석 대상 DataFrame.
        schema: 데이터 스키마.
    """

    def __init__(self, df: pd.DataFrame, schema: DataSchema) -> None:
        self._df = df
        self._schema = schema

    def summary(self) -> pd.DataFrame:
        """전체 요약 통계를 반환합니다.

        수치형과 범주형 컬럼을 모두 포함하는 통합 요약표를 생성합니다.

        Returns:
            요약 통계 DataFrame.
        """
        rows: list[dict] = []
        for col_info in self._schema.columns:
            series = self._df[col_info.name]
            row: dict = {
                "column": col_info.name,
                "type": col_info.inferred_type.value,
                "count": int(series.count()),
                "missing": col_info.n_missing,
                "missing_%": round(col_info.missing_ratio * 100, 2),
                "unique": col_info.n_unique,
            }

            if col_info.inferred_type == ColumnType.NUMERIC:
                row.update(self._numeric_stats(series))
            elif col_info.inferred_type in (ColumnType.CATEGORICAL, ColumnType.BOOLEAN):
                row.update(self._categorical_stats(series))

            rows.append(row)

        return pd.DataFrame(rows).set_index("column")

    def numeric_summary(self) -> pd.DataFrame:
        """수치형 컬럼만의 요약 통계를 반환합니다."""
        cols = self._schema.numeric_columns
        if not cols:
            return pd.DataFrame()
        return self._df[cols].describe().T

    def categorical_summary(self) -> pd.DataFrame:
        """범주형 컬럼만의 요약 통계를 반환합니다."""
        cols = self._schema.categorical_columns
        if not cols:
            return pd.DataFrame()

        rows: list[dict] = []
        for col in cols:
            series = self._df[col]
            top_val = series.mode().iloc[0] if not series.mode().empty else None
            rows.append(
                {
                    "column": col,
                    "count": int(series.count()),
                    "unique": int(series.nunique()),
                    "top": top_val,
                    "freq": int(series.value_counts().iloc[0]) if top_val is not None else 0,
                }
            )
        return pd.DataFrame(rows).set_index("column")

    # ── 내부 헬퍼 ───────────────────────────────────────

    @staticmethod
    def _numeric_stats(series: pd.Series) -> dict:
        """수치형 컬럼의 통계를 딕셔너리로 반환합니다."""
        desc = series.describe()
        q1 = float(desc.get("25%", np.nan))
        q3 = float(desc.get("75%", np.nan))
        return {
            "mean": round(float(series.mean()), 4),
            "median": round(float(series.median()), 4),
            "std": round(float(series.std()), 4),
            "min": float(series.min()),
            "max": float(series.max()),
            "range": round(float(series.max() - series.min()), 4),
            "q1": round(q1, 4),
            "q3": round(q3, 4),
            "iqr": round(q3 - q1, 4),
        }

    @staticmethod
    def _categorical_stats(series: pd.Series) -> dict:
        """범주형 컬럼의 통계를 딕셔너리로 반환합니다."""
        vc = series.value_counts()
        top_val = vc.index[0] if len(vc) > 0 else None
        return {
            "top": top_val,
            "freq": int(vc.iloc[0]) if len(vc) > 0 else 0,
        }
