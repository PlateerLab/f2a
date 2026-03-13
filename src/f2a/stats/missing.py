"""결측치 분석 모듈."""

from __future__ import annotations

import pandas as pd

from f2a.core.schema import DataSchema


class MissingStats:
    """결측치 패턴을 분석합니다.

    Args:
        df: 분석 대상 DataFrame.
        schema: 데이터 스키마.
    """

    def __init__(self, df: pd.DataFrame, schema: DataSchema) -> None:
        self._df = df
        self._schema = schema

    def column_summary(self) -> pd.DataFrame:
        """컬럼별 결측치 요약을 반환합니다.

        Returns:
            각 컬럼의 결측 수, 비율, 타입 정보를 포함한 DataFrame.
        """
        rows: list[dict] = []
        for col_info in self._schema.columns:
            rows.append(
                {
                    "column": col_info.name,
                    "missing_count": col_info.n_missing,
                    "missing_ratio": col_info.missing_ratio,
                    "missing_%": round(col_info.missing_ratio * 100, 2),
                    "dtype": col_info.dtype,
                }
            )

        result = pd.DataFrame(rows).set_index("column")
        return result.sort_values("missing_count", ascending=False)

    def row_missing_distribution(self) -> pd.DataFrame:
        """행 단위 결측 수 분포를 반환합니다.

        Returns:
            행별 결측 수의 빈도표.
        """
        row_missing = self._df.isna().sum(axis=1)
        dist = row_missing.value_counts().sort_index()
        return pd.DataFrame(
            {
                "missing_per_row": dist.index,
                "row_count": dist.values,
                "row_%": (dist.values / len(self._df) * 100).round(2),
            }
        )

    def missing_matrix(self) -> pd.DataFrame:
        """결측치 매트릭스 (boolean)를 반환합니다.

        시각화에 사용되는 결측 여부 행렬입니다.

        Returns:
            결측이면 True인 boolean DataFrame.
        """
        return self._df.isna()

    def total_missing_ratio(self) -> float:
        """전체 결측 비율을 반환합니다."""
        total_cells = self._df.shape[0] * self._df.shape[1]
        if total_cells == 0:
            return 0.0
        return round(float(self._df.isna().sum().sum() / total_cells), 4)
