"""데이터 스키마 추론 및 관리."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from f2a.utils.type_inference import ColumnType, infer_all_types


@dataclass
class ColumnInfo:
    """개별 컬럼에 대한 메타 정보."""

    name: str
    dtype: str
    inferred_type: ColumnType
    n_unique: int
    n_missing: int
    missing_ratio: float


@dataclass
class DataSchema:
    """데이터프레임 전체의 스키마 정보."""

    n_rows: int
    n_cols: int
    columns: list[ColumnInfo] = field(default_factory=list)
    memory_usage_mb: float = 0.0

    @property
    def numeric_columns(self) -> list[str]:
        """수치형 컬럼 이름 목록."""
        return [c.name for c in self.columns if c.inferred_type == ColumnType.NUMERIC]

    @property
    def categorical_columns(self) -> list[str]:
        """범주형 컬럼 이름 목록."""
        return [c.name for c in self.columns if c.inferred_type == ColumnType.CATEGORICAL]

    @property
    def text_columns(self) -> list[str]:
        """텍스트 컬럼 이름 목록."""
        return [c.name for c in self.columns if c.inferred_type == ColumnType.TEXT]

    @property
    def datetime_columns(self) -> list[str]:
        """일시 컬럼 이름 목록."""
        return [c.name for c in self.columns if c.inferred_type == ColumnType.DATETIME]

    def summary_dict(self) -> dict[str, str | int | float]:
        """스키마 요약을 딕셔너리로 반환합니다."""
        return {
            "rows": self.n_rows,
            "columns": self.n_cols,
            "numeric": len(self.numeric_columns),
            "categorical": len(self.categorical_columns),
            "text": len(self.text_columns),
            "datetime": len(self.datetime_columns),
            "memory_mb": round(self.memory_usage_mb, 2),
        }


def infer_schema(df: pd.DataFrame) -> DataSchema:
    """DataFrame으로부터 스키마를 추론합니다.

    Args:
        df: 분석 대상 DataFrame.

    Returns:
        추론된 :class:`DataSchema`.
    """
    type_map = infer_all_types(df)
    columns: list[ColumnInfo] = []

    for col in df.columns:
        n_missing = int(df[col].isna().sum())
        columns.append(
            ColumnInfo(
                name=col,
                dtype=str(df[col].dtype),
                inferred_type=type_map[col],
                n_unique=int(df[col].nunique()),
                n_missing=n_missing,
                missing_ratio=round(n_missing / len(df), 4) if len(df) > 0 else 0.0,
            )
        )

    memory_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)

    return DataSchema(
        n_rows=len(df),
        n_cols=len(df.columns),
        columns=columns,
        memory_usage_mb=round(memory_mb, 2),
    )
