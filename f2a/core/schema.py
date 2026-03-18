"""Data schema inference and management."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from f2a.utils.type_inference import ColumnType, infer_all_types


@dataclass
class ColumnInfo:
    """Metadata for an individual column."""

    name: str
    dtype: str
    inferred_type: ColumnType
    n_unique: int
    n_missing: int
    missing_ratio: float


@dataclass
class DataSchema:
    """Schema information for an entire DataFrame."""

    n_rows: int
    n_cols: int
    columns: list[ColumnInfo] = field(default_factory=list)
    memory_usage_mb: float = 0.0

    @property
    def numeric_columns(self) -> list[str]:
        """List of numeric column names."""
        return [c.name for c in self.columns if c.inferred_type == ColumnType.NUMERIC]

    @property
    def categorical_columns(self) -> list[str]:
        """List of categorical column names."""
        return [c.name for c in self.columns if c.inferred_type == ColumnType.CATEGORICAL]

    @property
    def text_columns(self) -> list[str]:
        """List of text column names."""
        return [c.name for c in self.columns if c.inferred_type == ColumnType.TEXT]

    @property
    def datetime_columns(self) -> list[str]:
        """List of datetime column names."""
        return [c.name for c in self.columns if c.inferred_type == ColumnType.DATETIME]

    def summary_dict(self) -> dict[str, str | int | float]:
        """Return schema summary as a dictionary."""
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
    """Infer schema from a DataFrame.

    Args:
        df: Target DataFrame to analyze.

    Returns:
        Inferred :class:`DataSchema`.
    """
    type_map = infer_all_types(df)
    columns: list[ColumnInfo] = []

    for col in df.columns:
        n_missing = int(df[col].isna().sum())
        try:
            n_unique = int(df[col].nunique())
        except TypeError:
            # Column contains unhashable types (e.g. numpy arrays, lists)
            n_unique = len(df[col].dropna())
        columns.append(
            ColumnInfo(
                name=col,
                dtype=str(df[col].dtype),
                inferred_type=type_map[col],
                n_unique=n_unique,
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
