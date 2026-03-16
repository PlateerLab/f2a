"""Automatic data type inference utilities."""

from __future__ import annotations

from enum import Enum

import pandas as pd


class ColumnType(str, Enum):
    """Column type classification."""

    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    TEXT = "text"
    DATETIME = "datetime"
    BOOLEAN = "boolean"


# Max unique value ratio to consider a column categorical
_CATEGORICAL_RATIO_THRESHOLD = 0.05  # 5%
# Max absolute unique count to consider a column categorical
_CATEGORICAL_UNIQUE_THRESHOLD = 50
# Min average string length to consider a column text
_TEXT_LENGTH_THRESHOLD = 50


def infer_column_type(series: pd.Series) -> ColumnType:
    """Infer the semantic type of a single column.

    Args:
        series: Target pandas Series to analyze.

    Returns:
        Inferred :class:`ColumnType`.
    """
    # Boolean check
    try:
        if series.dtype == "bool" or set(series.dropna().unique()) <= {True, False, 0, 1}:
            return ColumnType.BOOLEAN
    except TypeError:
        # Column contains unhashable types (e.g. numpy arrays, lists)
        return ColumnType.TEXT

    # Datetime check
    if pd.api.types.is_datetime64_any_dtype(series):
        return ColumnType.DATETIME

    # Numeric check
    if pd.api.types.is_numeric_dtype(series):
        n_unique = series.nunique()
        n_total = len(series)
        # Treat as categorical if very few unique values
        if n_unique <= 10 and n_total > 100:
            return ColumnType.CATEGORICAL
        return ColumnType.NUMERIC

    # String types
    if pd.api.types.is_string_dtype(series) or pd.api.types.is_object_dtype(series):
        n_unique = series.nunique()
        n_total = len(series.dropna())

        if n_total == 0:
            return ColumnType.TEXT

        # Attempt datetime parsing
        try:
            pd.to_datetime(series.dropna().head(20))
            return ColumnType.DATETIME
        except (ValueError, TypeError):
            pass

        # Determine text vs categorical by unique ratio and string length
        ratio = n_unique / n_total if n_total > 0 else 1.0
        avg_len = series.dropna().astype(str).str.len().mean()

        if avg_len > _TEXT_LENGTH_THRESHOLD:
            return ColumnType.TEXT
        if n_unique <= _CATEGORICAL_UNIQUE_THRESHOLD or ratio <= _CATEGORICAL_RATIO_THRESHOLD:
            return ColumnType.CATEGORICAL
        return ColumnType.TEXT

    return ColumnType.TEXT


def infer_all_types(df: pd.DataFrame) -> dict[str, ColumnType]:
    """Infer types for all columns in a DataFrame.

    Args:
        df: Target DataFrame to analyze.

    Returns:
        Column name → :class:`ColumnType` mapping.
    """
    return {col: infer_column_type(df[col]) for col in df.columns}
