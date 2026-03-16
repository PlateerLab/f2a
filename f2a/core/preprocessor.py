"""Data preprocessing and data-quality detection module.

The :class:`Preprocessor` inspects raw data, detects quality issues, and
produces a lightly-cleaned copy suitable for downstream analysis.

Cleaning is *non-destructive* — the original DataFrame is never mutated.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from f2a.core.schema import DataSchema
from f2a.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class PreprocessingResult:
    """Results of the preprocessing step."""

    original_shape: tuple[int, int] = (0, 0)
    cleaned_shape: tuple[int, int] = (0, 0)

    # Detected issues
    constant_columns: list[str] = field(default_factory=list)
    duplicate_rows_count: int = 0
    high_missing_columns: list[dict[str, Any]] = field(default_factory=list)
    id_like_columns: list[str] = field(default_factory=list)
    mixed_type_columns: list[str] = field(default_factory=list)
    highly_correlated_pairs: list[tuple[str, str, float]] = field(default_factory=list)
    infinite_value_columns: list[str] = field(default_factory=list)

    # Applied transformations
    cleaning_log: list[str] = field(default_factory=list)

    # Quality indicators
    completeness: float = 1.0

    # The cleaned DataFrame (not shown in repr)
    cleaned_df: pd.DataFrame = field(default_factory=pd.DataFrame, repr=False)

    def summary_dict(self) -> dict[str, Any]:
        """Return a concise summary as a dictionary."""
        return {
            "original_rows": self.original_shape[0],
            "original_cols": self.original_shape[1],
            "cleaned_rows": self.cleaned_shape[0],
            "cleaned_cols": self.cleaned_shape[1],
            "constant_columns": len(self.constant_columns),
            "duplicate_rows": self.duplicate_rows_count,
            "high_missing_columns": len(self.high_missing_columns),
            "id_like_columns": len(self.id_like_columns),
            "mixed_type_columns": len(self.mixed_type_columns),
            "infinite_value_columns": len(self.infinite_value_columns),
            "completeness": round(self.completeness, 4),
            "cleaning_steps": len(self.cleaning_log),
        }

    def issues_table(self) -> pd.DataFrame:
        """Return a DataFrame summarising all detected issues."""
        rows: list[dict[str, Any]] = []

        for col in self.constant_columns:
            rows.append({"issue": "Constant column", "column": col, "detail": "Single unique value"})
        for item in self.high_missing_columns:
            rows.append({
                "issue": "High missing ratio",
                "column": item["column"],
                "detail": f"{item['missing_ratio'] * 100:.1f}%",
            })
        for col in self.id_like_columns:
            rows.append({"issue": "ID-like column", "column": col, "detail": "All values unique"})
        for col in self.mixed_type_columns:
            rows.append({"issue": "Mixed types", "column": col, "detail": "Multiple Python types"})
        for col in self.infinite_value_columns:
            rows.append({"issue": "Infinite values", "column": col, "detail": "Contains inf/-inf"})
        if self.duplicate_rows_count > 0:
            rows.append({
                "issue": "Duplicate rows",
                "column": "(all)",
                "detail": f"{self.duplicate_rows_count} rows",
            })

        return pd.DataFrame(rows) if rows else pd.DataFrame(columns=["issue", "column", "detail"])


class Preprocessor:
    """Analyse and lightly clean a DataFrame for optimal analysis.

    Args:
        df: Raw DataFrame.
        schema: Inferred :class:`DataSchema`.
    """

    def __init__(self, df: pd.DataFrame, schema: DataSchema) -> None:
        self._df = df
        self._schema = schema

    def run(self) -> PreprocessingResult:
        """Execute the full preprocessing pipeline.

        Returns:
            :class:`PreprocessingResult` with cleaned data and issue reports.
        """
        result = PreprocessingResult(
            original_shape=(len(self._df), len(self._df.columns)),
        )

        df = self._df.copy()

        # 1. Detect & remove constant columns
        result.constant_columns = self._detect_constant_columns(df)
        if result.constant_columns:
            result.cleaning_log.append(
                f"Removed {len(result.constant_columns)} constant column(s): "
                f"{', '.join(result.constant_columns[:5])}"
                + ("..." if len(result.constant_columns) > 5 else "")
            )
            df = df.drop(columns=result.constant_columns)

        # 2. Detect & remove exact duplicate rows
        result.duplicate_rows_count = int(df.duplicated().sum())
        if result.duplicate_rows_count > 0:
            pct = result.duplicate_rows_count / len(df) * 100
            result.cleaning_log.append(
                f"Removed {result.duplicate_rows_count} duplicate row(s) ({pct:.1f}%)"
            )
            df = df.drop_duplicates().reset_index(drop=True)

        # 3. Detect high-missing columns (>= 50 %)
        result.high_missing_columns = self._detect_high_missing(df, threshold=0.5)

        # 4. Detect ID-like columns (all unique values)
        result.id_like_columns = self._detect_id_columns(df)

        # 5. Detect mixed-type columns
        result.mixed_type_columns = self._detect_mixed_types(df)

        # 6. Detect infinite values in numeric columns
        result.infinite_value_columns = self._detect_infinite_values(df)
        if result.infinite_value_columns:
            for col in result.infinite_value_columns:
                count = int(np.isinf(df[col]).sum())
                result.cleaning_log.append(
                    f"Replaced {count} infinite value(s) in '{col}' with NaN"
                )
                df[col] = df[col].replace([np.inf, -np.inf], np.nan)

        # 7. Compute completeness
        total_cells = df.shape[0] * df.shape[1]
        if total_cells > 0:
            result.completeness = 1.0 - float(df.isna().sum().sum() / total_cells)
        else:
            result.completeness = 1.0

        result.cleaned_df = df
        result.cleaned_shape = (len(df), len(df.columns))

        logger.info(
            "Preprocessing complete: %s -> %s (%d steps)",
            result.original_shape,
            result.cleaned_shape,
            len(result.cleaning_log),
        )
        return result

    # ── Internal detectors ────────────────────────────────

    @staticmethod
    def _detect_constant_columns(df: pd.DataFrame) -> list[str]:
        """Find columns that have at most one unique non-null value."""
        return [col for col in df.columns if df[col].nunique(dropna=True) <= 1]

    @staticmethod
    def _detect_high_missing(df: pd.DataFrame, threshold: float = 0.5) -> list[dict[str, Any]]:
        """Find columns where the missing ratio exceeds *threshold*."""
        result: list[dict[str, Any]] = []
        for col in df.columns:
            ratio = float(df[col].isna().mean())
            if ratio >= threshold:
                result.append({"column": col, "missing_ratio": round(ratio, 4)})
        return sorted(result, key=lambda x: x["missing_ratio"], reverse=True)

    @staticmethod
    def _detect_id_columns(df: pd.DataFrame) -> list[str]:
        """Detect columns where every value is unique (likely an ID)."""
        if len(df) < 10:
            return []
        return [col for col in df.columns if df[col].nunique() == len(df)]

    @staticmethod
    def _detect_mixed_types(df: pd.DataFrame) -> list[str]:
        """Detect object columns containing more than one Python type."""
        mixed: list[str] = []
        for col in df.columns:
            if df[col].dtype == object:
                non_null = df[col].dropna()
                if len(non_null) > 0 and non_null.apply(type).nunique() > 1:
                    mixed.append(col)
        return mixed

    @staticmethod
    def _detect_infinite_values(df: pd.DataFrame) -> list[str]:
        """Detect numeric columns that contain ``inf`` / ``-inf``."""
        inf_cols: list[str] = []
        for col in df.select_dtypes(include="number").columns:
            if np.isinf(df[col]).any():
                inf_cols.append(col)
        return inf_cols
