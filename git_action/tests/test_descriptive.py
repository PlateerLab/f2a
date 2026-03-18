"""Descriptive statistics tests."""

from __future__ import annotations

import pandas as pd
import pytest

from f2a.core.schema import infer_schema
from f2a.stats.descriptive import DescriptiveStats
from f2a.stats.correlation import CorrelationStats
from f2a.stats.missing import MissingStats


class TestDescriptiveStats:
    """DescriptiveStats tests."""

    def test_summary_returns_dataframe(self, sample_mixed_df: pd.DataFrame) -> None:
        schema = infer_schema(sample_mixed_df)
        stats = DescriptiveStats(sample_mixed_df, schema)
        result = stats.summary()
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(sample_mixed_df.columns)

    def test_numeric_summary(self, sample_numeric_df: pd.DataFrame) -> None:
        schema = infer_schema(sample_numeric_df)
        stats = DescriptiveStats(sample_numeric_df, schema)
        result = stats.numeric_summary()
        assert isinstance(result, pd.DataFrame)
        assert "mean" in result.columns

    def test_summary_contains_expected_columns(self, sample_mixed_df: pd.DataFrame) -> None:
        schema = infer_schema(sample_mixed_df)
        stats = DescriptiveStats(sample_mixed_df, schema)
        result = stats.summary()
        assert "type" in result.columns
        assert "count" in result.columns
        assert "missing" in result.columns


class TestCorrelationStats:
    """CorrelationStats tests."""

    def test_pearson(self, sample_numeric_df: pd.DataFrame) -> None:
        schema = infer_schema(sample_numeric_df)
        corr = CorrelationStats(sample_numeric_df, schema)
        result = corr.pearson()
        assert isinstance(result, pd.DataFrame)
        assert result.shape[0] == result.shape[1]

    def test_spearman(self, sample_numeric_df: pd.DataFrame) -> None:
        schema = infer_schema(sample_numeric_df)
        corr = CorrelationStats(sample_numeric_df, schema)
        result = corr.spearman()
        assert isinstance(result, pd.DataFrame)


class TestMissingStats:
    """MissingStats tests."""

    def test_column_summary(self, sample_mixed_df: pd.DataFrame) -> None:
        schema = infer_schema(sample_mixed_df)
        miss = MissingStats(sample_mixed_df, schema)
        result = miss.column_summary()
        assert isinstance(result, pd.DataFrame)
        assert "missing_count" in result.columns

    def test_total_missing_ratio(self, sample_mixed_df: pd.DataFrame) -> None:
        schema = infer_schema(sample_mixed_df)
        miss = MissingStats(sample_mixed_df, schema)
        ratio = miss.total_missing_ratio()
        assert 0.0 <= ratio <= 1.0
