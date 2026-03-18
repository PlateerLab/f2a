"""Visualization tests."""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")  # Render without GUI during tests

import matplotlib.pyplot as plt
import pandas as pd
import pytest

from f2a.core.schema import infer_schema
from f2a.viz.plots import BasicPlotter
from f2a.viz.corr_plots import CorrelationPlotter
from f2a.viz.missing_plots import MissingPlotter


class TestBasicPlotter:
    """BasicPlotter tests."""

    def test_histograms(self, sample_numeric_df: pd.DataFrame) -> None:
        schema = infer_schema(sample_numeric_df)
        plotter = BasicPlotter(sample_numeric_df, schema)
        fig = plotter.histograms()
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_boxplots(self, sample_numeric_df: pd.DataFrame) -> None:
        schema = infer_schema(sample_numeric_df)
        plotter = BasicPlotter(sample_numeric_df, schema)
        fig = plotter.boxplots()
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_bar_charts(self, sample_mixed_df: pd.DataFrame) -> None:
        schema = infer_schema(sample_mixed_df)
        plotter = BasicPlotter(sample_mixed_df, schema)
        fig = plotter.bar_charts()
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestCorrelationPlotter:
    """CorrelationPlotter tests."""

    def test_heatmap(self, sample_numeric_df: pd.DataFrame) -> None:
        schema = infer_schema(sample_numeric_df)
        plotter = CorrelationPlotter(sample_numeric_df, schema)
        fig = plotter.heatmap()
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestMissingPlotter:
    """MissingPlotter tests."""

    def test_bar(self, sample_mixed_df: pd.DataFrame) -> None:
        schema = infer_schema(sample_mixed_df)
        plotter = MissingPlotter(sample_mixed_df, schema)
        fig = plotter.bar()
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_matrix(self, sample_mixed_df: pd.DataFrame) -> None:
        schema = infer_schema(sample_mixed_df)
        plotter = MissingPlotter(sample_mixed_df, schema)
        fig = plotter.matrix()
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
