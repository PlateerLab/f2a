"""
Visualization module for f2a.

Charts are generated using matplotlib/seaborn and embedded as base64 PNG
in the HTML report. This module re-exports key plotting utilities.
"""

from f2a.viz.theme import apply_dark_theme, F2A_PALETTE
from f2a.viz.plots import (
    plot_correlation_heatmap,
    plot_distribution_grid,
    plot_missing_heatmap,
    plot_outlier_boxplots,
    plot_pca_variance,
    plot_quality_radar,
)

__all__ = [
    "apply_dark_theme",
    "F2A_PALETTE",
    "plot_correlation_heatmap",
    "plot_distribution_grid",
    "plot_missing_heatmap",
    "plot_outlier_boxplots",
    "plot_pca_variance",
    "plot_quality_radar",
]
