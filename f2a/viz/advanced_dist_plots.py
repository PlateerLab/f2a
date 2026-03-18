"""Advanced distribution visualization module.

Provides best-fit overlay plots, ECDF plots, power transform comparison,
and KDE bandwidth comparison plots.
"""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from scipy import stats as sp_stats

from f2a.viz.theme import DEFAULT_THEME, F2ATheme


class AdvancedDistPlotter:
    """Visualise advanced distribution analysis results.

    Args:
        theme: Visualisation theme.
    """

    def __init__(self, theme: F2ATheme | None = None) -> None:
        self._theme = theme or DEFAULT_THEME
        self._theme.apply()

    def best_fit_overlay(
        self,
        df: pd.DataFrame,
        best_fit_df: pd.DataFrame,
        max_cols: int = 9,
    ) -> plt.Figure:
        """Overlay best-fit distribution on histograms.

        Args:
            df: Original DataFrame with numeric columns.
            best_fit_df: DataFrame from AdvancedDistributionStats.best_fit().
            max_cols: Max columns to plot.
        """
        if best_fit_df.empty:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "No best-fit data", ha="center", va="center")
            return fig

        cols = list(best_fit_df.index[:max_cols])
        n = len(cols)
        ncols = min(3, n)
        nrows = (n + ncols - 1) // ncols

        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
        if n == 1:
            axes = [axes]
        else:
            axes = list(axes.flat)

        for idx, col in enumerate(cols):
            ax = axes[idx]
            series = df[col].dropna()
            if len(series) < 5:
                ax.set_visible(False)
                continue

            # Histogram
            ax.hist(series, bins=30, density=True, alpha=0.5, color="#3498db", label="Data")

            # Best-fit curve overlay
            row = best_fit_df.loc[col]
            dist_name = row["best_distribution"]
            try:
                dist = getattr(sp_stats, dist_name)
                params = dist.fit(series)
                x = np.linspace(series.min(), series.max(), 200)
                pdf = dist.pdf(x, *params)
                ax.plot(x, pdf, "r-", linewidth=2,
                        label=f"{dist_name} (AIC={row['aic']:.0f})")
            except Exception:
                pass

            ax.set_title(col, fontsize=10)
            ax.legend(fontsize=7)

        for idx in range(n, len(axes)):
            axes[idx].set_visible(False)

        fig.suptitle("Best-Fit Distribution Overlay", fontsize=self._theme.title_size + 2, y=1.02)
        fig.tight_layout()
        return fig

    def ecdf_plot(
        self,
        ecdf_data: dict[str, pd.DataFrame],
        max_cols: int = 9,
    ) -> plt.Figure:
        """Plot Empirical Cumulative Distribution Functions.

        Args:
            ecdf_data: Dictionary mapping column name to ECDF DataFrame.
            max_cols: Max columns to plot.
        """
        if not ecdf_data:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "No ECDF data", ha="center", va="center")
            return fig

        cols = list(ecdf_data.keys())[:max_cols]
        n = len(cols)
        ncols = min(3, n)
        nrows = (n + ncols - 1) // ncols

        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
        if n == 1:
            axes = [axes]
        else:
            axes = list(axes.flat)

        for idx, col in enumerate(cols):
            ax = axes[idx]
            edf = ecdf_data[col]
            ax.step(edf["x"], edf["ecdf"], where="post", color="#2980B9", linewidth=1.5)
            ax.fill_between(edf["x"], edf["ecdf"], step="post", alpha=0.15, color="#3498DB")
            ax.set_title(col, fontsize=10)
            ax.set_ylabel("ECDF")
            ax.set_ylim(0, 1.05)
            ax.axhline(y=0.5, color="#95A5A6", linestyle="--", alpha=0.5)

        for idx in range(n, len(axes)):
            axes[idx].set_visible(False)

        fig.suptitle("Empirical CDF", fontsize=self._theme.title_size + 2, y=1.02)
        fig.tight_layout()
        return fig

    def power_transform_plot(
        self,
        df: pd.DataFrame,
        power_df: pd.DataFrame,
        max_cols: int = 6,
    ) -> plt.Figure:
        """Compare original vs. power-transformed distributions.

        Args:
            df: Original DataFrame.
            power_df: DataFrame from power_transform_recommendation().
            max_cols: Max columns to show.
        """
        if power_df.empty:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "No power transform data", ha="center", va="center")
            return fig

        # Only show columns that need transformation
        need_transform = power_df[power_df["needs_transform"] == True]  # noqa: E712
        if need_transform.empty:
            need_transform = power_df.head(max_cols)
        cols = list(need_transform.index[:max_cols])
        n = len(cols)

        fig, axes = plt.subplots(n, 2, figsize=(10, 3.5 * n), squeeze=False)

        for idx, col in enumerate(cols):
            row = power_df.loc[col]
            series = df[col].dropna()
            if len(series) < 10:
                axes[idx][0].set_visible(False)
                axes[idx][1].set_visible(False)
                continue

            ax_orig = axes[idx][0]
            ax_trans = axes[idx][1]

            # Original
            ax_orig.hist(series, bins=30, color="#e74c3c", alpha=0.6)
            ax_orig.set_title(f"{col} (original, skew={row['original_skewness']:.2f})",
                              fontsize=9)

            # Transformed
            method = row["recommended_method"]
            if method == "box-cox" and (series > 0).all():
                try:
                    transformed, _ = sp_stats.boxcox(series.values)
                    ax_trans.hist(transformed, bins=30, color="#27ae60", alpha=0.6)
                    ax_trans.set_title(
                        f"{col} (Box-Cox, skew={row['transformed_skewness']:.2f})",
                        fontsize=9,
                    )
                except Exception:
                    ax_trans.text(0.5, 0.5, "Transform failed", ha="center", va="center")
            elif method == "yeo-johnson":
                try:
                    transformed, _ = sp_stats.yeojohnson(series.values)
                    ax_trans.hist(transformed, bins=30, color="#27ae60", alpha=0.6)
                    ax_trans.set_title(
                        f"{col} (Yeo-Johnson, skew={row['transformed_skewness']:.2f})",
                        fontsize=9,
                    )
                except Exception:
                    ax_trans.text(0.5, 0.5, "Transform failed", ha="center", va="center")
            else:
                ax_trans.text(0.5, 0.5, "No transform needed", ha="center", va="center",
                              transform=ax_trans.transAxes)

        fig.suptitle("Power Transform Comparison", fontsize=self._theme.title_size + 2, y=1.02)
        fig.tight_layout()
        return fig

    def jarque_bera_summary(self, jb_df: pd.DataFrame) -> plt.Figure:
        """Visualize Jarque-Bera test results as a bar chart.

        Args:
            jb_df: DataFrame from jarque_bera().
        """
        if jb_df.empty:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "No JB test data", ha="center", va="center")
            return fig

        fig, ax = plt.subplots(figsize=(max(8, len(jb_df) * 0.8), 5))

        cols = jb_df.index.tolist()
        p_vals = jb_df["p_value"].values
        colors = ["#27ae60" if p > 0.05 else "#e74c3c" for p in p_vals]

        bars = ax.barh(range(len(cols)), -np.log10(p_vals + 1e-15), color=colors)
        ax.set_yticks(range(len(cols)))
        ax.set_yticklabels(cols)
        ax.invert_yaxis()
        ax.set_xlabel("-log10(p-value)")
        ax.axvline(x=-np.log10(0.05), color="#f39c12", linestyle="--",
                    label="α = 0.05", alpha=0.7)
        ax.legend()
        ax.set_title("Jarque-Bera Normality Test", fontsize=self._theme.title_size)
        fig.tight_layout()
        return fig
