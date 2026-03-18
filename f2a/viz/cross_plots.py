"""Cross-analysis visualization module.

Provides charts that surface cross-dimensional patterns discovered by
``CrossAnalysis``:

* **anomaly_by_cluster_bar** — per-cluster anomaly rate comparison.
* **missing_correlation_heatmap** — correlation of missingness indicators.
* **simpson_paradox_highlight** — Simpson's paradox direction-reversal plot.
* **importance_vs_missing_scatter** — feature importance vs. missing rate.
* **unified_2d_scatter** — 2-D embedding coloured by cluster + anomaly.
"""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns

from f2a.viz.theme import DEFAULT_THEME, F2ATheme


class CrossPlotter:
    """Visualise cross-dimensional analysis results.

    Args:
        theme: Visualisation theme.
    """

    def __init__(self, theme: F2ATheme | None = None) -> None:
        self._theme = theme or DEFAULT_THEME
        self._theme.apply()

    # ── anomaly by cluster ────────────────────────────────

    def anomaly_by_cluster_bar(self, cross_result: dict[str, Any]) -> plt.Figure:
        """Grouped bar chart of anomaly rates per cluster.

        Args:
            cross_result: Dict from ``CrossAnalysis.outlier_by_cluster()``.
        """
        df = cross_result.get("per_cluster")
        if df is None or (isinstance(df, pd.DataFrame) and df.empty):
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "No cluster-anomaly data", ha="center", va="center")
            return fig

        if isinstance(df, pd.DataFrame):
            data = df
        else:
            data = pd.DataFrame(df)

        fig, ax = plt.subplots(figsize=(10, 5))

        clusters = data.get("cluster", data.index if "cluster" not in data.columns else data["cluster"])
        anomaly_rates = data.get("anomaly_rate", [0] * len(clusters))

        colors = ["#e74c3c" if r > 0.15 else "#f39c12" if r > 0.05 else "#2ecc71"
                  for r in anomaly_rates]
        ax.bar(range(len(clusters)), anomaly_rates, color=colors, edgecolor="white")
        ax.set_xticks(range(len(clusters)))
        ax.set_xticklabels([f"C{c}" for c in clusters], fontsize=9)
        ax.set_ylabel("Anomaly Rate")
        ax.set_title("Anomaly Rate by Cluster", fontsize=self._theme.title_size)
        ax.axhline(y=float(np.mean(list(anomaly_rates))), color="#7f8c8d",
                   linestyle="--", alpha=0.6, label="Mean")
        ax.legend()
        fig.tight_layout()
        return fig

    # ── missing-correlation heatmap ───────────────────────

    def missing_correlation_heatmap(self, cross_result: dict[str, Any]) -> plt.Figure:
        """Heatmap of point-biserial correlations between missingness flags.

        Args:
            cross_result: Dict from ``CrossAnalysis.missing_correlation()``.
        """
        df = cross_result.get("correlations")
        if df is None or (isinstance(df, pd.DataFrame) and df.empty):
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "No missing-correlation data", ha="center", va="center")
            return fig

        if not isinstance(df, pd.DataFrame):
            df = pd.DataFrame(df)

        # Build a pivot: missing_col × numeric_col → correlation
        if "missing_col" in df.columns and "numeric_col" in df.columns and "correlation" in df.columns:
            pivot = df.pivot_table(
                index="missing_col",
                columns="numeric_col",
                values="correlation",
                aggfunc="first",
            )
        else:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "Unexpected data format", ha="center", va="center")
            return fig

        fig, ax = plt.subplots(figsize=(max(8, 0.6 * pivot.shape[1] + 2),
                                        max(5, 0.5 * pivot.shape[0] + 2)))
        sns.heatmap(
            pivot.astype(float),
            ax=ax,
            cmap="RdBu_r",
            center=0,
            annot=True,
            fmt=".2f",
            linewidths=0.5,
            cbar_kws={"label": "Point-biserial r"},
        )
        ax.set_title("Missingness ↔ Numeric Correlation",
                     fontsize=self._theme.title_size)
        fig.tight_layout()
        return fig

    # ── Simpson's paradox highlight ───────────────────────

    def simpson_paradox_scatter(self, cross_result: dict[str, Any]) -> plt.Figure:
        """Scatter plot highlighting Simpson's paradox direction reversals.

        Shows overall correlation line vs. per-cluster regression lines for
        the most prominent reversal.

        Args:
            cross_result: Dict from ``CrossAnalysis.simpson_paradox()``.
        """
        cases = cross_result.get("cases", [])
        if not cases:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "No Simpson's paradox detected", ha="center", va="center")
            return fig

        # Show the strongest case
        case = cases[0] if isinstance(cases, list) else cases

        fig, ax = plt.subplots(figsize=(10, 6))

        overall_r = case.get("overall_corr", 0)
        col_a = case.get("col_a", "X")
        col_b = case.get("col_b", "Y")
        cluster_corrs = case.get("cluster_correlations", {})

        # Draw overall trend arrow
        ax.annotate(
            "",
            xy=(0.85, 0.5 + 0.3 * np.sign(overall_r)),
            xytext=(0.15, 0.5 - 0.3 * np.sign(overall_r)),
            xycoords="axes fraction",
            arrowprops=dict(arrowstyle="->", color="#e74c3c", lw=3, alpha=0.5),
        )
        ax.text(0.5, 0.5 + 0.35 * np.sign(overall_r),
                f"Overall r = {overall_r:+.3f}",
                transform=ax.transAxes, ha="center", fontsize=12,
                color="#e74c3c", fontweight="bold")

        # Cluster-level arrows
        palette = self._theme.get_colors(max(len(cluster_corrs), 1))
        y_start = 0.2
        for idx, (cid, r) in enumerate(cluster_corrs.items()):
            y = y_start + idx * 0.08
            color = palette[idx % len(palette)]
            direction = "→" if r > 0 else "←"
            ax.text(0.15, y, f"Cluster {cid}: r = {r:+.3f} {direction}",
                    transform=ax.transAxes, fontsize=10, color=color)

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        ax.set_title(
            f"Simpson's Paradox: {col_a} vs {col_b}",
            fontsize=self._theme.title_size,
        )
        fig.tight_layout()
        return fig

    # ── importance vs. missing scatter ────────────────────

    def importance_vs_missing_scatter(self, cross_result: dict[str, Any]) -> plt.Figure:
        """Scatter: feature importance (x) vs. missing rate (y).

        High-importance + high-missing = information loss risk.

        Args:
            cross_result: Dict from ``CrossAnalysis.importance_vs_missing()``.
        """
        df = cross_result.get("risk_table")
        if df is None or (isinstance(df, pd.DataFrame) and df.empty):
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "No importance–missing data", ha="center", va="center")
            return fig

        if not isinstance(df, pd.DataFrame):
            df = pd.DataFrame(df)

        fig, ax = plt.subplots(figsize=(10, 7))

        importance = df.get("importance", pd.Series(dtype=float))
        missing_rate = df.get("missing_rate", pd.Series(dtype=float))
        risk = df.get("risk_score", importance * missing_rate if len(importance) else pd.Series(dtype=float))

        scatter = ax.scatter(
            importance,
            missing_rate,
            c=risk,
            cmap="YlOrRd",
            s=80,
            edgecolors="white",
            linewidths=0.5,
            alpha=0.8,
        )
        plt.colorbar(scatter, ax=ax, label="Risk Score")

        # Label top-risk points
        if "column" in df.columns:
            top_risk = df.nlargest(5, "risk_score") if "risk_score" in df.columns else df.head(5)
            for _, row in top_risk.iterrows():
                ax.annotate(
                    row["column"],
                    (row.get("importance", 0), row.get("missing_rate", 0)),
                    fontsize=8,
                    alpha=0.8,
                    xytext=(5, 5),
                    textcoords="offset points",
                )

        ax.set_xlabel("Feature Importance")
        ax.set_ylabel("Missing Rate")
        ax.set_title("Importance vs. Missing (Information Loss Risk)",
                     fontsize=self._theme.title_size)

        # Danger zone shading
        ax.axhspan(0.3, 1.0, xmin=0.5, xmax=1.0, alpha=0.08, color="red")
        ax.text(0.78, 0.85, "⚠ Danger Zone", transform=ax.transAxes,
                fontsize=10, color="#e74c3c", alpha=0.6)

        fig.tight_layout()
        return fig

    # ── unified 2-D embedding scatter ─────────────────────

    def unified_2d_scatter(self, cross_result: dict[str, Any]) -> plt.Figure:
        """2-D embedding scatter coloured by cluster ID, anomaly shape.

        Args:
            cross_result: Dict from ``CrossAnalysis.unified_2d_embedding()``.
        """
        df = cross_result.get("embedding_df")
        if df is None or (isinstance(df, pd.DataFrame) and df.empty):
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "No embedding data", ha="center", va="center")
            return fig

        if not isinstance(df, pd.DataFrame):
            df = pd.DataFrame(df)

        fig, ax = plt.subplots(figsize=(10, 8))

        x_col = "x" if "x" in df.columns else df.columns[0]
        y_col = "y" if "y" in df.columns else df.columns[1]
        cluster_col = "cluster" if "cluster" in df.columns else None
        anomaly_col = "is_anomaly" if "is_anomaly" in df.columns else None

        clusters = df[cluster_col].values if cluster_col else np.zeros(len(df))
        unique_clusters = sorted(set(clusters))
        palette = self._theme.get_colors(max(len(unique_clusters), 1))

        # Normal points
        for idx, cid in enumerate(unique_clusters):
            mask = clusters == cid
            if anomaly_col:
                mask = mask & ~df[anomaly_col].astype(bool).values
            ax.scatter(
                df.loc[mask, x_col],
                df.loc[mask, y_col],
                c=[palette[idx % len(palette)]],
                label=f"Cluster {cid}",
                s=30,
                alpha=0.6,
                edgecolors="none",
            )

        # Anomaly points overlaid
        if anomaly_col and df[anomaly_col].any():
            anom_mask = df[anomaly_col].astype(bool).values
            ax.scatter(
                df.loc[anom_mask, x_col],
                df.loc[anom_mask, y_col],
                c="none",
                edgecolors="#e74c3c",
                marker="x",
                s=50,
                linewidths=1.5,
                label="Anomaly",
                zorder=5,
            )

        ax.set_xlabel("Dimension 1")
        ax.set_ylabel("Dimension 2")
        method = cross_result.get("method", "Embedding")
        ax.set_title(f"Unified {method} (Cluster + Anomaly)",
                     fontsize=self._theme.title_size)
        ax.legend(fontsize=8, bbox_to_anchor=(1.01, 1), loc="upper left")
        fig.tight_layout()
        return fig
