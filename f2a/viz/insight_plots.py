"""Insight-engine visualization module.

Provides charts that turn InsightEngine output into actionable visuals:

* **severity_bar** — horizontal bar chart of insights by severity.
* **category_sunburst** — category breakdown (tree-map fallback).
* **top_insights_table_fig** — top-N insights as a table figure.
* **action_items_summary** — action-item bubble chart.
"""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

from f2a.viz.theme import DEFAULT_THEME, F2ATheme

# Severity → (colour, z-order)
_SEV_PALETTE: dict[str, str] = {
    "critical": "#e74c3c",
    "warning": "#f39c12",
    "info": "#3498db",
    "opportunity": "#2ecc71",
}


class InsightPlotter:
    """Visualise auto-generated insights.

    Args:
        theme: Visualisation theme.
    """

    def __init__(self, theme: F2ATheme | None = None) -> None:
        self._theme = theme or DEFAULT_THEME
        self._theme.apply()

    # ── severity bar chart ────────────────────────────────

    def severity_bar(self, insights: list[dict[str, Any]]) -> plt.Figure:
        """Horizontal bar chart: insight count by severity.

        Args:
            insights: List of insight dicts (must contain ``severity`` key).

        Returns:
            matplotlib Figure.
        """
        if not insights:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.text(0.5, 0.5, "No insights generated", ha="center", va="center")
            return fig

        sev_order = ["critical", "warning", "info", "opportunity"]
        counts = pd.Series([i.get("severity", "info") for i in insights]).value_counts()
        counts = counts.reindex(sev_order).fillna(0).astype(int)

        fig, ax = plt.subplots(figsize=(8, 4))
        colors = [_SEV_PALETTE.get(s, "#95a5a6") for s in counts.index]
        bars = ax.barh(counts.index, counts.values, color=colors, edgecolor="white")

        for bar, val in zip(bars, counts.values):
            if val > 0:
                ax.text(
                    bar.get_width() + 0.3,
                    bar.get_y() + bar.get_height() / 2,
                    str(int(val)),
                    va="center",
                    fontweight="bold",
                )

        ax.set_xlabel("Count")
        ax.set_title("Insights by Severity", fontsize=self._theme.title_size)
        ax.invert_yaxis()
        fig.tight_layout()
        return fig

    # ── category breakdown (treemap-style) ────────────────

    def category_treemap(self, insights: list[dict[str, Any]]) -> plt.Figure:
        """Category-breakdown chart (squarified treemap approximation).

        Args:
            insights: List of insight dicts (must contain ``category`` key).

        Returns:
            matplotlib Figure.
        """
        if not insights:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "No insights", ha="center", va="center")
            return fig

        cats = pd.Series([i.get("category", "other") for i in insights]).value_counts()

        fig, ax = plt.subplots(figsize=(10, 6))
        palette = self._theme.get_colors(len(cats))
        wedges, texts, pcts = ax.pie(
            cats.values,
            labels=cats.index,
            autopct=lambda p: f"{p:.0f}%" if p > 5 else "",
            colors=palette,
            startangle=140,
            pctdistance=0.8,
        )
        for t in texts:
            t.set_fontsize(9)
        ax.set_title("Insights by Category", fontsize=self._theme.title_size)
        fig.tight_layout()
        return fig

    # ── top-N insights as a table figure ──────────────────

    def top_insights_table(
        self,
        insights: list[dict[str, Any]],
        n: int = 10,
    ) -> plt.Figure:
        """Render top-N insights as a matplotlib table figure.

        Args:
            insights: List of insight dicts, sorted by priority_score desc.
            n: Number of insights to show.

        Returns:
            matplotlib Figure.
        """
        if not insights:
            fig, ax = plt.subplots(figsize=(12, 2))
            ax.text(0.5, 0.5, "No insights", ha="center", va="center")
            ax.axis("off")
            return fig

        top = sorted(insights, key=lambda i: i.get("priority_score", 0), reverse=True)[:n]

        cell_text = []
        for rank, ins in enumerate(top, 1):
            cell_text.append([
                str(rank),
                ins.get("severity", "")[:4].upper(),
                ins.get("category", ""),
                ins.get("title", "")[:60],
                f"{ins.get('priority_score', 0):.1f}",
            ])

        col_labels = ["#", "Sev", "Category", "Title", "Score"]

        nrows = len(cell_text)
        fig_h = max(2.5, 0.45 * nrows + 1.2)
        fig, ax = plt.subplots(figsize=(14, fig_h))
        ax.axis("off")

        tbl = ax.table(
            cellText=cell_text,
            colLabels=col_labels,
            cellLoc="left",
            colWidths=[0.05, 0.07, 0.15, 0.60, 0.08],
            loc="center",
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(9)
        tbl.scale(1.0, 1.4)

        # Color header
        for j in range(len(col_labels)):
            tbl[0, j].set_facecolor("#34495e")
            tbl[0, j].set_text_props(color="white", fontweight="bold")

        # Severity-based row coloring
        for i, row in enumerate(cell_text, 1):
            sev = row[1].lower()[:4]
            color = _SEV_PALETTE.get(
                {"crit": "critical", "warn": "warning", "info": "info", "oppo": "opportunity"}.get(sev, "info"),
                "#ecf0f1",
            )
            for j in range(len(col_labels)):
                tbl[i, j].set_facecolor(color + "22")  # translucent

        ax.set_title(f"Top {n} Insights", fontsize=self._theme.title_size, pad=20)
        fig.tight_layout()
        return fig

    # ── action items summary ──────────────────────────────

    def action_items_chart(self, insights: list[dict[str, Any]]) -> plt.Figure:
        """Aggregated action-item frequency bar chart.

        Collects all ``action_items`` across insights and shows top-15 most
        common actions.

        Args:
            insights: List of insight dicts.

        Returns:
            matplotlib Figure.
        """
        actions: list[str] = []
        for ins in insights:
            actions.extend(ins.get("action_items", []))

        if not actions:
            fig, ax = plt.subplots(figsize=(8, 3))
            ax.text(0.5, 0.5, "No action items", ha="center", va="center")
            ax.axis("off")
            return fig

        action_counts = pd.Series(actions).value_counts().head(15)

        fig, ax = plt.subplots(figsize=(12, max(4, 0.4 * len(action_counts))))
        palette = self._theme.get_colors(len(action_counts))
        ax.barh(
            range(len(action_counts)),
            action_counts.values,
            color=palette,
            edgecolor="white",
        )
        ax.set_yticks(range(len(action_counts)))
        ax.set_yticklabels(
            [a[:70] for a in action_counts.index],
            fontsize=8,
        )
        ax.invert_yaxis()
        ax.set_xlabel("Frequency")
        ax.set_title("Recommended Actions", fontsize=self._theme.title_size)
        fig.tight_layout()
        return fig
