"""
Core plot functions for f2a reports.

All functions return a matplotlib Figure object.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_VIZ = True
except ImportError:
    HAS_VIZ = False

from f2a.viz.theme import apply_dark_theme, F2A_PALETTE


def _ensure_viz():
    if not HAS_VIZ:
        raise ImportError("matplotlib and seaborn are required for visualization")
    apply_dark_theme()


# ─── Correlation Heatmap ─────────────────────────────────────────────

def plot_correlation_heatmap(
    matrix: list[list[float]],
    labels: list[str],
    title: str = "Correlation Matrix",
    figsize: tuple[int, int] = (10, 8),
) -> Any:
    """Plot a correlation matrix heatmap."""
    _ensure_viz()

    arr = np.array(matrix)
    fig, ax = plt.subplots(figsize=figsize)

    mask = np.triu(np.ones_like(arr, dtype=bool), k=1)
    sns.heatmap(
        arr,
        mask=mask,
        annot=True if len(labels) <= 15 else False,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
        linewidths=0.5,
        square=True,
    )
    ax.set_title(title, fontsize=14, pad=12)
    fig.tight_layout()
    return fig


# ─── Distribution Grid ──────────────────────────────────────────────

def plot_distribution_grid(
    columns_data: dict[str, list[float]],
    cols_per_row: int = 3,
    figsize_per_subplot: tuple[float, float] = (4, 3),
) -> Any:
    """Plot histograms for multiple numeric columns in a grid."""
    _ensure_viz()

    n = len(columns_data)
    if n == 0:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "No numeric columns", ha="center", va="center")
        return fig

    n_rows = (n + cols_per_row - 1) // cols_per_row
    w = figsize_per_subplot[0] * cols_per_row
    h = figsize_per_subplot[1] * n_rows
    fig, axes = plt.subplots(n_rows, cols_per_row, figsize=(w, h))
    axes = np.array(axes).flatten() if n > 1 else [axes]

    for idx, (col_name, values) in enumerate(columns_data.items()):
        ax = axes[idx]
        vals = [v for v in values if v is not None and not np.isnan(v)]
        if vals:
            ax.hist(vals, bins=30, color=F2A_PALETTE[idx % len(F2A_PALETTE)], alpha=0.75, edgecolor="none")
        ax.set_title(col_name, fontsize=10)
        ax.tick_params(labelsize=8)

    # Hide empty subplots
    for idx in range(n, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle("Distribution Overview", fontsize=14, y=1.02)
    fig.tight_layout()
    return fig


# ─── Missing Values Heatmap ─────────────────────────────────────────

def plot_missing_heatmap(
    missing_matrix: list[list[bool]],
    column_names: list[str],
    figsize: tuple[int, int] = (12, 6),
) -> Any:
    """Plot a binary heatmap of missing values."""
    _ensure_viz()

    arr = np.array(missing_matrix, dtype=float)
    fig, ax = plt.subplots(figsize=figsize)

    ax.imshow(arr.T, aspect="auto", cmap="YlOrRd", interpolation="nearest")
    ax.set_yticks(range(len(column_names)))
    ax.set_yticklabels(column_names, fontsize=8)
    ax.set_xlabel("Row index")
    ax.set_title("Missing Values Pattern", fontsize=14, pad=12)
    fig.tight_layout()
    return fig


# ─── Outlier Boxplots ───────────────────────────────────────────────

def plot_outlier_boxplots(
    columns_data: dict[str, list[float]],
    cols_per_row: int = 4,
    figsize_per_subplot: tuple[float, float] = (3, 4),
) -> Any:
    """Box plots for outlier visualization."""
    _ensure_viz()

    n = len(columns_data)
    if n == 0:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "No numeric columns", ha="center", va="center")
        return fig

    n_rows = (n + cols_per_row - 1) // cols_per_row
    w = figsize_per_subplot[0] * cols_per_row
    h = figsize_per_subplot[1] * n_rows
    fig, axes = plt.subplots(n_rows, cols_per_row, figsize=(w, h))
    axes = np.array(axes).flatten() if n > 1 else [axes]

    for idx, (col_name, values) in enumerate(columns_data.items()):
        ax = axes[idx]
        vals = [v for v in values if v is not None and not np.isnan(v)]
        if vals:
            bp = ax.boxplot(vals, vert=True, patch_artist=True)
            for patch in bp["boxes"]:
                patch.set_facecolor(F2A_PALETTE[idx % len(F2A_PALETTE)])
                patch.set_alpha(0.6)
        ax.set_title(col_name, fontsize=10)
        ax.tick_params(labelsize=8)

    for idx in range(n, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle("Outlier Detection (Boxplots)", fontsize=14, y=1.02)
    fig.tight_layout()
    return fig


# ─── PCA Variance Plot ──────────────────────────────────────────────

def plot_pca_variance(
    explained_variance: list[float],
    cumulative_variance: list[float],
    figsize: tuple[int, int] = (8, 5),
) -> Any:
    """Scree plot + cumulative variance explained."""
    _ensure_viz()

    n = len(explained_variance)
    x = list(range(1, n + 1))

    fig, ax1 = plt.subplots(figsize=figsize)

    ax1.bar(x, [v * 100 for v in explained_variance], color=F2A_PALETTE[0], alpha=0.7, label="Individual")
    ax1.set_xlabel("Principal Component")
    ax1.set_ylabel("Variance Explained (%)")

    ax2 = ax1.twinx()
    ax2.plot(x, [v * 100 for v in cumulative_variance], "o-", color=F2A_PALETTE[1], label="Cumulative")
    ax2.axhline(y=90, color=F2A_PALETTE[3], linestyle="--", alpha=0.5, label="90% Threshold")
    ax2.set_ylabel("Cumulative (%)")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right")

    ax1.set_title("PCA — Variance Explained", fontsize=14, pad=12)
    fig.tight_layout()
    return fig


# ─── Quality Radar Chart ────────────────────────────────────────────

def plot_quality_radar(
    dimensions: list[dict],
    figsize: tuple[int, int] = (7, 7),
) -> Any:
    """Radar chart for quality dimensions."""
    _ensure_viz()

    labels = [d["name"] for d in dimensions]
    values = [d["score"] for d in dimensions]

    n = len(labels)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    values_plot = values + [values[0]]
    angles += [angles[0]]

    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(polar=True))
    ax.fill(angles, values_plot, color=F2A_PALETTE[0], alpha=0.2)
    ax.plot(angles, values_plot, "o-", color=F2A_PALETTE[0], linewidth=2)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_title("Data Quality Dimensions", fontsize=14, pad=20)

    fig.tight_layout()
    return fig
