"""Dimension-reduction visualization module.

Provides charts missing from the original report for the **Dim. Reduction**
advanced sub-tab:

* **tsne_scatter** — t-SNE 2-D scatter.
* **umap_scatter** — UMAP 2-D scatter (graceful degradation if umap-learn
  unavailable).
* **explained_variance_curve** — PCA cumulative explained variance.
* **factor_loadings_heatmap** — per-component feature loadings.
* **feature_contribution_bar** — top feature contributions per component.
* **biplot** — PCA biplot with loading arrows.
"""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from f2a.viz.theme import DEFAULT_THEME, F2ATheme


class DimReductionPlotter:
    """Visualise dimension-reduction results (PCA / t-SNE / UMAP).

    Args:
        theme: Visualisation theme.
    """

    def __init__(self, theme: F2ATheme | None = None) -> None:
        self._theme = theme or DEFAULT_THEME
        self._theme.apply()

    # ── t-SNE scatter ─────────────────────────────────────

    def tsne_scatter(
        self,
        df: pd.DataFrame,
        numeric_cols: list[str],
        perplexity: float = 30.0,
        max_sample: int = 3000,
        color_col: str | None = None,
    ) -> plt.Figure:
        """2-D t-SNE scatter plot.

        Args:
            df: Source DataFrame.
            numeric_cols: Numeric column names to embed.
            perplexity: t-SNE perplexity parameter.
            max_sample: Maximum sample size for efficiency.
            color_col: Optional column name to colour points by.

        Returns:
            matplotlib Figure.
        """
        try:
            from sklearn.manifold import TSNE
            from sklearn.preprocessing import StandardScaler
        except ImportError:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "scikit-learn required", ha="center", va="center")
            return fig

        sub = df[numeric_cols].dropna()
        if len(sub) < 10:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "Insufficient data for t-SNE", ha="center", va="center")
            return fig

        if len(sub) > max_sample:
            sub = sub.sample(max_sample, random_state=42)

        X = StandardScaler().fit_transform(sub.values)
        perp = min(perplexity, max(5.0, len(X) / 4))
        emb = TSNE(n_components=2, perplexity=perp, random_state=42, init="pca").fit_transform(X)

        fig, ax = plt.subplots(figsize=(10, 8))

        if color_col and color_col in df.columns:
            c = df.loc[sub.index, color_col]
            if c.dtype.kind in ("i", "f"):
                scatter = ax.scatter(emb[:, 0], emb[:, 1], c=c, cmap="viridis",
                                     s=15, alpha=0.6, edgecolors="none")
                plt.colorbar(scatter, ax=ax, label=color_col)
            else:
                for cat in c.unique()[:12]:
                    mask = c == cat
                    ax.scatter(emb[mask, 0], emb[mask, 1], label=str(cat)[:20],
                              s=15, alpha=0.6)
                ax.legend(fontsize=7, bbox_to_anchor=(1.01, 1), loc="upper left")
        else:
            ax.scatter(emb[:, 0], emb[:, 1], s=10, alpha=0.5, c="#3498db")

        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")
        ax.set_title("t-SNE 2-D Embedding", fontsize=self._theme.title_size)
        fig.tight_layout()
        return fig

    # ── UMAP scatter ──────────────────────────────────────

    def umap_scatter(
        self,
        df: pd.DataFrame,
        numeric_cols: list[str],
        n_neighbors: int = 15,
        max_sample: int = 5000,
        color_col: str | None = None,
    ) -> plt.Figure:
        """2-D UMAP scatter plot.

        Falls back to t-SNE if umap-learn is not installed.

        Args:
            df: Source DataFrame.
            numeric_cols: Numeric column names.
            n_neighbors: UMAP neighbourhood parameter.
            max_sample: Maximum sample size.
            color_col: Optional column for colouring.

        Returns:
            matplotlib Figure.
        """
        try:
            from umap import UMAP
            from sklearn.preprocessing import StandardScaler
        except ImportError:
            # Graceful fallback
            return self.tsne_scatter(df, numeric_cols, max_sample=max_sample,
                                    color_col=color_col)

        sub = df[numeric_cols].dropna()
        if len(sub) < 10:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "Insufficient data for UMAP", ha="center", va="center")
            return fig

        if len(sub) > max_sample:
            sub = sub.sample(max_sample, random_state=42)

        X = StandardScaler().fit_transform(sub.values)
        emb = UMAP(n_components=2, n_neighbors=min(n_neighbors, len(X) - 1),
                    random_state=42).fit_transform(X)

        fig, ax = plt.subplots(figsize=(10, 8))

        if color_col and color_col in df.columns:
            c = df.loc[sub.index, color_col]
            if c.dtype.kind in ("i", "f"):
                scatter = ax.scatter(emb[:, 0], emb[:, 1], c=c, cmap="viridis",
                                     s=15, alpha=0.6, edgecolors="none")
                plt.colorbar(scatter, ax=ax, label=color_col)
            else:
                for cat in c.unique()[:12]:
                    mask = c == cat
                    ax.scatter(emb[mask, 0], emb[mask, 1], label=str(cat)[:20],
                              s=15, alpha=0.6)
                ax.legend(fontsize=7, bbox_to_anchor=(1.01, 1), loc="upper left")
        else:
            ax.scatter(emb[:, 0], emb[:, 1], s=10, alpha=0.5, c="#2ecc71")

        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
        ax.set_title("UMAP 2-D Embedding", fontsize=self._theme.title_size)
        fig.tight_layout()
        return fig

    # ── cumulative explained variance ─────────────────────

    def explained_variance_curve(self, pca_result: dict[str, Any]) -> plt.Figure:
        """PCA cumulative explained variance curve.

        Args:
            pca_result: Dict from ``PCAAnalysis.summary()`` containing
                ``explained_variance_df``.

        Returns:
            matplotlib Figure.
        """
        ev_df = pca_result.get("explained_variance_df")
        if ev_df is None or (isinstance(ev_df, pd.DataFrame) and ev_df.empty):
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "No PCA variance data", ha="center", va="center")
            return fig

        if not isinstance(ev_df, pd.DataFrame):
            ev_df = pd.DataFrame(ev_df)

        fig, ax = plt.subplots(figsize=(10, 5))

        components = range(1, len(ev_df) + 1)
        individual = ev_df.get("explained_variance_ratio",
                               ev_df.get("variance_ratio", pd.Series(dtype=float)))
        cumulative = ev_df.get("cumulative_variance",
                               individual.cumsum() if len(individual) else pd.Series(dtype=float))

        ax.bar(components, individual, color="#3498db", alpha=0.7, label="Individual")
        ax.plot(components, cumulative, "ro-", linewidth=2, markersize=5, label="Cumulative")
        ax.axhline(y=0.95, color="#e74c3c", linestyle="--", alpha=0.5, label="95% threshold")
        ax.set_xlabel("Component")
        ax.set_ylabel("Explained Variance Ratio")
        ax.set_title("PCA Explained Variance", fontsize=self._theme.title_size)
        ax.legend(fontsize=9)
        ax.set_xticks(list(components))
        fig.tight_layout()
        return fig

    # ── factor loadings heatmap ───────────────────────────

    def factor_loadings_heatmap(self, pca_result: dict[str, Any]) -> plt.Figure:
        """Heatmap of PCA factor loadings (components × features).

        Args:
            pca_result: Dict with ``loadings_df`` (components as rows,
                features as columns).

        Returns:
            matplotlib Figure.
        """
        loadings = pca_result.get("loadings_df")
        if loadings is None or (isinstance(loadings, pd.DataFrame) and loadings.empty):
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "No loadings data", ha="center", va="center")
            return fig

        if not isinstance(loadings, pd.DataFrame):
            loadings = pd.DataFrame(loadings)

        # Limit to first 15 features × 8 components for readability
        loadings = loadings.iloc[:8, :15]

        fig, ax = plt.subplots(figsize=(max(8, 0.6 * loadings.shape[1] + 2),
                                        max(4, 0.6 * loadings.shape[0] + 2)))
        sns.heatmap(
            loadings.astype(float),
            ax=ax,
            cmap="RdBu_r",
            center=0,
            annot=True,
            fmt=".2f",
            linewidths=0.5,
            cbar_kws={"label": "Loading"},
        )
        ax.set_title("PCA Factor Loadings", fontsize=self._theme.title_size)
        ax.set_ylabel("Component")
        ax.set_xlabel("Feature")
        fig.tight_layout()
        return fig

    # ── top feature contributions per component ───────────

    def feature_contribution_bar(
        self,
        pca_result: dict[str, Any],
        n_components: int = 3,
        n_features: int = 10,
    ) -> plt.Figure:
        """Bar chart of top contributing features per PCA component.

        Args:
            pca_result: Dict with ``loadings_df``.
            n_components: Number of components to show.
            n_features: Number of features per component.

        Returns:
            matplotlib Figure.
        """
        loadings = pca_result.get("loadings_df")
        if loadings is None or (isinstance(loadings, pd.DataFrame) and loadings.empty):
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "No loadings data", ha="center", va="center")
            return fig

        if not isinstance(loadings, pd.DataFrame):
            loadings = pd.DataFrame(loadings)

        n_comp = min(n_components, len(loadings))
        fig, axes = plt.subplots(1, n_comp, figsize=(5 * n_comp, 5))
        if n_comp == 1:
            axes = [axes]

        palette = self._theme.get_colors(2)

        for idx in range(n_comp):
            ax = axes[idx]
            row = loadings.iloc[idx].abs().sort_values(ascending=False).head(n_features)
            signs = loadings.iloc[idx].loc[row.index]
            colors = [palette[0] if s >= 0 else palette[1] for s in signs]

            ax.barh(range(len(row)), row.values, color=colors, edgecolor="white")
            ax.set_yticks(range(len(row)))
            ax.set_yticklabels(row.index, fontsize=8)
            ax.invert_yaxis()
            ax.set_xlabel("|Loading|")
            ax.set_title(f"PC{idx + 1}", fontsize=self._theme.title_size - 1)

        fig.suptitle("Top Feature Contributions", fontsize=self._theme.title_size + 1, y=1.02)
        fig.tight_layout()
        return fig

    # ── PCA biplot ────────────────────────────────────────

    def biplot(
        self,
        df: pd.DataFrame,
        numeric_cols: list[str],
        n_arrows: int = 8,
        max_sample: int = 2000,
    ) -> plt.Figure:
        """PCA biplot: scatter of PC1 vs PC2 with loading arrows.

        Args:
            df: Source DataFrame.
            numeric_cols: Numeric column names.
            n_arrows: Number of loading arrows to draw.
            max_sample: Maximum sample size.

        Returns:
            matplotlib Figure.
        """
        try:
            from sklearn.decomposition import PCA
            from sklearn.preprocessing import StandardScaler
        except ImportError:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "scikit-learn required", ha="center", va="center")
            return fig

        sub = df[numeric_cols].dropna()
        if len(sub) < 10 or len(numeric_cols) < 2:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "Insufficient data for biplot", ha="center", va="center")
            return fig

        if len(sub) > max_sample:
            sub = sub.sample(max_sample, random_state=42)

        X = StandardScaler().fit_transform(sub.values)
        pca = PCA(n_components=2)
        scores = pca.fit_transform(X)

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.scatter(scores[:, 0], scores[:, 1], s=10, alpha=0.3, c="#3498db")

        # Loading arrows
        loadings = pca.components_.T  # (features, 2)
        scale_factor = max(abs(scores[:, 0]).max(), abs(scores[:, 1]).max()) * 0.9
        magnitudes = np.sqrt((loadings ** 2).sum(axis=1))
        top_idx = magnitudes.argsort()[-n_arrows:]

        for i in top_idx:
            lx, ly = loadings[i] * scale_factor
            ax.annotate(
                numeric_cols[i],
                xy=(lx, ly),
                xytext=(0, 0),
                arrowprops=dict(arrowstyle="->", color="#e74c3c", lw=1.5),
                fontsize=8,
                color="#e74c3c",
                ha="center",
            )

        var1 = pca.explained_variance_ratio_[0] * 100
        var2 = pca.explained_variance_ratio_[1] * 100
        ax.set_xlabel(f"PC1 ({var1:.1f}%)")
        ax.set_ylabel(f"PC2 ({var2:.1f}%)")
        ax.set_title("PCA Biplot", fontsize=self._theme.title_size)
        ax.axhline(0, color="grey", linewidth=0.5, alpha=0.5)
        ax.axvline(0, color="grey", linewidth=0.5, alpha=0.5)
        fig.tight_layout()
        return fig
