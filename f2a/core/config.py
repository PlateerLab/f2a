"""Analysis configuration module.

Provides :class:`AnalysisConfig` to control which analysis steps are executed.
All steps are enabled by default.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AnalysisConfig:
    """Configuration for the f2a analysis pipeline.

    All analysis steps are enabled by default.  Set individual flags to
    ``False`` to skip specific analyses.

    Example::

        import f2a
        from f2a import AnalysisConfig

        # Run only descriptive stats and correlation
        config = AnalysisConfig(
            distribution=False,
            outlier=False,
            categorical=False,
            feature_importance=False,
            pca=False,
            duplicates=False,
        )
        report = f2a.analyze("data.csv", config=config)
    """

    # ── Analysis toggles ──────────────────────────────────
    preprocessing: bool = True
    descriptive: bool = True
    distribution: bool = True
    correlation: bool = True
    outlier: bool = True
    categorical: bool = True
    feature_importance: bool = True
    pca: bool = True
    duplicates: bool = True
    quality_score: bool = True

    # ── Visualization toggle ──────────────────────────────
    visualizations: bool = True

    # ── Sub-options ───────────────────────────────────────
    outlier_method: str = "iqr"
    """``"iqr"`` (default) or ``"zscore"``."""

    outlier_threshold: float = 1.5
    """IQR multiplier (default 1.5) or z-score cutoff (use 3.0 with zscore)."""

    correlation_threshold: float = 0.9
    """Absolute correlation coefficient threshold for high-correlation warnings."""

    pca_max_components: int = 10
    """Maximum number of PCA components to compute."""

    max_categories: int = 50
    """Maximum categories to display in categorical charts."""

    max_plot_columns: int = 20
    """Maximum columns per plot grid (prevents overly large figures)."""

    # ── Advanced analysis ─────────────────────────────────
    advanced: bool = True
    """Enable the Advanced analysis tab (clustering, anomaly, etc.)."""

    advanced_distribution: bool = True
    """Best-fit distribution, power transform, Jarque-Bera, ECDF."""

    advanced_correlation: bool = True
    """Partial correlation, MI matrix, bootstrap CI, network graph."""

    clustering: bool = True
    """K-Means, DBSCAN, hierarchical clustering."""

    advanced_dimreduction: bool = True
    """t-SNE, UMAP (optional), Factor Analysis."""

    feature_insights: bool = True
    """Interaction, monotonic, binning, cardinality, leakage detection."""

    advanced_anomaly: bool = True
    """Isolation Forest, LOF, Mahalanobis, consensus."""

    statistical_tests: bool = True
    """Levene, Kruskal-Wallis, Mann-Whitney, goodness-of-fit, Grubbs."""

    data_profiling: bool = True
    """Automated insights, type recommendation, health dashboard."""

    # ── Enhancement modules (v2) ──────────────────────────
    insight_engine: bool = True
    """Auto-generate prioritised natural-language insights."""

    cross_analysis: bool = True
    """Cross-dimensional analysis (outlier × cluster, Simpson, etc.)."""

    column_role: bool = True
    """Auto-detect column semantic roles (ID, target, feature, …)."""

    ml_readiness: bool = True
    """Multi-dimensional ML-readiness scoring."""

    # ── Advanced sub-options ──────────────────────────────
    max_cluster_k: int = 10
    """Maximum k for K-Means elbow search."""

    tsne_perplexity: float = 30.0
    """t-SNE perplexity parameter."""

    bootstrap_iterations: int = 1000
    """Number of bootstrap resamples for correlation CI."""

    max_sample_for_advanced: int = 5000
    """Max rows sampled for expensive advanced analyses (t-SNE, UMAP, etc.)."""

    n_distribution_fits: int = 7
    """Number of candidate distributions to fit."""

    @staticmethod
    def minimal() -> "AnalysisConfig":
        """Return a config with only core analyses (descriptive + missing)."""
        return AnalysisConfig(
            preprocessing=False,
            distribution=False,
            correlation=False,
            outlier=False,
            categorical=False,
            feature_importance=False,
            pca=False,
            duplicates=False,
            quality_score=False,
            advanced=False,
        )

    @staticmethod
    def fast() -> "AnalysisConfig":
        """Return a config that skips expensive analyses (PCA, feature importance, advanced)."""
        return AnalysisConfig(
            pca=False,
            feature_importance=False,
            advanced=False,
        )

    @staticmethod
    def basic_only() -> "AnalysisConfig":
        """Return a config with all Basic analyses on, all Advanced off."""
        return AnalysisConfig(advanced=False)
