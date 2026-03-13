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
        )

    @staticmethod
    def fast() -> "AnalysisConfig":
        """Return a config that skips expensive analyses (PCA, feature importance)."""
        return AnalysisConfig(
            pca=False,
            feature_importance=False,
        )
