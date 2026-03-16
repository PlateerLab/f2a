"""Analysis orchestrator — coordinates the entire analysis pipeline.

This module connects preprocessing, statistical analysis, visualization,
and report generation into a single ``analyze()`` entry point.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")          # non-interactive backend
import matplotlib.pyplot as plt
import pandas as pd

from f2a.core.config import AnalysisConfig
from f2a.core.loader import DataLoader
from f2a.core.preprocessor import Preprocessor, PreprocessingResult
from f2a.core.schema import DataSchema, infer_schema
from f2a.stats.categorical import CategoricalStats
from f2a.stats.correlation import CorrelationStats
from f2a.stats.descriptive import DescriptiveStats
from f2a.stats.distribution import DistributionStats
from f2a.stats.duplicates import DuplicateStats
from f2a.stats.feature_importance import FeatureImportanceStats
from f2a.stats.missing import MissingStats
from f2a.stats.outlier import OutlierStats
from f2a.stats.pca_analysis import PCAStats
from f2a.stats.quality import QualityStats
from f2a.utils.logging import get_logger
from f2a.utils.validators import validate_source

logger = get_logger(__name__)


# =====================================================================
#  Result containers
# =====================================================================

@dataclass
class StatsResult:
    """Container for ALL statistical analysis results."""

    # Descriptive
    summary: pd.DataFrame = field(default_factory=pd.DataFrame)
    numeric_summary: pd.DataFrame = field(default_factory=pd.DataFrame)
    categorical_summary: pd.DataFrame = field(default_factory=pd.DataFrame)

    # Correlation
    correlation_matrix: pd.DataFrame = field(default_factory=pd.DataFrame)
    spearman_matrix: pd.DataFrame = field(default_factory=pd.DataFrame)
    cramers_v_matrix: pd.DataFrame = field(default_factory=pd.DataFrame)
    vif_table: pd.DataFrame = field(default_factory=pd.DataFrame)

    # Missing
    missing_info: pd.DataFrame = field(default_factory=pd.DataFrame)

    # Distribution
    distribution_info: pd.DataFrame = field(default_factory=pd.DataFrame)

    # Outlier
    outlier_summary: pd.DataFrame = field(default_factory=pd.DataFrame)

    # Categorical analysis
    categorical_analysis: pd.DataFrame = field(default_factory=pd.DataFrame)
    chi_square_matrix: pd.DataFrame = field(default_factory=pd.DataFrame)

    # Feature importance
    feature_importance: pd.DataFrame = field(default_factory=pd.DataFrame)

    # PCA
    pca_variance: pd.DataFrame = field(default_factory=pd.DataFrame)
    pca_loadings: pd.DataFrame = field(default_factory=pd.DataFrame)
    pca_summary: dict[str, Any] = field(default_factory=dict)

    # Duplicates
    duplicate_stats: dict[str, Any] = field(default_factory=dict)

    # Quality
    quality_scores: dict[str, Any] = field(default_factory=dict)
    quality_by_column: pd.DataFrame = field(default_factory=pd.DataFrame)

    # Preprocessing
    preprocessing: PreprocessingResult | None = None

    # Advanced analysis
    advanced_stats: dict[str, Any] = field(default_factory=dict)

    def get_numeric_summary(self) -> pd.DataFrame:
        return self.numeric_summary

    def get_categorical_summary(self) -> pd.DataFrame:
        return self.categorical_summary


@dataclass
class VizResult:
    """Container for lazy visualization generation."""

    _df: pd.DataFrame
    _schema: DataSchema
    _config: AnalysisConfig = field(default_factory=AnalysisConfig)
    _stats: StatsResult = field(default_factory=StatsResult)
    _figures: dict[str, plt.Figure] = field(default_factory=dict)

    # -- Core plots -------------------------------------------------------

    def plot_distributions(self) -> plt.Figure:
        from f2a.viz.plots import BasicPlotter
        p = BasicPlotter(self._df, self._schema)
        fig = p.histograms(columns=self._schema.numeric_columns[:self._config.max_plot_columns])
        self._figures["distributions"] = fig
        return fig

    def plot_boxplots(self) -> plt.Figure:
        from f2a.viz.plots import BasicPlotter
        p = BasicPlotter(self._df, self._schema)
        fig = p.boxplots(columns=self._schema.numeric_columns[:self._config.max_plot_columns])
        self._figures["boxplots"] = fig
        return fig

    def plot_bar_charts(self) -> plt.Figure:
        from f2a.viz.plots import BasicPlotter
        p = BasicPlotter(self._df, self._schema)
        fig = p.bar_charts(columns=self._schema.categorical_columns[:self._config.max_plot_columns])
        self._figures["bar_charts"] = fig
        return fig

    def plot_correlation(self, method: str = "pearson") -> plt.Figure:
        from f2a.viz.corr_plots import CorrelationPlotter
        p = CorrelationPlotter(self._df, self._schema)
        fig = p.heatmap(method=method)
        self._figures[f"correlation_{method}"] = fig
        return fig

    def plot_missing(self) -> plt.Figure:
        from f2a.viz.missing_plots import MissingPlotter
        p = MissingPlotter(self._df, self._schema)
        fig = p.bar()
        self._figures["missing_bar"] = fig
        return fig

    def plot_missing_matrix(self) -> plt.Figure:
        from f2a.viz.missing_plots import MissingPlotter
        p = MissingPlotter(self._df, self._schema)
        fig = p.matrix()
        self._figures["missing_matrix"] = fig
        return fig

    # -- Distribution plots -----------------------------------------------

    def plot_violins(self) -> plt.Figure:
        from f2a.viz.dist_plots import DistributionPlotter
        p = DistributionPlotter(self._df, self._schema)
        fig = p.violin_plots(columns=self._schema.numeric_columns[:self._config.max_plot_columns])
        self._figures["violins"] = fig
        return fig

    def plot_qq(self) -> plt.Figure:
        from f2a.viz.dist_plots import DistributionPlotter
        p = DistributionPlotter(self._df, self._schema)
        fig = p.qq_plots(columns=self._schema.numeric_columns[:self._config.max_plot_columns])
        self._figures["qq"] = fig
        return fig

    # -- Outlier plots ----------------------------------------------------

    def plot_outliers(self) -> plt.Figure:
        from f2a.viz.outlier_plots import OutlierPlotter
        p = OutlierPlotter(self._df, self._schema)
        fig = p.box_strip(columns=self._schema.numeric_columns[:self._config.max_plot_columns])
        self._figures["outliers"] = fig
        return fig

    # -- Categorical plots ------------------------------------------------

    def plot_categorical_frequency(self) -> plt.Figure:
        from f2a.viz.categorical_plots import CategoricalPlotter
        p = CategoricalPlotter(self._df, self._schema)
        fig = p.frequency_bars(
            columns=self._schema.categorical_columns[:self._config.max_plot_columns],
            top_n=self._config.max_categories,
        )
        self._figures["categorical_freq"] = fig
        return fig

    def plot_chi_square_heatmap(self) -> plt.Figure:
        from f2a.viz.categorical_plots import CategoricalPlotter
        p = CategoricalPlotter(self._df, self._schema)
        fig = p.chi_square_heatmap(self._stats.chi_square_matrix)
        self._figures["chi_square"] = fig
        return fig

    # -- PCA plots --------------------------------------------------------

    def plot_pca_scree(self) -> plt.Figure:
        from f2a.viz.pca_plots import PCAPlotter
        p = PCAPlotter()
        fig = p.scree_plot(self._stats.pca_variance)
        self._figures["pca_scree"] = fig
        return fig

    def plot_pca_loadings(self) -> plt.Figure:
        from f2a.viz.pca_plots import PCAPlotter
        p = PCAPlotter()
        fig = p.loadings_heatmap(self._stats.pca_loadings)
        self._figures["pca_loadings"] = fig
        return fig

    # -- Quality / Feature importance plots --------------------------------

    def plot_quality(self) -> plt.Figure:
        from f2a.viz.quality_plots import QualityPlotter
        p = QualityPlotter()
        fig = p.dimension_bar(self._stats.quality_scores)
        self._figures["quality"] = fig
        return fig

    def plot_column_quality(self) -> plt.Figure:
        from f2a.viz.quality_plots import QualityPlotter
        p = QualityPlotter()
        fig = p.column_quality_heatmap(self._stats.quality_by_column)
        self._figures["column_quality"] = fig
        return fig

    def plot_feature_importance(self) -> plt.Figure:
        from f2a.viz.quality_plots import QualityPlotter
        p = QualityPlotter()
        fig = p.feature_importance_bar(self._stats.feature_importance)
        self._figures["feature_importance"] = fig
        return fig

    # -- Advanced plots ---------------------------------------------------

    def plot_best_fit_overlay(self) -> plt.Figure:
        from f2a.viz.advanced_dist_plots import AdvancedDistPlotter
        p = AdvancedDistPlotter()
        bf = self._stats.advanced_stats.get("advanced_distribution", {}).get("best_fit")
        if bf is None or bf.empty:
            return None  # type: ignore[return-value]
        fig = p.best_fit_overlay(self._df, bf)
        self._figures["best_fit_overlay"] = fig
        return fig

    def plot_ecdf(self) -> plt.Figure:
        from f2a.viz.advanced_dist_plots import AdvancedDistPlotter
        p = AdvancedDistPlotter()
        ecdf_data = self._stats.advanced_stats.get("ecdf_data", {})
        if not ecdf_data:
            return None  # type: ignore[return-value]
        fig = p.ecdf_plot(ecdf_data)
        self._figures["ecdf"] = fig
        return fig

    def plot_power_transform(self) -> plt.Figure:
        from f2a.viz.advanced_dist_plots import AdvancedDistPlotter
        p = AdvancedDistPlotter()
        pt = self._stats.advanced_stats.get("advanced_distribution", {}).get("power_transform")
        if pt is None or pt.empty:
            return None  # type: ignore[return-value]
        fig = p.power_transform_plot(self._df, pt)
        self._figures["power_transform"] = fig
        return fig

    def plot_jarque_bera(self) -> plt.Figure:
        from f2a.viz.advanced_dist_plots import AdvancedDistPlotter
        p = AdvancedDistPlotter()
        jb = self._stats.advanced_stats.get("advanced_distribution", {}).get("jarque_bera")
        if jb is None or jb.empty:
            return None  # type: ignore[return-value]
        fig = p.jarque_bera_summary(jb)
        self._figures["jarque_bera"] = fig
        return fig

    def plot_partial_correlation(self) -> plt.Figure:
        from f2a.viz.advanced_corr_plots import AdvancedCorrPlotter
        p = AdvancedCorrPlotter()
        pcorr = self._stats.advanced_stats.get("advanced_correlation", {}).get("partial_correlation")
        if pcorr is None or pcorr.empty:
            return None  # type: ignore[return-value]
        fig = p.partial_correlation_heatmap(pcorr)
        self._figures["partial_correlation"] = fig
        return fig

    def plot_mi_heatmap(self) -> plt.Figure:
        from f2a.viz.advanced_corr_plots import AdvancedCorrPlotter
        p = AdvancedCorrPlotter()
        mi = self._stats.advanced_stats.get("advanced_correlation", {}).get("mutual_information")
        if mi is None or mi.empty:
            return None  # type: ignore[return-value]
        fig = p.mi_heatmap(mi)
        self._figures["mi_heatmap"] = fig
        return fig

    def plot_bootstrap_ci(self) -> plt.Figure:
        from f2a.viz.advanced_corr_plots import AdvancedCorrPlotter
        p = AdvancedCorrPlotter()
        bci = self._stats.advanced_stats.get("advanced_correlation", {}).get("bootstrap_ci")
        if bci is None or bci.empty:
            return None  # type: ignore[return-value]
        fig = p.bootstrap_ci_plot(bci)
        self._figures["bootstrap_ci"] = fig
        return fig

    def plot_correlation_network(self) -> plt.Figure:
        from f2a.viz.advanced_corr_plots import AdvancedCorrPlotter
        p = AdvancedCorrPlotter()
        net = self._stats.advanced_stats.get("advanced_correlation", {}).get("network")
        if not net or not net.get("edges"):
            return None  # type: ignore[return-value]
        fig = p.correlation_network(net)
        self._figures["correlation_network"] = fig
        return fig

    def plot_distance_correlation(self) -> plt.Figure:
        from f2a.viz.advanced_corr_plots import AdvancedCorrPlotter
        p = AdvancedCorrPlotter()
        dc = self._stats.advanced_stats.get("advanced_correlation", {}).get("distance_correlation")
        if dc is None or dc.empty:
            return None  # type: ignore[return-value]
        fig = p.distance_correlation_heatmap(dc)
        self._figures["distance_correlation"] = fig
        return fig

    def plot_elbow_silhouette(self) -> plt.Figure:
        from f2a.viz.cluster_plots import ClusterPlotter
        p = ClusterPlotter()
        km = self._stats.advanced_stats.get("clustering", {}).get("kmeans")
        if not km:
            return None  # type: ignore[return-value]
        fig = p.elbow_silhouette(km)
        self._figures["elbow_silhouette"] = fig
        return fig

    def plot_cluster_scatter(self) -> plt.Figure:
        from f2a.viz.cluster_plots import ClusterPlotter
        p = ClusterPlotter()
        km = self._stats.advanced_stats.get("clustering", {}).get("kmeans")
        if not km:
            return None  # type: ignore[return-value]
        fig = p.cluster_scatter_2d(
            self._df, self._schema.numeric_columns, km,
        )
        self._figures["cluster_scatter"] = fig
        return fig

    def plot_dendrogram(self) -> plt.Figure:
        from f2a.viz.cluster_plots import ClusterPlotter
        p = ClusterPlotter()
        hc = self._stats.advanced_stats.get("clustering", {}).get("hierarchical")
        if not hc:
            return None  # type: ignore[return-value]
        fig = p.dendrogram(hc)
        self._figures["dendrogram"] = fig
        return fig

    def plot_cluster_profiles(self) -> plt.Figure:
        from f2a.viz.cluster_plots import ClusterPlotter
        p = ClusterPlotter()
        profiles = self._stats.advanced_stats.get("clustering", {}).get("profiles")
        if profiles is None or profiles.empty:
            return None  # type: ignore[return-value]
        fig = p.cluster_profile_heatmap(profiles)
        self._figures["cluster_profiles"] = fig
        return fig

    def plot_anomaly_scatter(self) -> plt.Figure:
        from f2a.viz.advanced_anomaly_plots import AdvancedAnomalyPlotter
        p = AdvancedAnomalyPlotter()
        iso = self._stats.advanced_stats.get("advanced_anomaly_full", {}).get("isolation_forest")
        if not iso:
            return None  # type: ignore[return-value]
        fig = p.anomaly_scatter_2d(
            self._df, self._schema.numeric_columns, iso,
        )
        self._figures["anomaly_scatter"] = fig
        return fig

    def plot_mahalanobis_hist(self) -> plt.Figure:
        from f2a.viz.advanced_anomaly_plots import AdvancedAnomalyPlotter
        p = AdvancedAnomalyPlotter()
        maha = self._stats.advanced_stats.get("advanced_anomaly_full", {}).get("mahalanobis")
        if not maha:
            return None  # type: ignore[return-value]
        fig = p.mahalanobis_histogram(maha)
        self._figures["mahalanobis_hist"] = fig
        return fig

    def plot_consensus_comparison(self) -> plt.Figure:
        from f2a.viz.advanced_anomaly_plots import AdvancedAnomalyPlotter
        p = AdvancedAnomalyPlotter()
        cons = self._stats.advanced_stats.get("advanced_anomaly", {}).get("consensus")
        if not cons:
            return None  # type: ignore[return-value]
        fig = p.consensus_comparison(cons)
        self._figures["consensus_comparison"] = fig
        return fig


# =====================================================================
#  Subset / Analysis Report
# =====================================================================

@dataclass
class SubsetReport:
    """Analysis results for a single subset/split partition."""

    subset: str
    split: str
    shape: tuple[int, int]
    schema: DataSchema
    stats: StatsResult
    viz: VizResult
    warnings: list[str] = field(default_factory=list)


@dataclass
class AnalysisReport:
    """Top-level container for analysis results.

    Attributes:
        dataset_name: Dataset name.
        shape: ``(rows, columns)`` tuple.
        schema: Data schema.
        stats: Statistical analysis results.
        viz: Visualization access object.
        warnings: List of warnings.
        subsets: Per-subset/split reports (empty for single partition).
        config: The :class:`AnalysisConfig` used.
    """

    dataset_name: str
    shape: tuple[int, int]
    schema: DataSchema
    stats: StatsResult
    viz: VizResult
    warnings: list[str] = field(default_factory=list)
    subsets: list[SubsetReport] = field(default_factory=list)
    config: AnalysisConfig = field(default_factory=AnalysisConfig)
    analysis_started_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))
    analysis_duration_sec: float = 0.0

    # -- Console output ---------------------------------------------------

    def show(self) -> None:
        """Print analysis summary to console."""
        sep = "=" * 60
        print(sep)
        print(f"  f2a Analysis Report: {self.dataset_name}")
        print(sep)

        if self.subsets:
            print(f"\n  Total Rows: {self.shape[0]:,}  |  Subsets: {len(self.subsets)}")
            for sr in self.subsets:
                print(f"\n{'-' * 60}")
                print(f"  [{sr.subset} / {sr.split}]  {sr.shape[0]:,} rows x {sr.shape[1]} cols")
                print(f"  Memory: {sr.schema.memory_usage_mb} MB")
                print(f"  Numeric: {len(sr.schema.numeric_columns)} | "
                      f"Categorical: {len(sr.schema.categorical_columns)} | "
                      f"Text: {len(sr.schema.text_columns)} | "
                      f"Datetime: {len(sr.schema.datetime_columns)}")
                if not sr.stats.summary.empty:
                    print()
                    print(sr.stats.summary.to_string())
                if sr.warnings:
                    print("\n  Warnings:")
                    for w in sr.warnings:
                        print(f"    - {w}")
        else:
            print(f"\n  Rows: {self.shape[0]:,}  |  Columns: {self.shape[1]}")
            print(f"  Memory: {self.schema.memory_usage_mb} MB")
            print(f"\n  Numeric: {len(self.schema.numeric_columns)}")
            print(f"  Categorical: {len(self.schema.categorical_columns)}")
            print(f"  Text: {len(self.schema.text_columns)}")
            print(f"  Datetime: {len(self.schema.datetime_columns)}")

            if self.stats.quality_scores:
                qs = self.stats.quality_scores
                print(f"\n  Data Quality: {qs.get('overall', 0) * 100:.1f}%")

            if self.stats.preprocessing:
                pp = self.stats.preprocessing
                n_issues = (
                    len(pp.constant_columns) + len(pp.high_missing_columns)
                    + len(pp.id_like_columns) + pp.duplicate_rows_count
                    + len(pp.mixed_type_columns) + len(pp.infinite_value_columns)
                )
                print(f"  Preprocessing: {len(pp.cleaning_log)} steps, {n_issues} issues found")

            print(f"\n{'-' * 60}")
            print("  Summary Statistics:")
            if not self.stats.summary.empty:
                print(self.stats.summary.to_string())

            if not self.stats.outlier_summary.empty:
                total_outliers = self.stats.outlier_summary.get("outlier_count", pd.Series()).sum()
                if total_outliers > 0:
                    print(f"\n  Outliers detected: {int(total_outliers)} total across numeric columns")

            if self.stats.pca_summary:
                ps = self.stats.pca_summary
                print(f"\n  PCA: {ps.get('components_for_90pct', '?')} components explain 90% variance")

            if self.warnings:
                print(f"\n{'-' * 60}")
                print("  Warnings:")
                for w in self.warnings:
                    print(f"    - {w}")

        print(sep)

    # -- HTML report ------------------------------------------------------

    def to_html(self, output_dir: str = ".") -> Path:
        """Generate and save an HTML report.

        Args:
            output_dir: Output directory path.

        Returns:
            Path to the saved HTML file.
        """
        from f2a.report.generator import ReportGenerator

        generator = ReportGenerator()
        safe_name = re.sub(r'[<>:"/\\|?*]', "_", self.dataset_name)
        safe_name = safe_name.strip(". ")[:120] or "report"
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = Path(output_dir) / f"{safe_name}_{ts}_report.html"

        if self.subsets:
            subset_sections = self._build_subset_sections()
            generator.save_html_multi(
                output_path=output_path,
                dataset_name=self.dataset_name,
                sections=subset_sections,
                config=self.config,
                analysis_started_at=self.analysis_started_at,
                analysis_duration_sec=self.analysis_duration_sec,
            )
        else:
            report_data = self._build_single_report_data()
            generator.save_html(output_path=output_path, **report_data)

        return output_path

    def _build_single_report_data(self) -> dict[str, Any]:
        figures = self._generate_figures(self.viz, self.stats, self.config)
        return {
            "dataset_name": self.dataset_name,
            "schema_summary": self.schema.summary_dict(),
            "stats": self.stats,
            "figures": figures,
            "warnings": self.warnings,
            "config": self.config,
            "analysis_started_at": self.analysis_started_at,
            "analysis_duration_sec": self.analysis_duration_sec,
        }

    def _build_subset_sections(self) -> list[dict[str, Any]]:
        sections: list[dict[str, Any]] = []
        for sr in self.subsets:
            figures = self._generate_figures(sr.viz, sr.stats, self.config)
            sections.append({
                "subset": sr.subset,
                "split": sr.split,
                "schema_summary": sr.schema.summary_dict(),
                "stats": sr.stats,
                "figures": figures,
                "warnings": sr.warnings,
            })
        return sections

    @staticmethod
    def _generate_figures(
        viz: VizResult,
        stats: StatsResult,
        config: AnalysisConfig,
    ) -> dict[str, plt.Figure]:
        """Generate all configured figures, catching individual failures."""
        figures: dict[str, plt.Figure] = {}

        if not config.visualizations:
            return figures

        plot_attempts: list[tuple[str, Any, bool]] = [
            ("Distribution Histograms", viz.plot_distributions, config.descriptive),
            ("Boxplots", viz.plot_boxplots, config.descriptive),
            ("Violin Plots", viz.plot_violins, config.distribution),
            ("Q-Q Plots", viz.plot_qq, config.distribution),
            ("Correlation Heatmap (Pearson)", lambda: viz.plot_correlation("pearson"), config.correlation),
            ("Correlation Heatmap (Spearman)", lambda: viz.plot_correlation("spearman"), config.correlation),
            ("Missing Data", viz.plot_missing, True),
            ("Missing Data Matrix", viz.plot_missing_matrix, True),
            ("Outlier Detection", viz.plot_outliers, config.outlier),
            ("Categorical Frequency", viz.plot_categorical_frequency, config.categorical),
            (
                "Chi-Square Heatmap",
                viz.plot_chi_square_heatmap,
                config.categorical and not stats.chi_square_matrix.empty,
            ),
            (
                "PCA Scree Plot",
                viz.plot_pca_scree,
                config.pca and not stats.pca_variance.empty,
            ),
            (
                "PCA Loadings",
                viz.plot_pca_loadings,
                config.pca and not stats.pca_loadings.empty,
            ),
            (
                "Data Quality Scores",
                viz.plot_quality,
                config.quality_score and bool(stats.quality_scores),
            ),
            (
                "Column Quality",
                viz.plot_column_quality,
                config.quality_score and not stats.quality_by_column.empty,
            ),
            (
                "Feature Importance",
                viz.plot_feature_importance,
                config.feature_importance and not stats.feature_importance.empty,
            ),
        ]

        # -- Advanced plots -----------------------------------------------
        if config.advanced:
            adv = stats.advanced_stats
            adv_attempts: list[tuple[str, Any, bool]] = [
                # A1. Advanced Distribution
                (
                    "Best-Fit Distribution Overlay",
                    viz.plot_best_fit_overlay,
                    config.advanced_distribution
                    and bool(adv.get("advanced_distribution", {}).get("best_fit") is not None),
                ),
                (
                    "ECDF Plot",
                    viz.plot_ecdf,
                    config.advanced_distribution and bool(adv.get("ecdf_data")),
                ),
                (
                    "Power Transform Comparison",
                    viz.plot_power_transform,
                    config.advanced_distribution
                    and bool(adv.get("advanced_distribution", {}).get("power_transform") is not None),
                ),
                (
                    "Jarque-Bera Normality Test",
                    viz.plot_jarque_bera,
                    config.advanced_distribution
                    and bool(adv.get("advanced_distribution", {}).get("jarque_bera") is not None),
                ),
                # A2. Advanced Correlation
                (
                    "Partial Correlation Heatmap",
                    viz.plot_partial_correlation,
                    config.advanced_correlation
                    and bool(adv.get("advanced_correlation", {}).get("partial_correlation") is not None),
                ),
                (
                    "Mutual Information Heatmap",
                    viz.plot_mi_heatmap,
                    config.advanced_correlation
                    and bool(adv.get("advanced_correlation", {}).get("mutual_information") is not None),
                ),
                (
                    "Bootstrap Correlation CI",
                    viz.plot_bootstrap_ci,
                    config.advanced_correlation
                    and bool(adv.get("advanced_correlation", {}).get("bootstrap_ci") is not None),
                ),
                (
                    "Correlation Network",
                    viz.plot_correlation_network,
                    config.advanced_correlation
                    and bool(adv.get("advanced_correlation", {}).get("network")),
                ),
                (
                    "Distance Correlation Heatmap",
                    viz.plot_distance_correlation,
                    config.advanced_correlation
                    and bool(adv.get("advanced_correlation", {}).get("distance_correlation") is not None),
                ),
                # A3. Clustering
                (
                    "Elbow & Silhouette",
                    viz.plot_elbow_silhouette,
                    config.clustering and bool(adv.get("clustering", {}).get("kmeans")),
                ),
                (
                    "Cluster Scatter",
                    viz.plot_cluster_scatter,
                    config.clustering and bool(adv.get("clustering", {}).get("kmeans")),
                ),
                (
                    "Dendrogram",
                    viz.plot_dendrogram,
                    config.clustering and bool(adv.get("clustering", {}).get("hierarchical")),
                ),
                (
                    "Cluster Profiles",
                    viz.plot_cluster_profiles,
                    config.clustering and bool(adv.get("clustering", {}).get("profiles") is not None),
                ),
                # A6. Advanced Anomaly
                (
                    "Anomaly Scatter",
                    viz.plot_anomaly_scatter,
                    config.advanced_anomaly
                    and bool(adv.get("advanced_anomaly_full", {}).get("isolation_forest")),
                ),
                (
                    "Mahalanobis Distance",
                    viz.plot_mahalanobis_hist,
                    config.advanced_anomaly
                    and bool(adv.get("advanced_anomaly_full", {}).get("mahalanobis")),
                ),
                (
                    "Consensus Anomaly Comparison",
                    viz.plot_consensus_comparison,
                    config.advanced_anomaly
                    and bool(adv.get("advanced_anomaly", {}).get("consensus")),
                ),
            ]
            plot_attempts.extend(adv_attempts)

        for name, fn, condition in plot_attempts:
            if not condition:
                continue
            try:
                fig = fn()
                if fig is not None:
                    figures[name] = fig
            except Exception as exc:
                logger.debug("Figure '%s' skipped: %s", name, exc)

        return figures

    # -- Dict export -------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Return analysis results as a dictionary."""
        result: dict[str, Any] = {
            "dataset_name": self.dataset_name,
            "shape": self.shape,
            "schema": self.schema.summary_dict(),
            "stats_summary": self.stats.summary.to_dict() if not self.stats.summary.empty else {},
            "correlation_matrix": (
                self.stats.correlation_matrix.to_dict()
                if not self.stats.correlation_matrix.empty else {}
            ),
            "outlier_summary": (
                self.stats.outlier_summary.to_dict()
                if not self.stats.outlier_summary.empty else {}
            ),
            "quality_scores": self.stats.quality_scores,
            "pca_summary": self.stats.pca_summary,
            "duplicate_stats": self.stats.duplicate_stats,
            "warnings": self.warnings,
        }
        if self.subsets:
            result["subsets"] = [
                {
                    "subset": sr.subset,
                    "split": sr.split,
                    "shape": sr.shape,
                    "schema": sr.schema.summary_dict(),
                    "stats_summary": sr.stats.summary.to_dict() if not sr.stats.summary.empty else {},
                    "quality_scores": sr.stats.quality_scores,
                    "warnings": sr.warnings,
                }
                for sr in self.subsets
            ]
        return result


# =====================================================================
#  Analyzer
# =====================================================================

class Analyzer:
    """Orchestrate the full analysis pipeline.

    Example::

        analyzer = Analyzer()
        report = analyzer.run("data.csv")
        report.show()
    """

    def __init__(self) -> None:
        self._loader = DataLoader()

    def run(
        self,
        source: str,
        config: AnalysisConfig | None = None,
        **kwargs: Any,
    ) -> AnalysisReport:
        """Execute the full analysis pipeline.

        Args:
            source: Data source (file path or HuggingFace address).
            config: Analysis configuration.  Defaults to all-on.
            **kwargs: Additional arguments passed to the loader.

        Returns:
            :class:`AnalysisReport` instance.
        """
        config = config or AnalysisConfig()
        source = validate_source(source)
        logger.info("Analysis started: %s", source)

        # 1. Load data
        df = self._loader.load(source, **kwargs)

        # 2. Check for multi-subset HuggingFace data
        has_partitions = "__subset__" in df.columns and "__split__" in df.columns

        if has_partitions:
            return self._run_multi_subset(source, df, config)

        return self._run_single(source, df, config)

    # -- Single partition --------------------------------------------------

    def _run_single(
        self, source: str, df: pd.DataFrame, config: AnalysisConfig,
    ) -> AnalysisReport:
        t0 = time.perf_counter()
        started_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

        schema = infer_schema(df)
        logger.info("Schema inference complete: %s", schema.summary_dict())

        warnings: list[str] = []
        stats = self._compute_stats(df, schema, warnings, config)

        viz_df = stats.preprocessing.cleaned_df if stats.preprocessing else df
        viz_schema = infer_schema(viz_df) if stats.preprocessing else schema

        dataset_name = (
            Path(source).stem
            if "/" not in source or "://" not in source
            else source
        )
        viz = VizResult(_df=viz_df, _schema=viz_schema, _config=config, _stats=stats)

        elapsed = round(time.perf_counter() - t0, 2)
        report = AnalysisReport(
            dataset_name=dataset_name,
            shape=(len(df), len(df.columns)),
            schema=schema,
            stats=stats,
            viz=viz,
            warnings=warnings,
            config=config,
            analysis_started_at=started_at,
            analysis_duration_sec=elapsed,
        )
        logger.info("Analysis complete: %s (%.2fs)", source, elapsed)
        return report

    # -- Multi-subset ------------------------------------------------------

    def _run_multi_subset(
        self, source: str, df: pd.DataFrame, config: AnalysisConfig,
    ) -> AnalysisReport:
        t0 = time.perf_counter()
        started_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

        groups = df.groupby(["__subset__", "__split__"], sort=False)

        subset_reports: list[SubsetReport] = []
        all_warnings: list[str] = []

        for (subset_name, split_name), group_df in groups:
            part_df = group_df.drop(columns=["__subset__", "__split__"]).reset_index(drop=True)

            schema = infer_schema(part_df)
            warnings: list[str] = []
            stats = self._compute_stats(part_df, schema, warnings, config)

            viz_df = stats.preprocessing.cleaned_df if stats.preprocessing else part_df
            viz_schema = infer_schema(viz_df) if stats.preprocessing else schema
            viz = VizResult(_df=viz_df, _schema=viz_schema, _config=config, _stats=stats)

            sr = SubsetReport(
                subset=str(subset_name),
                split=str(split_name),
                shape=(len(part_df), len(part_df.columns)),
                schema=schema,
                stats=stats,
                viz=viz,
                warnings=warnings,
            )
            subset_reports.append(sr)
            all_warnings.extend(f"[{subset_name}/{split_name}] {w}" for w in warnings)
            logger.info(
                "Subset analysis complete: %s/%s (%d rows x %d cols)",
                subset_name, split_name, len(part_df), len(part_df.columns),
            )

        first = subset_reports[0]
        total_rows = sum(sr.shape[0] for sr in subset_reports)

        elapsed = round(time.perf_counter() - t0, 2)
        report = AnalysisReport(
            dataset_name=source,
            shape=(total_rows, first.shape[1]),
            schema=first.schema,
            stats=first.stats,
            viz=first.viz,
            warnings=all_warnings,
            subsets=subset_reports,
            config=config,
            analysis_started_at=started_at,
            analysis_duration_sec=elapsed,
        )
        logger.info(
            "Multi-subset analysis complete: %s (%d subsets, %d total rows, %.2fs)",
            source, len(subset_reports), total_rows, elapsed,
        )
        return report

    # -- Stats computation -------------------------------------------------

    def _compute_stats(
        self,
        df: pd.DataFrame,
        schema: DataSchema,
        warnings: list[str],
        config: AnalysisConfig,
    ) -> StatsResult:
        """Perform all configured statistical analyses."""
        result = StatsResult()

        # 0. Preprocessing
        analysis_df = df
        if config.preprocessing:
            try:
                pp = Preprocessor(df, schema)
                result.preprocessing = pp.run()
                analysis_df = result.preprocessing.cleaned_df
                schema = infer_schema(analysis_df)

                for log_entry in result.preprocessing.cleaning_log:
                    logger.info("Preprocessing: %s", log_entry)
                if result.preprocessing.high_missing_columns:
                    for item in result.preprocessing.high_missing_columns:
                        warnings.append(
                            f"High missing ratio: {item['column']} "
                            f"({item['missing_ratio'] * 100:.1f}%)"
                        )
                if result.preprocessing.id_like_columns:
                    warnings.append(
                        f"ID-like columns detected: "
                        f"{', '.join(result.preprocessing.id_like_columns[:5])}"
                    )
            except Exception as exc:
                logger.warning("Preprocessing failed: %s", exc)

        # 1. Descriptive statistics
        if config.descriptive:
            try:
                desc = DescriptiveStats(analysis_df, schema)
                result.summary = desc.summary()
                result.numeric_summary = desc.numeric_summary()
                result.categorical_summary = desc.categorical_summary()
            except Exception as exc:
                logger.warning("Descriptive stats failed: %s", exc)

        # 2. Distribution analysis
        if config.distribution:
            try:
                dist = DistributionStats(analysis_df, schema)
                result.distribution_info = dist.analyze()
            except Exception as exc:
                logger.warning("Distribution analysis failed: %s", exc)

        # 3. Correlation analysis
        if config.correlation:
            try:
                corr = CorrelationStats(analysis_df, schema)
                result.correlation_matrix = corr.pearson()
                result.spearman_matrix = corr.spearman()
                result.cramers_v_matrix = corr.cramers_v_matrix()

                try:
                    result.vif_table = corr.vif()
                except Exception:
                    pass

                high_corrs = corr.high_correlations(threshold=config.correlation_threshold)
                for col_a, col_b, val in high_corrs:
                    warnings.append(f"High correlation: {col_a} <-> {col_b} (r={val})")
            except Exception as exc:
                logger.warning("Correlation analysis failed: %s", exc)

        # 4. Missing data analysis (always run)
        try:
            miss = MissingStats(analysis_df, schema)
            result.missing_info = miss.column_summary()
            total_missing = miss.total_missing_ratio()
            if total_missing > 0.1:
                warnings.append(
                    f"Overall missing ratio is high: {total_missing * 100:.1f}%"
                )
        except Exception as exc:
            logger.warning("Missing data analysis failed: %s", exc)

        # 5. Outlier detection
        if config.outlier:
            try:
                out = OutlierStats(analysis_df, schema)
                kw: dict[str, Any] = {}
                if config.outlier_method == "iqr":
                    kw["multiplier"] = config.outlier_threshold
                else:
                    kw["threshold"] = config.outlier_threshold
                result.outlier_summary = out.summary(method=config.outlier_method, **kw)

                if not result.outlier_summary.empty and "outlier_%" in result.outlier_summary.columns:
                    for col_name, row in result.outlier_summary.iterrows():
                        if row.get("outlier_%", 0) > 10:
                            warnings.append(
                                f"High outlier ratio in '{col_name}': {row['outlier_%']:.1f}%"
                            )
            except Exception as exc:
                logger.warning("Outlier detection failed: %s", exc)

        # 6. Categorical analysis
        if config.categorical:
            try:
                cat = CategoricalStats(analysis_df, schema)
                result.categorical_analysis = cat.summary()
                result.chi_square_matrix = cat.chi_square_matrix()
            except Exception as exc:
                logger.warning("Categorical analysis failed: %s", exc)

        # 7. Feature importance
        if config.feature_importance:
            try:
                fi = FeatureImportanceStats(analysis_df, schema)
                result.feature_importance = fi.variance_ranking()
            except Exception as exc:
                logger.warning("Feature importance failed: %s", exc)

        # 8. PCA
        if config.pca:
            try:
                pca = PCAStats(
                    analysis_df, schema, max_components=config.pca_max_components,
                )
                result.pca_variance = pca.variance_explained()
                result.pca_loadings = pca.loadings()
                result.pca_summary = pca.summary()
            except Exception as exc:
                logger.warning("PCA analysis failed: %s", exc)

        # 9. Duplicates
        if config.duplicates:
            try:
                dup = DuplicateStats(analysis_df, schema)
                result.duplicate_stats = dup.summary()
            except Exception as exc:
                logger.warning("Duplicate detection failed: %s", exc)

        # 10. Quality score
        if config.quality_score:
            try:
                qs = QualityStats(analysis_df, schema)
                result.quality_scores = qs.summary()
                result.quality_by_column = qs.column_quality()
            except Exception as exc:
                logger.warning("Quality scoring failed: %s", exc)

        # 11. Advanced analyses
        if config.advanced:
            self._compute_advanced_stats(analysis_df, schema, result, config)

        return result

    # -- Advanced stats computation ----------------------------------------

    def _compute_advanced_stats(
        self,
        df: pd.DataFrame,
        schema: DataSchema,
        result: StatsResult,
        config: AnalysisConfig,
    ) -> None:
        """Compute advanced analysis modules and populate result.advanced_stats."""
        adv = result.advanced_stats

        # A1. Advanced Distribution
        if config.advanced_distribution:
            try:
                from f2a.stats.advanced_distribution import AdvancedDistributionStats
                ad = AdvancedDistributionStats(
                    df, schema,
                    n_fits=config.n_distribution_fits,
                    max_sample=config.max_sample_for_advanced,
                )
                adv["advanced_distribution"] = ad.summary()
                adv["ecdf_data"] = ad.ecdf()
            except Exception as exc:
                logger.debug("Advanced distribution failed: %s", exc)

        # A2. Advanced Correlation
        if config.advanced_correlation:
            try:
                from f2a.stats.advanced_correlation import AdvancedCorrelationStats
                ac = AdvancedCorrelationStats(
                    df, schema,
                    bootstrap_iterations=config.bootstrap_iterations,
                    max_sample=config.max_sample_for_advanced,
                )
                adv["advanced_correlation"] = ac.summary()
            except Exception as exc:
                logger.debug("Advanced correlation failed: %s", exc)

        # A3. Clustering
        if config.clustering:
            try:
                from f2a.stats.clustering import ClusteringStats
                cl = ClusteringStats(
                    df, schema,
                    max_k=config.max_cluster_k,
                    max_sample=config.max_sample_for_advanced,
                )
                adv["clustering"] = cl.summary()
            except Exception as exc:
                logger.debug("Clustering failed: %s", exc)

        # A4. Dimensionality Reduction
        if config.advanced_dimreduction:
            try:
                from f2a.stats.advanced_dimreduction import AdvancedDimReductionStats
                dr = AdvancedDimReductionStats(
                    df, schema,
                    tsne_perplexity=config.tsne_perplexity,
                    max_sample=config.max_sample_for_advanced,
                )
                adv["dimreduction"] = dr.summary()
            except Exception as exc:
                logger.debug("Dimensionality reduction failed: %s", exc)

        # A5. Feature Insights
        if config.feature_insights:
            try:
                from f2a.stats.feature_insights import FeatureInsightsStats
                fi = FeatureInsightsStats(
                    df, schema,
                    max_sample=config.max_sample_for_advanced,
                )
                adv["feature_insights"] = fi.summary()
            except Exception as exc:
                logger.debug("Feature insights failed: %s", exc)

        # A6. Advanced Anomaly Detection
        if config.advanced_anomaly:
            try:
                from f2a.stats.advanced_anomaly import AdvancedAnomalyStats
                aa = AdvancedAnomalyStats(
                    df, schema,
                    max_sample=config.max_sample_for_advanced,
                )
                stripped, full = aa.summary_full()
                adv["advanced_anomaly"] = stripped
                adv["advanced_anomaly_full"] = full
            except Exception as exc:
                logger.debug("Advanced anomaly detection failed: %s", exc)

        # A7. Statistical Tests
        if config.statistical_tests:
            try:
                from f2a.stats.statistical_tests import StatisticalTests
                st = StatisticalTests(df, schema)
                adv["statistical_tests"] = st.summary()
            except Exception as exc:
                logger.debug("Statistical tests failed: %s", exc)

        # A8. Data Profiling (aggregated summary)
        if config.data_profiling:
            try:
                profile: dict[str, Any] = {
                    "n_rows": len(df),
                    "n_cols": len(df.columns),
                    "memory_mb": round(df.memory_usage(deep=True).sum() / 1024 / 1024, 2),
                    "numeric_ratio": round(
                        len(schema.numeric_columns) / max(len(df.columns), 1), 3,
                    ),
                    "categorical_ratio": round(
                        len(schema.categorical_columns) / max(len(df.columns), 1), 3,
                    ),
                    "missing_ratio": round(
                        df.isnull().sum().sum() / max(df.size, 1), 4,
                    ),
                    "duplicate_row_ratio": round(
                        df.duplicated().sum() / max(len(df), 1), 4,
                    ),
                }
                adv["data_profiling"] = profile
            except Exception as exc:
                logger.debug("Data profiling failed: %s", exc)


# =====================================================================
#  Public entry point
# =====================================================================

def analyze(
    source: str,
    config: AnalysisConfig | None = None,
    **kwargs: Any,
) -> AnalysisReport:
    """Analyze a data source and return a comprehensive report.

    This function is the main entry point for ``f2a``.

    Args:
        source: File path or HuggingFace dataset address.
        config: :class:`AnalysisConfig` to control which analyses run.
            Defaults to all analyses enabled.
        **kwargs: Additional arguments passed to the data loader.

    Returns:
        :class:`AnalysisReport` with statistics, visualization, and report
        generation capabilities.

    Example::

        import f2a
        report = f2a.analyze("sales.csv")
        report.show()
        report.to_html("output/")
    """
    analyzer = Analyzer()
    return analyzer.run(source, config=config, **kwargs)
