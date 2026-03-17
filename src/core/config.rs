use serde::{Deserialize, Serialize};

/// Master configuration for analysis – mirrors `f2a.AnalysisConfig`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalysisConfig {
    // ── Basic analysis toggles ──────────────────────────────────
    pub descriptive: bool,
    pub correlation: bool,
    pub distribution: bool,
    pub missing: bool,
    pub outlier: bool,
    pub categorical: bool,
    pub feature_importance: bool,
    pub pca: bool,
    pub duplicates: bool,
    pub quality: bool,
    pub preprocessing: bool,

    // ── Advanced analysis toggles ───────────────────────────────
    pub advanced: bool,
    pub advanced_distribution: bool,
    pub advanced_correlation: bool,
    pub clustering: bool,
    pub advanced_dimreduction: bool,
    pub feature_insights: bool,
    pub advanced_anomaly: bool,
    pub statistical_tests: bool,
    pub data_profiling: bool,
    pub insight_engine: bool,
    pub cross_analysis: bool,
    pub column_role: bool,
    pub ml_readiness: bool,

    // ── Parameters ──────────────────────────────────────────────
    pub outlier_threshold: f64,
    pub outlier_method: OutlierMethod,
    pub correlation_threshold: f64,
    pub pca_max_components: usize,
    pub max_categories: usize,
    pub max_plot_columns: usize,
    pub max_cluster_k: usize,
    pub tsne_perplexity: f64,
    pub bootstrap_iterations: usize,
    pub max_sample_for_advanced: usize,
    pub n_distribution_fits: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum OutlierMethod {
    Iqr,
    Zscore,
}

impl Default for AnalysisConfig {
    fn default() -> Self {
        Self {
            // Basic – all on by default
            descriptive: true,
            correlation: true,
            distribution: true,
            missing: true,
            outlier: true,
            categorical: true,
            feature_importance: true,
            pca: true,
            duplicates: true,
            quality: true,
            preprocessing: true,
            // Advanced – all on by default
            advanced: true,
            advanced_distribution: true,
            advanced_correlation: true,
            clustering: true,
            advanced_dimreduction: true,
            feature_insights: true,
            advanced_anomaly: true,
            statistical_tests: true,
            data_profiling: true,
            insight_engine: true,
            cross_analysis: true,
            column_role: true,
            ml_readiness: true,
            // Parameters
            outlier_threshold: 1.5,
            outlier_method: OutlierMethod::Iqr,
            correlation_threshold: 0.9,
            pca_max_components: 10,
            max_categories: 50,
            max_plot_columns: 20,
            max_cluster_k: 10,
            tsne_perplexity: 30.0,
            bootstrap_iterations: 1000,
            max_sample_for_advanced: 5000,
            n_distribution_fits: 7,
        }
    }
}

impl AnalysisConfig {
    /// Minimal config – only descriptive stats.
    pub fn minimal() -> Self {
        Self {
            descriptive: true,
            correlation: false,
            distribution: false,
            missing: false,
            outlier: false,
            categorical: false,
            feature_importance: false,
            pca: false,
            duplicates: false,
            quality: false,
            preprocessing: false,
            advanced: false,
            advanced_distribution: false,
            advanced_correlation: false,
            clustering: false,
            advanced_dimreduction: false,
            feature_insights: false,
            advanced_anomaly: false,
            statistical_tests: false,
            data_profiling: false,
            insight_engine: false,
            cross_analysis: false,
            column_role: false,
            ml_readiness: false,
            ..Default::default()
        }
    }

    /// Fast config – skip heavy analyses (PCA, feature importance, advanced).
    pub fn fast() -> Self {
        Self {
            pca: false,
            feature_importance: false,
            advanced: false,
            advanced_distribution: false,
            advanced_correlation: false,
            clustering: false,
            advanced_dimreduction: false,
            feature_insights: false,
            advanced_anomaly: false,
            statistical_tests: false,
            data_profiling: false,
            insight_engine: false,
            cross_analysis: false,
            column_role: false,
            ml_readiness: false,
            ..Default::default()
        }
    }

    /// Basic-only – all basic analyses on, all advanced off.
    pub fn basic_only() -> Self {
        Self {
            advanced: false,
            advanced_distribution: false,
            advanced_correlation: false,
            clustering: false,
            advanced_dimreduction: false,
            feature_insights: false,
            advanced_anomaly: false,
            statistical_tests: false,
            data_profiling: false,
            insight_engine: false,
            cross_analysis: false,
            column_role: false,
            ml_readiness: false,
            ..Default::default()
        }
    }
}
