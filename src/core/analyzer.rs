//! Top-level analysis orchestrator.
//!
//! `Analyzer::run` executes the entire pipeline:
//!   load → schema → preprocess → basic stats → advanced stats → collect results
//!
//! Results are returned as a `serde_json::Value` tree so that the PyO3
//! boundary can convert it straight into a Python dict.

use std::collections::HashMap;

use polars::prelude::*;
use serde::{Deserialize, Serialize};

use crate::core::config::AnalysisConfig;
use crate::core::loader::DataLoader;
use crate::core::preprocessor::{PreprocessingResult, Preprocessor};
use crate::core::schema::DataSchema;
use crate::utils::errors::F2aResult;

// Individual stats modules
use crate::stats::advanced_anomaly::AdvancedAnomalyStats;
use crate::stats::advanced_correlation::AdvancedCorrelationStats;
use crate::stats::advanced_dimreduction::AdvancedDimReductionStats;
use crate::stats::advanced_distribution::AdvancedDistributionStats;
use crate::stats::categorical::CategoricalStats;
use crate::stats::clustering::ClusteringStats;
use crate::stats::column_role::ColumnRoleClassifier;
use crate::stats::correlation::CorrelationStats;
use crate::stats::cross_analysis::CrossAnalysisStats;
use crate::stats::descriptive::DescriptiveStats;
use crate::stats::distribution::DistributionStats;
use crate::stats::duplicates::DuplicateStats;
use crate::stats::feature_importance::FeatureImportanceStats;
use crate::stats::feature_insights::FeatureInsightsStats;
use crate::stats::insight_engine::InsightEngine;
use crate::stats::missing::MissingStats;
use crate::stats::ml_readiness::MlReadinessStats;
use crate::stats::outlier::OutlierStats;
use crate::stats::pca::PcaStats;
use crate::stats::quality::QualityStats;
use crate::stats::statistical_tests::StatisticalTests;

// ─── AnalysisReport ─────────────────────────────────────────────────

/// Full analysis report – serialisable to JSON for the Python layer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalysisReport {
    pub source: String,
    pub schema: DataSchema,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub preprocessing: Option<PreprocessingResult>,
    pub config: AnalysisConfig,
    pub results: HashMap<String, serde_json::Value>,
}

// ─── Analyzer ───────────────────────────────────────────────────────

pub struct Analyzer;

impl Analyzer {
    /// Run the full analysis pipeline on a file path.
    pub fn run_from_file(source: &str, config: &AnalysisConfig) -> F2aResult<AnalysisReport> {
        let df = DataLoader::load(source)?;
        Self::run(source, df, config)
    }

    /// Run the full analysis pipeline on an already-loaded DataFrame.
    pub fn run(source: &str, df: DataFrame, config: &AnalysisConfig) -> F2aResult<AnalysisReport> {
        // ── 1. Schema ───────────────────────────────────────────
        let schema = DataSchema::from_dataframe(&df);

        // ── 2. Preprocessing ────────────────────────────────────
        let (work_df, preprocess_result) = if config.preprocessing {
            let (cleaned, result) = Preprocessor::process(&df, &schema, 0.95);
            (cleaned, Some(result))
        } else {
            (df.clone(), None)
        };

        // Re-derive schema on the cleaned frame
        let work_schema = if config.preprocessing {
            DataSchema::from_dataframe(&work_df)
        } else {
            schema.clone()
        };

        // ── 3. Compute all enabled analyses ─────────────────────
        let mut results: HashMap<String, serde_json::Value> = HashMap::new();

        // ── Basic analyses ──────────────────────────────────────
        if config.descriptive {
            let s = DescriptiveStats::new(&work_df, &work_schema);
            if let Ok(val) = serde_json::to_value(s.compute()) {
                results.insert("descriptive".into(), val);
            }
        }

        if config.correlation {
            let s = CorrelationStats::new(&work_df, &work_schema, config.correlation_threshold);
            if let Ok(val) = serde_json::to_value(s.compute()) {
                results.insert("correlation".into(), val);
            }
        }

        if config.distribution {
            let s = DistributionStats::new(&work_df, &work_schema);
            if let Ok(val) = serde_json::to_value(s.compute()) {
                results.insert("distribution".into(), val);
            }
        }

        if config.missing {
            let s = MissingStats::new(&work_df, &work_schema);
            if let Ok(val) = serde_json::to_value(s.compute()) {
                results.insert("missing".into(), val);
            }
        }

        if config.outlier {
            let s = OutlierStats::new(
                &work_df,
                &work_schema,
                config.outlier_method,
                config.outlier_threshold,
            );
            if let Ok(val) = serde_json::to_value(s.compute()) {
                results.insert("outlier".into(), val);
            }
        }

        if config.categorical {
            let s = CategoricalStats::new(&work_df, &work_schema, config.max_categories);
            if let Ok(val) = serde_json::to_value(s.compute()) {
                results.insert("categorical".into(), val);
            }
        }

        if config.duplicates {
            let s = DuplicateStats::new(&work_df, &work_schema);
            if let Ok(val) = serde_json::to_value(s.compute()) {
                results.insert("duplicates".into(), val);
            }
        }

        if config.quality {
            let s = QualityStats::new(&work_df, &work_schema);
            if let Ok(val) = serde_json::to_value(s.compute()) {
                results.insert("quality".into(), val);
            }
        }

        if config.feature_importance {
            let s = FeatureImportanceStats::new(&work_df, &work_schema);
            if let Ok(val) = serde_json::to_value(s.compute()) {
                results.insert("feature_importance".into(), val);
            }
        }

        if config.pca {
            let s = PcaStats::new(&work_df, &work_schema, config.pca_max_components);
            if let Some(pca_result) = s.compute() {
                if let Ok(val) = serde_json::to_value(pca_result) {
                    results.insert("pca".into(), val);
                }
            }
        }

        if config.statistical_tests {
            let s = StatisticalTests::new(&work_df, &work_schema);
            if let Ok(val) = serde_json::to_value(s.compute()) {
                results.insert("statistical_tests".into(), val);
            }
        }

        // ── Advanced analyses (gated by `config.advanced`) ──────
        if config.advanced {
            let max_sample = config.max_sample_for_advanced;

            if config.clustering {
                let s =
                    ClusteringStats::new(&work_df, &work_schema, config.max_cluster_k, max_sample);
                if let Ok(val) = serde_json::to_value(s.compute()) {
                    results.insert("clustering".into(), val);
                }
            }

            if config.advanced_anomaly {
                let s = AdvancedAnomalyStats::new(&work_df, &work_schema, max_sample, 0.05);
                if let Ok(val) = serde_json::to_value(s.compute()) {
                    results.insert("advanced_anomaly".into(), val);
                }
            }

            if config.advanced_correlation {
                let s = AdvancedCorrelationStats::new(
                    &work_df,
                    &work_schema,
                    config.bootstrap_iterations,
                    max_sample,
                );
                if let Ok(val) = serde_json::to_value(s.compute()) {
                    results.insert("advanced_correlation".into(), val);
                }
            }

            if config.advanced_distribution {
                let s = AdvancedDistributionStats::new(
                    &work_df,
                    &work_schema,
                    config.n_distribution_fits,
                    max_sample,
                );
                if let Ok(val) = serde_json::to_value(s.compute()) {
                    results.insert("advanced_distribution".into(), val);
                }
            }

            if config.advanced_dimreduction {
                let s = AdvancedDimReductionStats::new(
                    &work_df,
                    &work_schema,
                    config.tsne_perplexity,
                    max_sample,
                );
                if let Ok(val) = serde_json::to_value(s.compute()) {
                    results.insert("advanced_dimreduction".into(), val);
                }
            }

            if config.feature_insights {
                let s = FeatureInsightsStats::new(&work_df, &work_schema, max_sample);
                if let Ok(val) = serde_json::to_value(s.compute()) {
                    results.insert("feature_insights".into(), val);
                }
            }

            if config.insight_engine {
                let s = InsightEngine::new(&work_df, &work_schema);
                if let Ok(val) = serde_json::to_value(s.compute()) {
                    results.insert("insight_engine".into(), val);
                }
            }

            if config.column_role {
                let s = ColumnRoleClassifier::new(&work_df, &work_schema, None);
                if let Ok(val) = serde_json::to_value(s.compute()) {
                    results.insert("column_role".into(), val);
                }
            }

            if config.cross_analysis {
                let s = CrossAnalysisStats::new(&work_df, &work_schema, 30);
                if let Ok(val) = serde_json::to_value(s.compute()) {
                    results.insert("cross_analysis".into(), val);
                }
            }

            if config.ml_readiness {
                let s = MlReadinessStats::new(&work_df, &work_schema);
                if let Ok(val) = serde_json::to_value(s.compute()) {
                    results.insert("ml_readiness".into(), val);
                }
            }
        }

        Ok(AnalysisReport {
            source: source.to_string(),
            schema,
            preprocessing: preprocess_result,
            config: config.clone(),
            results,
        })
    }
}
