use polars::prelude::*;
use serde::{Deserialize, Serialize};

use crate::core::schema::DataSchema;
use crate::stats::descriptive::DescriptiveStats;

// ─── Result types ───────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureRanking {
    pub column: String,
    pub variance: f64,
    pub std: f64,
    pub cv: f64, // coefficient of variation
    pub range: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeanAbsCorrelation {
    pub column: String,
    pub mean_abs_corr: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureImportanceResult {
    pub variance_ranking: Vec<FeatureRanking>,
    pub mean_abs_correlations: Vec<MeanAbsCorrelation>,
}

// ─── Implementation ─────────────────────────────────────────────────

pub struct FeatureImportanceStats<'a> {
    df: &'a DataFrame,
    schema: &'a DataSchema,
}

impl<'a> FeatureImportanceStats<'a> {
    pub fn new(df: &'a DataFrame, schema: &'a DataSchema) -> Self {
        Self { df, schema }
    }

    pub fn compute(&self) -> FeatureImportanceResult {
        let variance_ranking = self.variance_ranking();
        let mean_abs_correlations = self.mean_abs_correlation();

        FeatureImportanceResult {
            variance_ranking,
            mean_abs_correlations,
        }
    }

    fn variance_ranking(&self) -> Vec<FeatureRanking> {
        let num_cols = self.schema.numeric_columns();
        let mut rankings: Vec<FeatureRanking> = num_cols
            .iter()
            .filter_map(|&col_name| {
                let col = self.df.column(col_name).ok()?;
                let values = DescriptiveStats::column_to_f64_vec(col)?;
                let n = values.len() as f64;
                if n < 2.0 {
                    return None;
                }

                let mean = values.iter().sum::<f64>() / n;
                let variance = values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0);
                let std = variance.sqrt();
                let cv = if mean.abs() > f64::EPSILON {
                    std / mean.abs()
                } else {
                    f64::NAN
                };
                let min = values.iter().cloned().fold(f64::INFINITY, f64::min);
                let max = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

                Some(FeatureRanking {
                    column: col_name.to_string(),
                    variance,
                    std,
                    cv,
                    range: max - min,
                })
            })
            .collect();

        rankings.sort_by(|a, b| {
            b.variance
                .partial_cmp(&a.variance)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        rankings
    }

    fn mean_abs_correlation(&self) -> Vec<MeanAbsCorrelation> {
        let num_cols = self.schema.numeric_columns();
        if num_cols.len() < 2 {
            return vec![];
        }

        // Compute pairwise Pearson correlations
        let mut col_vals: Vec<(&str, Vec<f64>)> = Vec::new();
        for &col_name in &num_cols {
            if let Some(vals) = self
                .df
                .column(col_name)
                .ok()
                .and_then(|c| DescriptiveStats::column_to_f64_vec(c))
            {
                col_vals.push((col_name, vals));
            }
        }

        let min_len = col_vals.iter().map(|(_, v)| v.len()).min().unwrap_or(0);
        let p = col_vals.len();

        let mut results: Vec<MeanAbsCorrelation> = Vec::new();

        for i in 0..p {
            let mut sum_abs_r = 0.0;
            let mut count = 0;
            for j in 0..p {
                if i == j {
                    continue;
                }
                let r = Self::pearson(&col_vals[i].1[..min_len], &col_vals[j].1[..min_len]);
                if r.is_finite() {
                    sum_abs_r += r.abs();
                    count += 1;
                }
            }
            results.push(MeanAbsCorrelation {
                column: col_vals[i].0.to_string(),
                mean_abs_corr: if count > 0 {
                    sum_abs_r / count as f64
                } else {
                    0.0
                },
            });
        }

        results.sort_by(|a, b| {
            b.mean_abs_corr
                .partial_cmp(&a.mean_abs_corr)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        results
    }

    fn pearson(x: &[f64], y: &[f64]) -> f64 {
        let n = x.len().min(y.len()) as f64;
        if n < 2.0 {
            return f64::NAN;
        }
        let mx = x.iter().sum::<f64>() / n;
        let my = y.iter().sum::<f64>() / n;
        let mut cov = 0.0;
        let mut sx = 0.0;
        let mut sy = 0.0;
        for i in 0..x.len().min(y.len()) {
            let dx = x[i] - mx;
            let dy = y[i] - my;
            cov += dx * dy;
            sx += dx * dx;
            sy += dy * dy;
        }
        if sx < f64::EPSILON || sy < f64::EPSILON {
            return 0.0;
        }
        cov / (sx * sy).sqrt()
    }
}
