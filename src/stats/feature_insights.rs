use polars::prelude::*;
use serde::{Deserialize, Serialize};

use crate::core::schema::DataSchema;
use crate::stats::descriptive::DescriptiveStats;

// ─── Result types ───────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InteractionEntry {
    pub col_a: String,
    pub col_b: String,
    pub interaction_strength: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonotonicEntry {
    pub col_a: String,
    pub col_b: String,
    pub pearson: f64,
    pub spearman: f64,
    pub gap: f64,
    pub is_nonlinear_monotonic: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BinningEntry {
    pub column: String,
    pub equal_width_entropy: f64,
    pub equal_freq_entropy: f64,
    pub recommended: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CardinalityEntry {
    pub column: String,
    pub cardinality: usize,
    pub cardinality_ratio: f64,
    pub recommended_encoding: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LeakageEntry {
    pub column: String,
    pub risk: String, // "low", "medium", "high"
    pub reason: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureInsightsResult {
    pub interactions: Vec<InteractionEntry>,
    pub monotonic: Vec<MonotonicEntry>,
    pub binning: Vec<BinningEntry>,
    pub cardinality: Vec<CardinalityEntry>,
    pub leakage: Vec<LeakageEntry>,
}

// ─── Implementation ─────────────────────────────────────────────────

pub struct FeatureInsightsStats<'a> {
    df: &'a DataFrame,
    schema: &'a DataSchema,
    max_sample: usize,
}

impl<'a> FeatureInsightsStats<'a> {
    pub fn new(df: &'a DataFrame, schema: &'a DataSchema, max_sample: usize) -> Self {
        Self {
            df,
            schema,
            max_sample,
        }
    }

    pub fn compute(&self) -> FeatureInsightsResult {
        let interactions = self.interaction_detection();
        let monotonic = self.monotonic_detection();
        let binning = self.binning_analysis(10);
        let cardinality = self.cardinality_analysis();
        let leakage = self.leakage_detection();

        FeatureInsightsResult {
            interactions,
            monotonic,
            binning,
            cardinality,
            leakage,
        }
    }

    fn interaction_detection(&self) -> Vec<InteractionEntry> {
        let num_cols = self.schema.numeric_columns();
        if num_cols.len() < 3 {
            return vec![];
        }

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
        let sample_len = min_len.min(self.max_sample);
        let mut results = Vec::new();

        for i in 0..col_vals.len() {
            for j in (i + 1)..col_vals.len() {
                // Compute product interaction correlated with other features
                let product: Vec<f64> = (0..sample_len)
                    .map(|k| col_vals[i].1[k] * col_vals[j].1[k])
                    .collect();

                let mut max_corr = 0.0f64;
                for k in 0..col_vals.len() {
                    if k == i || k == j {
                        continue;
                    }
                    let r = Self::pearson(&product, &col_vals[k].1[..sample_len]);
                    if r.abs() > max_corr.abs() {
                        max_corr = r;
                    }
                }

                if max_corr.abs() > 0.3 {
                    results.push(InteractionEntry {
                        col_a: col_vals[i].0.to_string(),
                        col_b: col_vals[j].0.to_string(),
                        interaction_strength: max_corr.abs(),
                    });
                }
            }
        }

        results.sort_by(|a, b| {
            b.interaction_strength
                .partial_cmp(&a.interaction_strength)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        results.truncate(20);
        results
    }

    fn monotonic_detection(&self) -> Vec<MonotonicEntry> {
        let num_cols = self.schema.numeric_columns();
        if num_cols.len() < 2 {
            return vec![];
        }

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
        let mut results = Vec::new();

        for i in 0..col_vals.len() {
            for j in (i + 1)..col_vals.len() {
                let pearson = Self::pearson(&col_vals[i].1[..min_len], &col_vals[j].1[..min_len]);
                let spearman = Self::spearman(&col_vals[i].1[..min_len], &col_vals[j].1[..min_len]);
                let gap = (spearman.abs() - pearson.abs()).abs();

                results.push(MonotonicEntry {
                    col_a: col_vals[i].0.to_string(),
                    col_b: col_vals[j].0.to_string(),
                    pearson,
                    spearman,
                    gap,
                    is_nonlinear_monotonic: gap > 0.1 && spearman.abs() > 0.5,
                });
            }
        }
        results
    }

    fn binning_analysis(&self, n_bins: usize) -> Vec<BinningEntry> {
        let num_cols = self.schema.numeric_columns();

        num_cols
            .iter()
            .filter_map(|&col_name| {
                let col = self.df.column(col_name).ok()?;
                let values = DescriptiveStats::column_to_f64_vec(col)?;
                if values.len() < n_bins * 2 {
                    return None;
                }

                let ew_entropy = Self::equal_width_entropy(&values, n_bins);
                let ef_entropy = Self::equal_freq_entropy(&values, n_bins);

                let recommended = if ef_entropy < ew_entropy {
                    "equal_frequency"
                } else {
                    "equal_width"
                };

                Some(BinningEntry {
                    column: col_name.to_string(),
                    equal_width_entropy: ew_entropy,
                    equal_freq_entropy: ef_entropy,
                    recommended: recommended.to_string(),
                })
            })
            .collect()
    }

    fn cardinality_analysis(&self) -> Vec<CardinalityEntry> {
        let n_rows = self.df.height();

        self.schema
            .categorical_columns()
            .iter()
            .filter_map(|&col_name| {
                let col = self.df.column(col_name).ok()?;
                let cardinality = col.n_unique().unwrap_or(0);
                let cardinality_ratio = if n_rows > 0 {
                    cardinality as f64 / n_rows as f64
                } else {
                    0.0
                };

                let recommended_encoding = if cardinality <= 2 {
                    "binary"
                } else if cardinality <= 10 {
                    "one_hot"
                } else if cardinality <= 50 {
                    "target_encoding"
                } else {
                    "hash_encoding"
                };

                Some(CardinalityEntry {
                    column: col_name.to_string(),
                    cardinality,
                    cardinality_ratio,
                    recommended_encoding: recommended_encoding.to_string(),
                })
            })
            .collect()
    }

    fn leakage_detection(&self) -> Vec<LeakageEntry> {
        let n_rows = self.df.height();
        let mut entries = Vec::new();

        for info in &self.schema.columns {
            let mut risk = "low".to_string();
            let mut reason = String::new();

            // High cardinality + unique → possible ID leakage
            if info.n_unique as f64 / n_rows.max(1) as f64 > 0.99 && info.n_missing == 0 {
                risk = "high".to_string();
                reason = "Near-unique column (possible ID or future info leakage)".into();
            }
            // Perfect or near-zero entropy in target-like columns
            else if info.missing_ratio > 0.5 {
                risk = "medium".to_string();
                reason =
                    "High missing ratio may indicate data not available at prediction time".into();
            }

            if risk != "low" {
                entries.push(LeakageEntry {
                    column: info.name.clone(),
                    risk,
                    reason,
                });
            }
        }

        entries
    }

    // ── Helpers ─────────────────────────────────────────────────

    fn pearson(x: &[f64], y: &[f64]) -> f64 {
        let n = x.len().min(y.len()) as f64;
        if n < 2.0 {
            return 0.0;
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

    fn spearman(x: &[f64], y: &[f64]) -> f64 {
        let rx = Self::rank(x);
        let ry = Self::rank(y);
        Self::pearson(&rx, &ry)
    }

    fn rank(values: &[f64]) -> Vec<f64> {
        let mut indexed: Vec<(usize, f64)> = values.iter().copied().enumerate().collect();
        indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        let mut ranks = vec![0.0f64; values.len()];
        let mut i = 0;
        while i < indexed.len() {
            let mut j = i + 1;
            while j < indexed.len() && (indexed[j].1 - indexed[i].1).abs() < f64::EPSILON {
                j += 1;
            }
            let avg = (i..j).map(|k| k + 1).sum::<usize>() as f64 / (j - i) as f64;
            for k in i..j {
                ranks[indexed[k].0] = avg;
            }
            i = j;
        }
        ranks
    }

    fn equal_width_entropy(values: &[f64], n_bins: usize) -> f64 {
        let min = values.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let range = (max - min).max(f64::EPSILON);

        let mut bins = vec![0usize; n_bins];
        for &v in values {
            let idx = ((v - min) / range * (n_bins - 1) as f64) as usize;
            bins[idx.min(n_bins - 1)] += 1;
        }

        let n = values.len() as f64;
        bins.iter()
            .filter(|&&c| c > 0)
            .map(|&c| {
                let p = c as f64 / n;
                -p * p.log2()
            })
            .sum()
    }

    fn equal_freq_entropy(values: &[f64], n_bins: usize) -> f64 {
        let mut sorted = values.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let bin_size = (sorted.len() / n_bins).max(1);

        let mut bins = Vec::new();
        for chunk in sorted.chunks(bin_size) {
            bins.push(chunk.len());
        }

        let n = values.len() as f64;
        bins.iter()
            .filter(|&&c| c > 0)
            .map(|&c| {
                let p = c as f64 / n;
                -p * p.log2()
            })
            .sum()
    }
}
