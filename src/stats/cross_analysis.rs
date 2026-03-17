use polars::prelude::*;
use serde::{Deserialize, Serialize};

use crate::core::schema::DataSchema;
use crate::stats::descriptive::DescriptiveStats;

// ─── Result types ───────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutlierClusterEntry {
    pub column: String,
    pub outlier_ratio_in_cluster: Vec<f64>,
    pub cluster_with_most_outliers: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MissingCorrelationEntry {
    pub col_a: String,
    pub col_b: String,
    pub jaccard_similarity: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimpsonEntry {
    pub feature: String,
    pub group: String,
    pub overall_direction: f64,
    pub subgroup_directions: Vec<f64>,
    pub is_paradox: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImportanceMissingEntry {
    pub column: String,
    pub variance_rank: f64,
    pub missing_ratio: f64,
    pub risk: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossAnalysisResult {
    pub outlier_cluster: Vec<OutlierClusterEntry>,
    pub missing_correlation: Vec<MissingCorrelationEntry>,
    pub simpson_candidates: Vec<SimpsonEntry>,
    pub importance_vs_missing: Vec<ImportanceMissingEntry>,
}

// ─── Implementation ─────────────────────────────────────────────────

pub struct CrossAnalysisStats<'a> {
    df: &'a DataFrame,
    schema: &'a DataSchema,
    max_pairs: usize,
}

impl<'a> CrossAnalysisStats<'a> {
    pub fn new(df: &'a DataFrame, schema: &'a DataSchema, max_pairs: usize) -> Self {
        Self {
            df,
            schema,
            max_pairs,
        }
    }

    pub fn compute(&self) -> CrossAnalysisResult {
        CrossAnalysisResult {
            outlier_cluster: self.outlier_cluster_cross(),
            missing_correlation: self.missing_correlation(),
            simpson_candidates: self.simpson_paradox_scan(),
            importance_vs_missing: self.importance_vs_missing(),
        }
    }

    // ── Outlier × Cluster interplay ─────────────────────────────

    fn outlier_cluster_cross(&self) -> Vec<OutlierClusterEntry> {
        let num_cols = self.schema.numeric_columns();
        let n = self.df.height();
        if num_cols.len() < 2 || n < 20 {
            return vec![];
        }

        // Simplified: split data into 3 equal bins per column and check outlier proportions
        let mut results = Vec::new();
        for &col_name in &num_cols {
            if let Some(vals) = self
                .df
                .column(col_name)
                .ok()
                .and_then(|c| DescriptiveStats::column_to_f64_vec(c))
            {
                let mut sorted = vals.clone();
                sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

                let q1 = sorted[n / 4];
                let q3 = sorted[3 * n / 4];
                let iqr = q3 - q1;
                let lower = q1 - 1.5 * iqr;
                let upper = q3 + 1.5 * iqr;

                // Split into 3 clusters by tertiles
                let t1 = sorted[n / 3];
                let t2 = sorted[2 * n / 3];

                let mut cluster_outlier_counts = [0usize; 3];
                let mut cluster_sizes = [0usize; 3];

                for &v in &vals {
                    let cluster = if v <= t1 {
                        0
                    } else if v <= t2 {
                        1
                    } else {
                        2
                    };
                    cluster_sizes[cluster] += 1;
                    if v < lower || v > upper {
                        cluster_outlier_counts[cluster] += 1;
                    }
                }

                let ratios: Vec<f64> = (0..3)
                    .map(|c| {
                        if cluster_sizes[c] > 0 {
                            cluster_outlier_counts[c] as f64 / cluster_sizes[c] as f64
                        } else {
                            0.0
                        }
                    })
                    .collect();

                let max_idx = ratios
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(i, _)| i)
                    .unwrap_or(0);

                results.push(OutlierClusterEntry {
                    column: col_name.to_string(),
                    outlier_ratio_in_cluster: ratios,
                    cluster_with_most_outliers: max_idx,
                });
            }
        }
        results.truncate(self.max_pairs);
        results
    }

    // ── Missing-value correlation (Jaccard similarity on NaN masks) ──

    fn missing_correlation(&self) -> Vec<MissingCorrelationEntry> {
        // Columns with at least some missing values
        let cols_with_missing: Vec<(&str, Vec<bool>)> = self
            .schema
            .columns
            .iter()
            .filter(|c| c.n_missing > 0)
            .filter_map(|c| {
                let col = self.df.column(&c.name).ok()?;
                let mask: Vec<bool> = (0..col.len())
                    .map(|i| col.get(i).ok().map(|v| v == AnyValue::Null).unwrap_or(true))
                    .collect();
                Some((c.name.as_str(), mask))
            })
            .collect();

        let mut results = Vec::new();
        for i in 0..cols_with_missing.len() {
            for j in (i + 1)..cols_with_missing.len() {
                let (na, ma) = &cols_with_missing[i];
                let (nb, mb) = &cols_with_missing[j];

                let jaccard = Self::jaccard_bool(ma, mb);
                if jaccard > 0.1 {
                    results.push(MissingCorrelationEntry {
                        col_a: na.to_string(),
                        col_b: nb.to_string(),
                        jaccard_similarity: jaccard,
                    });
                }
            }
        }

        results.sort_by(|a, b| {
            b.jaccard_similarity
                .partial_cmp(&a.jaccard_similarity)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        results.truncate(self.max_pairs);
        results
    }

    // ── Simpson's paradox scan ──────────────────────────────────

    fn simpson_paradox_scan(&self) -> Vec<SimpsonEntry> {
        let num_cols = self.schema.numeric_columns();
        let cat_cols = self.schema.categorical_columns();
        if num_cols.len() < 2 || cat_cols.is_empty() {
            return vec![];
        }

        let mut results = Vec::new();

        // For each pair (numeric_x, numeric_y), grouped by each categorical
        for ci in 0..num_cols.len().min(5) {
            for cj in (ci + 1)..num_cols.len().min(5) {
                let x_name = num_cols[ci];
                let y_name = num_cols[cj];

                let x_vals = match self
                    .df
                    .column(x_name)
                    .ok()
                    .and_then(|c| DescriptiveStats::column_to_f64_vec(c))
                {
                    Some(v) => v,
                    None => continue,
                };
                let y_vals = match self
                    .df
                    .column(y_name)
                    .ok()
                    .and_then(|c| DescriptiveStats::column_to_f64_vec(c))
                {
                    Some(v) => v,
                    None => continue,
                };

                let min_len = x_vals.len().min(y_vals.len());
                let overall_corr = Self::pearson(&x_vals[..min_len], &y_vals[..min_len]);

                for &cat_name in &cat_cols {
                    if let Ok(cat_col) = self.df.column(cat_name) {
                        // Group by category
                        let groups = Self::group_indices(cat_col);
                        if groups.len() < 2 || groups.len() > 10 {
                            continue;
                        }

                        let sub_corrs: Vec<f64> = groups
                            .iter()
                            .map(|(_, indices)| {
                                let sx: Vec<f64> = indices
                                    .iter()
                                    .filter(|&&i| i < min_len)
                                    .map(|&i| x_vals[i])
                                    .collect();
                                let sy: Vec<f64> = indices
                                    .iter()
                                    .filter(|&&i| i < min_len)
                                    .map(|&i| y_vals[i])
                                    .collect();
                                if sx.len() > 5 {
                                    Self::pearson(&sx, &sy)
                                } else {
                                    0.0
                                }
                            })
                            .collect();

                        // Check for paradox: overall direction differs from subgroup directions
                        let paradox = sub_corrs.iter().all(|&r| r > 0.0 && overall_corr < -0.1)
                            || sub_corrs.iter().all(|&r| r < 0.0 && overall_corr > 0.1);

                        if paradox {
                            results.push(SimpsonEntry {
                                feature: format!("{} vs {}", x_name, y_name),
                                group: cat_name.to_string(),
                                overall_direction: overall_corr,
                                subgroup_directions: sub_corrs,
                                is_paradox: true,
                            });
                        }
                    }
                }
            }
        }

        results
    }

    // ── Feature importance vs missing ───────────────────────────

    fn importance_vs_missing(&self) -> Vec<ImportanceMissingEntry> {
        let num_cols = self.schema.numeric_columns();
        if num_cols.is_empty() {
            return vec![];
        }

        // Compute variance as a proxy for importance
        let mut var_entries: Vec<(String, f64, f64)> = Vec::new();
        for &col_name in &num_cols {
            if let Some(vals) = self
                .df
                .column(col_name)
                .ok()
                .and_then(|c| DescriptiveStats::column_to_f64_vec(c))
            {
                if vals.is_empty() {
                    continue;
                }
                let mean = vals.iter().sum::<f64>() / vals.len() as f64;
                let var = vals.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / vals.len() as f64;

                let missing_ratio = self
                    .schema
                    .columns
                    .iter()
                    .find(|c| c.name == col_name)
                    .map(|c| c.missing_ratio)
                    .unwrap_or(0.0);

                var_entries.push((col_name.to_string(), var, missing_ratio));
            }
        }

        if var_entries.is_empty() {
            return vec![];
        }

        // Rank by variance (higher = more important)
        var_entries.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        var_entries
            .iter()
            .enumerate()
            .map(|(rank, (name, _var, missing_ratio))| {
                let variance_rank = (rank + 1) as f64 / var_entries.len() as f64;
                let risk = if variance_rank < 0.3 && *missing_ratio > 0.1 {
                    "high"
                } else if variance_rank < 0.5 && *missing_ratio > 0.05 {
                    "medium"
                } else {
                    "low"
                };
                ImportanceMissingEntry {
                    column: name.clone(),
                    variance_rank,
                    missing_ratio: *missing_ratio,
                    risk: risk.to_string(),
                }
            })
            .collect()
    }

    // ── Helpers ─────────────────────────────────────────────────

    fn jaccard_bool(a: &[bool], b: &[bool]) -> f64 {
        let min_len = a.len().min(b.len());
        let mut inter = 0usize;
        let mut union = 0usize;
        for i in 0..min_len {
            if a[i] || b[i] {
                union += 1;
            }
            if a[i] && b[i] {
                inter += 1;
            }
        }
        if union == 0 {
            0.0
        } else {
            inter as f64 / union as f64
        }
    }

    fn group_indices(col: &Column) -> Vec<(String, Vec<usize>)> {
        let mut groups: std::collections::HashMap<String, Vec<usize>> =
            std::collections::HashMap::new();
        for i in 0..col.len() {
            if let Ok(val) = col.get(i) {
                let key = format!("{}", val);
                groups.entry(key).or_default().push(i);
            }
        }
        groups.into_iter().collect()
    }

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
}
