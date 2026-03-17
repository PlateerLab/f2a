use ndarray::Array2;
use polars::prelude::*;
use serde::{Deserialize, Serialize};

use crate::core::schema::DataSchema;
use crate::stats::descriptive::DescriptiveStats;

// ─── Result types ───────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnomalyMethodResult {
    pub method: String,
    pub n_anomalies: usize,
    pub anomaly_ratio: f64,
    pub scores: Vec<f64>,
    pub labels: Vec<bool>, // true = anomaly
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsensusAnomaly {
    pub n_anomalies: usize,
    pub anomaly_ratio: f64,
    pub labels: Vec<bool>,
    pub vote_counts: Vec<u8>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdvancedAnomalyResult {
    pub isolation_forest: Option<AnomalyMethodResult>,
    pub local_outlier_factor: Option<AnomalyMethodResult>,
    pub mahalanobis: Option<AnomalyMethodResult>,
    pub consensus: Option<ConsensusAnomaly>,
}

// ─── Implementation ─────────────────────────────────────────────────

pub struct AdvancedAnomalyStats<'a> {
    df: &'a DataFrame,
    schema: &'a DataSchema,
    max_sample: usize,
    contamination: f64,
}

impl<'a> AdvancedAnomalyStats<'a> {
    pub fn new(
        df: &'a DataFrame,
        schema: &'a DataSchema,
        max_sample: usize,
        contamination: f64,
    ) -> Self {
        Self {
            df,
            schema,
            max_sample,
            contamination,
        }
    }

    pub fn compute(&self) -> AdvancedAnomalyResult {
        let matrix = match self.prepare_data() {
            Some(m) => m,
            None => {
                return AdvancedAnomalyResult {
                    isolation_forest: None,
                    local_outlier_factor: None,
                    mahalanobis: None,
                    consensus: None,
                };
            }
        };

        let isolation_forest = self.isolation_forest(&matrix);
        let local_outlier_factor = self.local_outlier_factor(&matrix);
        let mahalanobis = self.mahalanobis_distance(&matrix);

        let consensus = self.compute_consensus(
            isolation_forest.as_ref(),
            local_outlier_factor.as_ref(),
            mahalanobis.as_ref(),
        );

        AdvancedAnomalyResult {
            isolation_forest,
            local_outlier_factor,
            mahalanobis,
            consensus,
        }
    }

    /// Isolation Forest: anomaly scores via random binary trees.
    fn isolation_forest(&self, data: &Array2<f64>) -> Option<AnomalyMethodResult> {
        let n = data.nrows();
        let d = data.ncols();
        if n < 10 || d == 0 {
            return None;
        }

        let n_trees = 100;
        let sample_size = n.min(256);
        let mut scores = vec![0.0f64; n];

        // Build multiple isolation trees and average path lengths
        for tree_idx in 0..n_trees {
            // Subsample indices (deterministic based on tree_idx)
            let step = (n as f64 / sample_size as f64).max(1.0);
            let indices: Vec<usize> = (0..sample_size)
                .map(|i| ((i as f64 * step + tree_idx as f64) as usize) % n)
                .collect();

            // For each point, compute isolation path length
            for i in 0..n {
                let path_len = self.isolation_path_length(data, i, &indices, d, 0, 10);
                scores[i] += path_len;
            }
        }

        // Average and normalize
        let c_n = Self::average_path_length(sample_size);
        for s in scores.iter_mut() {
            *s /= n_trees as f64;
            // Anomaly score: s(x) = 2^(-E[h(x)] / c(n))
            *s = 2.0f64.powf(-(*s) / c_n);
        }

        // Threshold: top `contamination` fraction
        let mut sorted_scores = scores.clone();
        sorted_scores.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        let threshold_idx = (n as f64 * self.contamination) as usize;
        let threshold = sorted_scores
            .get(threshold_idx.min(sorted_scores.len().saturating_sub(1)))
            .copied()
            .unwrap_or(0.5);

        let labels: Vec<bool> = scores.iter().map(|&s| s >= threshold).collect();
        let n_anomalies = labels.iter().filter(|&&l| l).count();

        Some(AnomalyMethodResult {
            method: "isolation_forest".into(),
            n_anomalies,
            anomaly_ratio: n_anomalies as f64 / n as f64,
            scores,
            labels,
        })
    }

    fn isolation_path_length(
        &self,
        data: &Array2<f64>,
        point_idx: usize,
        subset: &[usize],
        n_features: usize,
        depth: usize,
        max_depth: usize,
    ) -> f64 {
        if depth >= max_depth || subset.len() <= 1 {
            return depth as f64 + Self::average_path_length(subset.len());
        }

        // Pick a random feature and split point
        let feat = (point_idx + depth) % n_features;
        let vals: Vec<f64> = subset.iter().map(|&i| data[[i, feat]]).collect();
        let min_val = vals.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_val = vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        if (max_val - min_val).abs() < f64::EPSILON {
            return depth as f64;
        }

        let split = (min_val + max_val) / 2.0;
        let point_val = data[[point_idx, feat]];

        let left: Vec<usize> = subset
            .iter()
            .filter(|&&i| data[[i, feat]] < split)
            .cloned()
            .collect();
        let right: Vec<usize> = subset
            .iter()
            .filter(|&&i| data[[i, feat]] >= split)
            .cloned()
            .collect();

        if point_val < split {
            self.isolation_path_length(data, point_idx, &left, n_features, depth + 1, max_depth)
        } else {
            self.isolation_path_length(data, point_idx, &right, n_features, depth + 1, max_depth)
        }
    }

    fn average_path_length(n: usize) -> f64 {
        if n <= 1 {
            return 0.0;
        }
        let nf = n as f64;
        2.0 * ((nf - 1.0).ln() + 0.5772156649) - 2.0 * (nf - 1.0) / nf
    }

    /// Local Outlier Factor (simplified).
    fn local_outlier_factor(&self, data: &Array2<f64>) -> Option<AnomalyMethodResult> {
        let n = data.nrows();
        if n < 20 {
            return None;
        }

        let k = ((n as f64).sqrt().ceil() as usize).min(20).max(5);

        // Compute k-nearest neighbor distances
        let mut knn_dists = vec![Vec::new(); n];
        let mut k_dist = vec![0.0f64; n];

        for i in 0..n {
            let mut dists: Vec<(usize, f64)> = (0..n)
                .filter(|&j| j != i)
                .map(|j| {
                    let d: f64 = data
                        .row(i)
                        .iter()
                        .zip(data.row(j).iter())
                        .map(|(a, b)| (a - b).powi(2))
                        .sum::<f64>()
                        .sqrt();
                    (j, d)
                })
                .collect();
            dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
            let knn: Vec<(usize, f64)> = dists.into_iter().take(k).collect();
            k_dist[i] = knn.last().map(|x| x.1).unwrap_or(0.0);
            knn_dists[i] = knn;
        }

        // Local reachability density
        let mut lrd = vec![0.0f64; n];
        for i in 0..n {
            let reach_sum: f64 = knn_dists[i].iter().map(|(j, d)| d.max(k_dist[*j])).sum();
            lrd[i] = if reach_sum > f64::EPSILON {
                k as f64 / reach_sum
            } else {
                1.0
            };
        }

        // LOF scores
        let scores: Vec<f64> = (0..n)
            .map(|i| {
                let lof: f64 = knn_dists[i]
                    .iter()
                    .map(|(j, _)| {
                        if lrd[i] > f64::EPSILON {
                            lrd[*j] / lrd[i]
                        } else {
                            1.0
                        }
                    })
                    .sum::<f64>()
                    / k as f64;
                lof
            })
            .collect();

        // Threshold
        let mut sorted_scores = scores.clone();
        sorted_scores.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        let threshold_idx = (n as f64 * self.contamination) as usize;
        let threshold = sorted_scores
            .get(threshold_idx.min(sorted_scores.len().saturating_sub(1)))
            .copied()
            .unwrap_or(1.5);

        let labels: Vec<bool> = scores.iter().map(|&s| s >= threshold).collect();
        let n_anomalies = labels.iter().filter(|&&l| l).count();

        Some(AnomalyMethodResult {
            method: "local_outlier_factor".into(),
            n_anomalies,
            anomaly_ratio: n_anomalies as f64 / n as f64,
            scores,
            labels,
        })
    }

    /// Mahalanobis distance based anomaly detection.
    fn mahalanobis_distance(&self, data: &Array2<f64>) -> Option<AnomalyMethodResult> {
        let n = data.nrows();
        let d = data.ncols();
        if n <= d + 1 || d == 0 {
            return None;
        }

        // Compute mean and covariance
        let means: Vec<f64> = (0..d)
            .map(|j| data.column(j).mean().unwrap_or(0.0))
            .collect();

        let mut cov = Array2::<f64>::zeros((d, d));
        for i in 0..d {
            for j in i..d {
                let val: f64 = (0..n)
                    .map(|k| (data[[k, i]] - means[i]) * (data[[k, j]] - means[j]))
                    .sum::<f64>()
                    / (n as f64 - 1.0);
                cov[[i, j]] = val;
                cov[[j, i]] = val;
            }
        }

        // Regularize covariance (add small diagonal)
        for i in 0..d {
            cov[[i, i]] += 1e-6;
        }

        // Invert covariance using Gaussian elimination
        let cov_inv = match Self::invert_matrix(&cov) {
            Some(inv) => inv,
            None => return None,
        };

        // Compute Mahalanobis distances
        let scores: Vec<f64> = (0..n)
            .map(|i| {
                let diff: Vec<f64> = (0..d).map(|j| data[[i, j]] - means[j]).collect();
                let mut md = 0.0;
                for j in 0..d {
                    for k in 0..d {
                        md += diff[j] * cov_inv[[j, k]] * diff[k];
                    }
                }
                md.sqrt()
            })
            .collect();

        // Chi-square threshold at df=d, alpha=contamination
        let threshold = (d as f64 * (1.0 + self.contamination * 3.0)).sqrt() * 2.0;

        let labels: Vec<bool> = scores.iter().map(|&s| s > threshold).collect();
        let n_anomalies = labels.iter().filter(|&&l| l).count();

        Some(AnomalyMethodResult {
            method: "mahalanobis".into(),
            n_anomalies,
            anomaly_ratio: n_anomalies as f64 / n as f64,
            scores,
            labels,
        })
    }

    /// Consensus: anomaly if ≥2 of 3 methods agree.
    fn compute_consensus(
        &self,
        if_result: Option<&AnomalyMethodResult>,
        lof_result: Option<&AnomalyMethodResult>,
        mah_result: Option<&AnomalyMethodResult>,
    ) -> Option<ConsensusAnomaly> {
        let _n = self.df.height().min(self.max_sample);
        let methods: Vec<&AnomalyMethodResult> = [if_result, lof_result, mah_result]
            .iter()
            .filter_map(|r| *r)
            .collect();

        if methods.len() < 2 {
            return None;
        }

        let min_n = methods.iter().map(|m| m.labels.len()).min().unwrap_or(0);

        let vote_counts: Vec<u8> = (0..min_n)
            .map(|i| {
                methods
                    .iter()
                    .map(|m| {
                        if m.labels.get(i).copied().unwrap_or(false) {
                            1u8
                        } else {
                            0u8
                        }
                    })
                    .sum()
            })
            .collect();

        let labels: Vec<bool> = vote_counts.iter().map(|&v| v >= 2).collect();
        let n_anomalies = labels.iter().filter(|&&l| l).count();

        Some(ConsensusAnomaly {
            n_anomalies,
            anomaly_ratio: if min_n > 0 {
                n_anomalies as f64 / min_n as f64
            } else {
                0.0
            },
            labels,
            vote_counts,
        })
    }

    // ── Helpers ─────────────────────────────────────────────────

    fn prepare_data(&self) -> Option<Array2<f64>> {
        let num_cols = self.schema.numeric_columns();
        if num_cols.len() < 2 {
            return None;
        }

        let mut col_data: Vec<Vec<f64>> = Vec::new();
        for &col_name in &num_cols {
            if let Some(vals) = self
                .df
                .column(col_name)
                .ok()
                .and_then(|c| DescriptiveStats::column_to_f64_vec(c))
            {
                col_data.push(vals);
            }
        }

        if col_data.is_empty() {
            return None;
        }

        let min_len = col_data.iter().map(|v| v.len()).min().unwrap_or(0);
        let sample_len = min_len.min(self.max_sample);
        let step = (min_len / sample_len).max(1);
        let n_cols = col_data.len();

        let mut matrix = Array2::<f64>::zeros((sample_len, n_cols));
        for (j, data) in col_data.iter().enumerate() {
            for (i_out, i_in) in (0..min_len).step_by(step).take(sample_len).enumerate() {
                matrix[[i_out, j]] = data[i_in];
            }
        }

        // Standardize
        for j in 0..n_cols {
            let col = matrix.column(j);
            let mean = col.mean().unwrap_or(0.0);
            let var: f64 = col.iter().map(|x| (x - mean).powi(2)).sum::<f64>()
                / (sample_len as f64 - 1.0).max(1.0);
            let std = var.sqrt();
            if std > f64::EPSILON {
                for i in 0..sample_len {
                    matrix[[i, j]] = (matrix[[i, j]] - mean) / std;
                }
            }
        }

        Some(matrix)
    }

    pub(crate) fn invert_matrix(a: &Array2<f64>) -> Option<Array2<f64>> {
        let n = a.nrows();
        if n != a.ncols() {
            return None;
        }

        let mut aug = Array2::<f64>::zeros((n, 2 * n));
        for i in 0..n {
            for j in 0..n {
                aug[[i, j]] = a[[i, j]];
            }
            aug[[i, n + i]] = 1.0;
        }

        for col in 0..n {
            let mut max_row = col;
            let mut max_val = aug[[col, col]].abs();
            for row in (col + 1)..n {
                if aug[[row, col]].abs() > max_val {
                    max_val = aug[[row, col]].abs();
                    max_row = row;
                }
            }
            if max_val < 1e-10 {
                return None;
            }

            if max_row != col {
                for j in 0..(2 * n) {
                    let tmp = aug[[col, j]];
                    aug[[col, j]] = aug[[max_row, j]];
                    aug[[max_row, j]] = tmp;
                }
            }

            let pivot = aug[[col, col]];
            for j in 0..(2 * n) {
                aug[[col, j]] /= pivot;
            }

            for row in 0..n {
                if row == col {
                    continue;
                }
                let factor = aug[[row, col]];
                for j in 0..(2 * n) {
                    aug[[row, j]] -= factor * aug[[col, j]];
                }
            }
        }

        let mut inv = Array2::<f64>::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                inv[[i, j]] = aug[[i, n + j]];
            }
        }
        Some(inv)
    }
}
