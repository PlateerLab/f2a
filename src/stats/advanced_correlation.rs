use ndarray::Array2;
use polars::prelude::*;
use serde::{Deserialize, Serialize};

use crate::core::schema::DataSchema;
use crate::stats::descriptive::DescriptiveStats;

// ─── Result types ───────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PartialCorrelationEntry {
    pub col_a: String,
    pub col_b: String,
    pub partial_r: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MutualInfoEntry {
    pub col_a: String,
    pub col_b: String,
    pub mi: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BootstrapCIEntry {
    pub col_a: String,
    pub col_b: String,
    pub r: f64,
    pub ci_lower: f64,
    pub ci_upper: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorrelationEdge {
    pub source: String,
    pub target: String,
    pub weight: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdvancedCorrelationResult {
    pub partial_correlations: Vec<PartialCorrelationEntry>,
    pub mutual_information: Vec<MutualInfoEntry>,
    pub bootstrap_ci: Vec<BootstrapCIEntry>,
    pub correlation_network: Vec<CorrelationEdge>,
    pub distance_correlations: Vec<(String, String, f64)>,
}

// ─── Implementation ─────────────────────────────────────────────────

pub struct AdvancedCorrelationStats<'a> {
    df: &'a DataFrame,
    schema: &'a DataSchema,
    bootstrap_iterations: usize,
    max_sample: usize,
}

impl<'a> AdvancedCorrelationStats<'a> {
    pub fn new(
        df: &'a DataFrame,
        schema: &'a DataSchema,
        bootstrap_iterations: usize,
        max_sample: usize,
    ) -> Self {
        Self {
            df,
            schema,
            bootstrap_iterations,
            max_sample,
        }
    }

    pub fn compute(&self) -> AdvancedCorrelationResult {
        let num_cols = self.schema.numeric_columns();
        let (names, matrix) = self.extract_matrix(&num_cols);

        let partial_correlations = if matrix.ncols() >= 3 {
            self.partial_correlation_matrix(&names, &matrix)
        } else {
            vec![]
        };

        let mutual_information = self.compute_mutual_information(&names, &matrix);
        let bootstrap_ci = self.bootstrap_correlation_ci(&names, &matrix);
        let correlation_network = self.build_correlation_network(&names, &matrix, 0.5);
        let distance_correlations = self.compute_distance_correlations(&names, &matrix);

        AdvancedCorrelationResult {
            partial_correlations,
            mutual_information,
            bootstrap_ci,
            correlation_network,
            distance_correlations,
        }
    }

    /// Partial correlation via precision matrix (inverse of correlation matrix).
    fn partial_correlation_matrix(
        &self,
        names: &[String],
        matrix: &Array2<f64>,
    ) -> Vec<PartialCorrelationEntry> {
        let p = matrix.ncols();
        let n = matrix.nrows();
        if p < 3 || n < p + 1 {
            return vec![];
        }

        // Compute correlation matrix
        let corr = self.pearson_matrix(matrix);

        // Regularize and invert
        let mut reg = corr.clone();
        for i in 0..p {
            reg[[i, i]] += 0.01;
        }

        let precision =
            match crate::stats::advanced_anomaly::AdvancedAnomalyStats::invert_matrix(&reg) {
                Some(inv) => inv,
                None => return vec![],
            };

        let mut results = Vec::new();
        for i in 0..p {
            for j in (i + 1)..p {
                let denom = (precision[[i, i]] * precision[[j, j]]).sqrt();
                let partial_r = if denom > f64::EPSILON {
                    -precision[[i, j]] / denom
                } else {
                    0.0
                };
                results.push(PartialCorrelationEntry {
                    col_a: names[i].clone(),
                    col_b: names[j].clone(),
                    partial_r: partial_r.clamp(-1.0, 1.0),
                });
            }
        }
        results
    }

    /// Mutual information estimation via histogram-based approach.
    fn compute_mutual_information(
        &self,
        names: &[String],
        matrix: &Array2<f64>,
    ) -> Vec<MutualInfoEntry> {
        let p = matrix.ncols();
        let n = matrix.nrows();
        if p < 2 || n < 20 {
            return vec![];
        }

        let n_bins = ((n as f64).sqrt().ceil() as usize).max(5).min(50);
        let mut results = Vec::new();

        for i in 0..p {
            for j in (i + 1)..p {
                let mi = Self::mutual_information_pair(
                    &matrix.column(i).to_vec(),
                    &matrix.column(j).to_vec(),
                    n_bins,
                );
                results.push(MutualInfoEntry {
                    col_a: names[i].clone(),
                    col_b: names[j].clone(),
                    mi,
                });
            }
        }
        results
    }

    fn mutual_information_pair(x: &[f64], y: &[f64], n_bins: usize) -> f64 {
        let n = x.len().min(y.len());
        if n == 0 {
            return 0.0;
        }

        let x_min = x.iter().cloned().fold(f64::INFINITY, f64::min);
        let x_max = x.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let y_min = y.iter().cloned().fold(f64::INFINITY, f64::min);
        let y_max = y.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        let x_range = (x_max - x_min).max(f64::EPSILON);
        let y_range = (y_max - y_min).max(f64::EPSILON);

        // Joint and marginal histograms
        let mut joint = vec![vec![0usize; n_bins]; n_bins];
        let mut x_hist = vec![0usize; n_bins];
        let mut y_hist = vec![0usize; n_bins];

        for k in 0..n {
            let xi = ((x[k] - x_min) / x_range * (n_bins - 1) as f64) as usize;
            let yi = ((y[k] - y_min) / y_range * (n_bins - 1) as f64) as usize;
            let xi = xi.min(n_bins - 1);
            let yi = yi.min(n_bins - 1);
            joint[xi][yi] += 1;
            x_hist[xi] += 1;
            y_hist[yi] += 1;
        }

        let nf = n as f64;
        let mut mi = 0.0f64;
        for i in 0..n_bins {
            for j in 0..n_bins {
                if joint[i][j] > 0 && x_hist[i] > 0 && y_hist[j] > 0 {
                    let pxy = joint[i][j] as f64 / nf;
                    let px = x_hist[i] as f64 / nf;
                    let py = y_hist[j] as f64 / nf;
                    mi += pxy * (pxy / (px * py)).ln();
                }
            }
        }

        mi.max(0.0)
    }

    /// Bootstrap confidence intervals for Pearson correlation.
    fn bootstrap_correlation_ci(
        &self,
        names: &[String],
        matrix: &Array2<f64>,
    ) -> Vec<BootstrapCIEntry> {
        let p = matrix.ncols();
        let n = matrix.nrows();
        if p < 2 || n < 20 {
            return vec![];
        }

        let n_iter = self.bootstrap_iterations.min(500);
        let mut results = Vec::new();

        for i in 0..p {
            for j in (i + 1)..p {
                let x = matrix.column(i);
                let y = matrix.column(j);
                let r = Self::pearson_vec(&x.to_vec(), &y.to_vec());

                // Bootstrap resampling
                let mut boot_rs: Vec<f64> = Vec::with_capacity(n_iter);
                for b in 0..n_iter {
                    // Deterministic pseudo-random indices
                    let boot_x: Vec<f64> = (0..n)
                        .map(|k| {
                            let idx = (k * 7 + b * 13 + 37) % n;
                            x[idx]
                        })
                        .collect();
                    let boot_y: Vec<f64> = (0..n)
                        .map(|k| {
                            let idx = (k * 7 + b * 13 + 37) % n;
                            y[idx]
                        })
                        .collect();
                    boot_rs.push(Self::pearson_vec(&boot_x, &boot_y));
                }

                boot_rs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                let ci_lower = boot_rs[(n_iter as f64 * 0.025) as usize];
                let ci_upper = boot_rs[(n_iter as f64 * 0.975) as usize];

                results.push(BootstrapCIEntry {
                    col_a: names[i].clone(),
                    col_b: names[j].clone(),
                    r,
                    ci_lower,
                    ci_upper,
                });
            }
        }
        results
    }

    fn build_correlation_network(
        &self,
        names: &[String],
        matrix: &Array2<f64>,
        threshold: f64,
    ) -> Vec<CorrelationEdge> {
        let p = matrix.ncols();
        let mut edges = Vec::new();

        for i in 0..p {
            for j in (i + 1)..p {
                let r = Self::pearson_vec(&matrix.column(i).to_vec(), &matrix.column(j).to_vec());
                if r.abs() >= threshold {
                    edges.push(CorrelationEdge {
                        source: names[i].clone(),
                        target: names[j].clone(),
                        weight: r,
                    });
                }
            }
        }
        edges
    }

    /// Distance correlation (Székely).
    fn compute_distance_correlations(
        &self,
        names: &[String],
        matrix: &Array2<f64>,
    ) -> Vec<(String, String, f64)> {
        let p = matrix.ncols();
        let n = matrix.nrows();
        if p < 2 || n < 10 {
            return vec![];
        }

        // Limit for computational cost
        let max_pairs = 50;
        let mut results = Vec::new();
        let mut count = 0;

        for i in 0..p {
            for j in (i + 1)..p {
                if count >= max_pairs {
                    break;
                }
                let dcor = Self::distance_correlation(
                    &matrix.column(i).to_vec(),
                    &matrix.column(j).to_vec(),
                );
                results.push((names[i].clone(), names[j].clone(), dcor));
                count += 1;
            }
        }
        results
    }

    fn distance_correlation(x: &[f64], y: &[f64]) -> f64 {
        let n = x.len().min(y.len());
        if n < 4 {
            return 0.0;
        }

        let a = Self::double_centered_distances(x);
        let b = Self::double_centered_distances(y);

        let dcov_xy: f64 = {
            let a = &a;
            let b = &b;
            (0..n)
                .flat_map(|i| (0..n).map(move |j| (i, j)))
                .map(|(i, j)| a[i * n + j] * b[i * n + j])
                .sum::<f64>()
                / (n * n) as f64
        };

        let dcov_xx: f64 = {
            let a = &a;
            (0..n)
                .flat_map(|i| (0..n).map(move |j| (i, j)))
                .map(|(i, j)| a[i * n + j] * a[i * n + j])
                .sum::<f64>()
                / (n * n) as f64
        };

        let dcov_yy: f64 = {
            let b = &b;
            (0..n)
                .flat_map(|i| (0..n).map(move |j| (i, j)))
                .map(|(i, j)| b[i * n + j] * b[i * n + j])
                .sum::<f64>()
                / (n * n) as f64
        };

        let denom = (dcov_xx * dcov_yy).sqrt();
        if denom > f64::EPSILON {
            (dcov_xy / denom).sqrt().clamp(0.0, 1.0)
        } else {
            0.0
        }
    }

    fn double_centered_distances(x: &[f64]) -> Vec<f64> {
        let n = x.len();
        let mut d = vec![0.0f64; n * n];

        // Distance matrix
        for i in 0..n {
            for j in 0..n {
                d[i * n + j] = (x[i] - x[j]).abs();
            }
        }

        // Row means, column means, grand mean
        let row_means: Vec<f64> = (0..n)
            .map(|i| (0..n).map(|j| d[i * n + j]).sum::<f64>() / n as f64)
            .collect();
        let col_means: Vec<f64> = (0..n)
            .map(|j| (0..n).map(|i| d[i * n + j]).sum::<f64>() / n as f64)
            .collect();
        let grand_mean: f64 = row_means.iter().sum::<f64>() / n as f64;

        // Double centering
        for i in 0..n {
            for j in 0..n {
                d[i * n + j] = d[i * n + j] - row_means[i] - col_means[j] + grand_mean;
            }
        }

        d
    }

    // ── Helpers ─────────────────────────────────────────────────

    fn pearson_matrix(&self, matrix: &Array2<f64>) -> Array2<f64> {
        let n = matrix.nrows();
        let p = matrix.ncols();
        let mut corr = Array2::<f64>::eye(p);

        let means: Vec<f64> = (0..p)
            .map(|j| matrix.column(j).mean().unwrap_or(0.0))
            .collect();
        let stds: Vec<f64> = (0..p)
            .map(|j| {
                let m = means[j];
                let v: f64 = matrix
                    .column(j)
                    .iter()
                    .map(|x| (x - m).powi(2))
                    .sum::<f64>()
                    / (n as f64 - 1.0).max(1.0);
                v.sqrt()
            })
            .collect();

        for i in 0..p {
            for j in (i + 1)..p {
                if stds[i] < f64::EPSILON || stds[j] < f64::EPSILON {
                    continue;
                }
                let cov: f64 = (0..n)
                    .map(|k| (matrix[[k, i]] - means[i]) * (matrix[[k, j]] - means[j]))
                    .sum::<f64>()
                    / (n as f64 - 1.0).max(1.0);
                let r = (cov / (stds[i] * stds[j])).clamp(-1.0, 1.0);
                corr[[i, j]] = r;
                corr[[j, i]] = r;
            }
        }
        corr
    }

    fn pearson_vec(x: &[f64], y: &[f64]) -> f64 {
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
        (cov / (sx * sy).sqrt()).clamp(-1.0, 1.0)
    }

    fn extract_matrix(&self, num_cols: &[&str]) -> (Vec<String>, Array2<f64>) {
        let mut names = Vec::new();
        let mut col_data = Vec::new();

        for &col_name in num_cols {
            if let Some(vals) = self
                .df
                .column(col_name)
                .ok()
                .and_then(|c| DescriptiveStats::column_to_f64_vec(c))
            {
                names.push(col_name.to_string());
                col_data.push(vals);
            }
        }

        if names.is_empty() {
            return (vec![], Array2::zeros((0, 0)));
        }

        let min_len = col_data.iter().map(|v| v.len()).min().unwrap_or(0);
        let sample_len = min_len.min(self.max_sample);
        let step = (min_len / sample_len).max(1);
        let n_cols = names.len();

        let mut matrix = Array2::<f64>::zeros((sample_len, n_cols));
        for (j, data) in col_data.iter().enumerate() {
            for (i_out, i_in) in (0..min_len).step_by(step).take(sample_len).enumerate() {
                matrix[[i_out, j]] = data[i_in];
            }
        }

        (names, matrix)
    }
}
