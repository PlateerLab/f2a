use ndarray::{Array1, Array2};
use polars::prelude::*;
use serde::{Deserialize, Serialize};

use crate::core::schema::DataSchema;
use crate::stats::descriptive::DescriptiveStats;

// ─── Result types ───────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PcaResult {
    /// Column names used for PCA.
    pub feature_names: Vec<String>,
    /// Variance explained ratio per component.
    pub variance_ratio: Vec<f64>,
    /// Cumulative variance ratio.
    pub cumulative_ratio: Vec<f64>,
    /// Eigenvalues.
    pub eigenvalues: Vec<f64>,
    /// Loadings matrix: features × components.
    pub loadings: Vec<Vec<f64>>,
    /// Number of components needed for 90% variance.
    pub components_for_90pct: usize,
    /// Total variance explained by all computed components.
    pub total_variance_explained: f64,
}

// ─── Implementation ─────────────────────────────────────────────────

pub struct PcaStats<'a> {
    df: &'a DataFrame,
    schema: &'a DataSchema,
    max_components: usize,
}

impl<'a> PcaStats<'a> {
    pub fn new(df: &'a DataFrame, schema: &'a DataSchema, max_components: usize) -> Self {
        Self {
            df,
            schema,
            max_components,
        }
    }

    pub fn compute(&self) -> Option<PcaResult> {
        let num_cols = self.schema.numeric_columns();
        if num_cols.len() < 2 {
            return None;
        }

        // Extract and standardize numeric matrix
        let (feature_names, raw_matrix) = self.extract_matrix(&num_cols);
        let n = raw_matrix.nrows();
        let p = raw_matrix.ncols();
        if n < 2 || p < 2 {
            return None;
        }

        let standardized = Self::standardize(&raw_matrix);

        // Covariance matrix (on standardized data = correlation matrix)
        let cov = Self::covariance_matrix(&standardized);

        // Eigendecomposition via power iteration (for top-k eigenvalues)
        let n_components = self.max_components.min(p).min(n - 1);
        let (eigenvalues, eigenvectors) = Self::eigen_decomposition(&cov, n_components);

        let total_eigenvalue_sum: f64 = {
            // Trace of covariance = sum of all eigenvalues
            (0..p).map(|i| cov[[i, i]]).sum()
        };

        let variance_ratio: Vec<f64> = eigenvalues
            .iter()
            .map(|&ev| {
                if total_eigenvalue_sum > f64::EPSILON {
                    ev / total_eigenvalue_sum
                } else {
                    0.0
                }
            })
            .collect();

        let mut cumulative_ratio = Vec::with_capacity(variance_ratio.len());
        let mut cum = 0.0;
        for &vr in &variance_ratio {
            cum += vr;
            cumulative_ratio.push(cum);
        }

        let components_for_90pct = cumulative_ratio
            .iter()
            .position(|&c| c >= 0.9)
            .map(|i| i + 1)
            .unwrap_or(n_components);

        let total_variance_explained = cum;

        // Loadings: eigenvectors as rows of features
        let loadings: Vec<Vec<f64>> = (0..p)
            .map(|feat| {
                (0..eigenvectors.ncols())
                    .map(|comp| eigenvectors[[feat, comp]])
                    .collect()
            })
            .collect();

        Some(PcaResult {
            feature_names,
            variance_ratio,
            cumulative_ratio,
            eigenvalues,
            loadings,
            components_for_90pct,
            total_variance_explained,
        })
    }

    // ── Helpers ─────────────────────────────────────────────────

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
        let n_cols = names.len();
        let mut matrix = Array2::<f64>::zeros((min_len, n_cols));
        for (j, data) in col_data.iter().enumerate() {
            for i in 0..min_len {
                matrix[[i, j]] = data[i];
            }
        }

        (names, matrix)
    }

    pub(crate) fn standardize(matrix: &Array2<f64>) -> Array2<f64> {
        let n = matrix.nrows();
        let p = matrix.ncols();
        let mut result = Array2::<f64>::zeros((n, p));

        for j in 0..p {
            let col = matrix.column(j);
            let mean = col.mean().unwrap_or(0.0);
            let std = {
                let var: f64 =
                    col.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n as f64 - 1.0).max(1.0);
                var.sqrt()
            };
            for i in 0..n {
                result[[i, j]] = if std > f64::EPSILON {
                    (matrix[[i, j]] - mean) / std
                } else {
                    0.0
                };
            }
        }
        result
    }

    pub(crate) fn covariance_matrix(matrix: &Array2<f64>) -> Array2<f64> {
        let n = matrix.nrows();
        let p = matrix.ncols();

        let means: Vec<f64> = (0..p)
            .map(|j| matrix.column(j).mean().unwrap_or(0.0))
            .collect();

        let mut cov = Array2::<f64>::zeros((p, p));
        for i in 0..p {
            for j in i..p {
                let val: f64 = (0..n)
                    .map(|k| (matrix[[k, i]] - means[i]) * (matrix[[k, j]] - means[j]))
                    .sum::<f64>()
                    / (n as f64 - 1.0).max(1.0);
                cov[[i, j]] = val;
                cov[[j, i]] = val;
            }
        }
        cov
    }

    /// Simple eigendecomposition via power iteration with deflation.
    pub(crate) fn eigen_decomposition(
        cov: &Array2<f64>,
        n_components: usize,
    ) -> (Vec<f64>, Array2<f64>) {
        let p = cov.nrows();
        let mut eigenvalues = Vec::with_capacity(n_components);
        let mut eigenvectors = Array2::<f64>::zeros((p, n_components));
        let mut matrix = cov.clone();

        for k in 0..n_components {
            let (eigenvalue, eigenvector) = Self::power_iteration(&matrix, 200);
            eigenvalues.push(eigenvalue);
            for i in 0..p {
                eigenvectors[[i, k]] = eigenvector[i];
            }

            // Deflate: A = A - λ * v * v^T
            for i in 0..p {
                for j in 0..p {
                    matrix[[i, j]] -= eigenvalue * eigenvector[i] * eigenvector[j];
                }
            }
        }

        (eigenvalues, eigenvectors)
    }

    /// Power iteration to find the dominant eigenvalue/eigenvector.
    fn power_iteration(matrix: &Array2<f64>, max_iter: usize) -> (f64, Array1<f64>) {
        let p = matrix.nrows();
        let mut v = Array1::<f64>::from_elem(p, 1.0 / (p as f64).sqrt());

        let mut eigenvalue = 0.0;

        for _ in 0..max_iter {
            let new_v = matrix.dot(&v);
            eigenvalue = new_v.dot(&v);
            let norm: f64 = new_v.iter().map(|x| x.powi(2)).sum::<f64>().sqrt();
            if norm < f64::EPSILON {
                break;
            }
            v = new_v / norm;
        }

        (eigenvalue.max(0.0), v)
    }
}
