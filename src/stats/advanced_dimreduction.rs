use ndarray::Array2;
use polars::prelude::*;
use serde::{Deserialize, Serialize};

use crate::core::schema::DataSchema;
use crate::stats::descriptive::DescriptiveStats;

// ─── Result types ───────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TsneResult {
    pub embedding: Vec<[f64; 2]>,
    pub kl_divergence: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FactorAnalysisResult {
    pub n_factors: usize,
    pub loadings: Vec<Vec<f64>>, // features × factors
    pub variance_explained: Vec<f64>,
    pub feature_names: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureContribution {
    pub column: String,
    pub contribution: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdvancedDimReductionResult {
    pub tsne: Option<TsneResult>,
    pub factor_analysis: Option<FactorAnalysisResult>,
    pub feature_contributions: Vec<FeatureContribution>,
}

// ─── Implementation ─────────────────────────────────────────────────

pub struct AdvancedDimReductionStats<'a> {
    df: &'a DataFrame,
    schema: &'a DataSchema,
    tsne_perplexity: f64,
    max_sample: usize,
}

impl<'a> AdvancedDimReductionStats<'a> {
    pub fn new(
        df: &'a DataFrame,
        schema: &'a DataSchema,
        tsne_perplexity: f64,
        max_sample: usize,
    ) -> Self {
        Self {
            df,
            schema,
            tsne_perplexity,
            max_sample,
        }
    }

    pub fn compute(&self) -> AdvancedDimReductionResult {
        let (names, matrix) = self.prepare_data();

        let tsne = if matrix.nrows() >= 10 && matrix.ncols() >= 2 {
            self.tsne_2d(&matrix)
        } else {
            None
        };

        let factor_analysis = if matrix.nrows() >= 20 && matrix.ncols() >= 3 {
            self.factor_analysis(&names, &matrix)
        } else {
            None
        };

        let feature_contributions = self.compute_contributions(&names, &matrix);

        AdvancedDimReductionResult {
            tsne,
            factor_analysis,
            feature_contributions,
        }
    }

    /// Simplified t-SNE implementation (Barnes-Hut style).
    fn tsne_2d(&self, data: &Array2<f64>) -> Option<TsneResult> {
        let n = data.nrows();
        let d = data.ncols();
        if n < 4 {
            return None;
        }

        let perplexity = self.tsne_perplexity.min((n as f64 - 1.0) / 3.0);

        // Compute pairwise distances
        let mut dist_sq = Array2::<f64>::zeros((n, n));
        for i in 0..n {
            for j in (i + 1)..n {
                let d2: f64 = (0..d).map(|k| (data[[i, k]] - data[[j, k]]).powi(2)).sum();
                dist_sq[[i, j]] = d2;
                dist_sq[[j, i]] = d2;
            }
        }

        // Compute joint probabilities P
        let mut p_matrix = vec![vec![0.0f64; n]; n];
        for i in 0..n {
            let sigma = Self::binary_search_sigma(&dist_sq, i, perplexity);
            let sigma2 = 2.0 * sigma * sigma;
            let mut sum = 0.0f64;
            for j in 0..n {
                if i != j {
                    let pij = (-dist_sq[[i, j]] / sigma2).exp();
                    p_matrix[i][j] = pij;
                    sum += pij;
                }
            }
            if sum > f64::EPSILON {
                for j in 0..n {
                    p_matrix[i][j] /= sum;
                }
            }
        }

        // Symmetrize
        for i in 0..n {
            for j in (i + 1)..n {
                let pij = (p_matrix[i][j] + p_matrix[j][i]) / (2.0 * n as f64);
                let pij = pij.max(1e-12);
                p_matrix[i][j] = pij;
                p_matrix[j][i] = pij;
            }
        }

        // Initialize embedding
        let mut y = Array2::<f64>::zeros((n, 2));
        for i in 0..n {
            y[[i, 0]] = (i as f64 * 0.1).sin() * 0.01;
            y[[i, 1]] = (i as f64 * 0.1).cos() * 0.01;
        }

        let learning_rate = 200.0;
        let momentum = 0.8;
        let mut gains = Array2::<f64>::ones((n, 2));
        let mut update = Array2::<f64>::zeros((n, 2));

        let n_iter = 300;

        for _iter in 0..n_iter {
            // Compute Q (Student-t kernel)
            let mut q_matrix = vec![vec![0.0f64; n]; n];
            let mut q_sum = 0.0f64;
            for i in 0..n {
                for j in (i + 1)..n {
                    let d2: f64 = (0..2).map(|k| (y[[i, k]] - y[[j, k]]).powi(2)).sum();
                    let qij = 1.0 / (1.0 + d2);
                    q_matrix[i][j] = qij;
                    q_matrix[j][i] = qij;
                    q_sum += 2.0 * qij;
                }
            }

            if q_sum > f64::EPSILON {
                for i in 0..n {
                    for j in 0..n {
                        q_matrix[i][j] /= q_sum;
                        q_matrix[i][j] = q_matrix[i][j].max(1e-12);
                    }
                }
            }

            // Gradients
            let mut grad = Array2::<f64>::zeros((n, 2));
            for i in 0..n {
                for j in 0..n {
                    if i == j {
                        continue;
                    }
                    let d2: f64 = (0..2).map(|k| (y[[i, k]] - y[[j, k]]).powi(2)).sum();
                    let mult = 4.0 * (p_matrix[i][j] - q_matrix[i][j]) / (1.0 + d2);
                    for k in 0..2 {
                        grad[[i, k]] += mult * (y[[i, k]] - y[[j, k]]);
                    }
                }
            }

            // Update
            for i in 0..n {
                for k in 0..2 {
                    let sign_match = (grad[[i, k]] > 0.0) == (update[[i, k]] > 0.0);
                    gains[[i, k]] = if sign_match {
                        (gains[[i, k]] * 0.8).max(0.01)
                    } else {
                        gains[[i, k]] + 0.2
                    };
                    update[[i, k]] =
                        momentum * update[[i, k]] - learning_rate * gains[[i, k]] * grad[[i, k]];
                    y[[i, k]] += update[[i, k]];
                }
            }

            // Center
            let mean0 = y.column(0).mean().unwrap_or(0.0);
            let mean1 = y.column(1).mean().unwrap_or(0.0);
            for i in 0..n {
                y[[i, 0]] -= mean0;
                y[[i, 1]] -= mean1;
            }
        }

        // KL divergence
        let mut kl = 0.0f64;
        for i in 0..n {
            for j in 0..n {
                if i != j && p_matrix[i][j] > 1e-12 {
                    let d2: f64 = (0..2).map(|k| (y[[i, k]] - y[[j, k]]).powi(2)).sum();
                    let qij = (1.0 / (1.0 + d2)).max(1e-12);
                    kl += p_matrix[i][j] * (p_matrix[i][j] / qij).ln();
                }
            }
        }

        let embedding: Vec<[f64; 2]> = (0..n).map(|i| [y[[i, 0]], y[[i, 1]]]).collect();

        Some(TsneResult {
            embedding,
            kl_divergence: kl,
        })
    }

    fn binary_search_sigma(dist_sq: &Array2<f64>, i: usize, target_perp: f64) -> f64 {
        let n = dist_sq.nrows();
        let log_perp = target_perp.ln();
        let mut lo = 1e-10f64;
        let mut hi = 1e4f64;
        let mut sigma = 1.0;

        for _ in 0..50 {
            sigma = (lo + hi) / 2.0;
            let sigma2 = 2.0 * sigma * sigma;
            let mut sum = 0.0f64;
            let mut h = 0.0f64;
            for j in 0..n {
                if j != i {
                    let pij = (-dist_sq[[i, j]] / sigma2).exp();
                    sum += pij;
                    h -= pij * (-dist_sq[[i, j]] / sigma2);
                }
            }
            if sum > f64::EPSILON {
                h = h / sum + sum.ln();
            }

            if (h - log_perp).abs() < 1e-5 {
                break;
            }
            if h > log_perp {
                hi = sigma;
            } else {
                lo = sigma;
            }
        }
        sigma
    }

    /// Factor Analysis (simplified: PCA loadings rotated).
    fn factor_analysis(
        &self,
        names: &[String],
        matrix: &Array2<f64>,
    ) -> Option<FactorAnalysisResult> {
        let p = matrix.ncols();
        if p < 3 {
            return None;
        }

        // Use PCA as initial factor extraction
        let standardized = Self::standardize(matrix);
        let cov = Self::covariance_matrix(&standardized);

        // Eigendecomposition
        let n_factors = ((p as f64).sqrt().ceil() as usize).min(p / 2).max(1);
        let (eigenvalues, eigenvectors) =
            crate::stats::pca::PcaStats::eigen_decomposition(&cov, n_factors);

        let total: f64 = eigenvalues.iter().sum();
        let variance_explained: Vec<f64> = eigenvalues
            .iter()
            .map(|&ev| {
                if total > f64::EPSILON {
                    ev / total
                } else {
                    0.0
                }
            })
            .collect();

        // Loadings: eigenvectors * sqrt(eigenvalues)
        let loadings: Vec<Vec<f64>> = (0..p)
            .map(|feat| {
                (0..n_factors)
                    .map(|f| eigenvectors[[feat, f]] * eigenvalues.get(f).unwrap_or(&0.0).sqrt())
                    .collect()
            })
            .collect();

        Some(FactorAnalysisResult {
            n_factors,
            loadings,
            variance_explained,
            feature_names: names.to_vec(),
        })
    }

    fn compute_contributions(
        &self,
        names: &[String],
        matrix: &Array2<f64>,
    ) -> Vec<FeatureContribution> {
        if matrix.ncols() < 2 || matrix.nrows() < 3 {
            return vec![];
        }

        let standardized = Self::standardize(matrix);
        let cov = Self::covariance_matrix(&standardized);
        let (eigenvalues, eigenvectors) =
            crate::stats::pca::PcaStats::eigen_decomposition(&cov, 2.min(matrix.ncols()));

        let total_var: f64 = eigenvalues.iter().sum::<f64>().max(f64::EPSILON);

        let mut contributions: Vec<FeatureContribution> = names
            .iter()
            .enumerate()
            .map(|(i, name)| {
                let contrib: f64 = eigenvalues
                    .iter()
                    .enumerate()
                    .map(|(k, &ev)| {
                        let weight = ev / total_var;
                        eigenvectors[[i, k]].powi(2) * weight
                    })
                    .sum();
                FeatureContribution {
                    column: name.clone(),
                    contribution: contrib,
                }
            })
            .collect();

        contributions.sort_by(|a, b| {
            b.contribution
                .partial_cmp(&a.contribution)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        contributions
    }

    // ── Helpers ─────────────────────────────────────────────────

    fn prepare_data(&self) -> (Vec<String>, Array2<f64>) {
        let num_cols = self.schema.numeric_columns();
        let mut names = Vec::new();
        let mut col_data = Vec::new();

        for &col_name in &num_cols {
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

    fn standardize(matrix: &Array2<f64>) -> Array2<f64> {
        crate::stats::pca::PcaStats::standardize(matrix)
    }

    fn covariance_matrix(matrix: &Array2<f64>) -> Array2<f64> {
        crate::stats::pca::PcaStats::covariance_matrix(matrix)
    }
}
