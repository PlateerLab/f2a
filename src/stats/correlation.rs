use indexmap::IndexMap;
use ndarray::{Array1, Array2};
use polars::prelude::*;
use serde::{Deserialize, Serialize};

use crate::core::schema::DataSchema;
use crate::stats::descriptive::DescriptiveStats;

// ─── Result types ───────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorrelationResult {
    /// Column names (axis labels).
    pub columns: Vec<String>,
    /// Pearson correlation matrix (row-major).
    pub pearson: Vec<Vec<f64>>,
    /// Spearman correlation matrix (row-major).
    pub spearman: Vec<Vec<f64>>,
    /// Cramér's V matrix for categorical columns.
    pub cramers_v: Option<CramersVResult>,
    /// Variance Inflation Factors per numeric column.
    pub vif: Vec<VifEntry>,
    /// Pairs with |r| ≥ threshold.
    pub high_correlation_pairs: Vec<HighCorrPair>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CramersVResult {
    pub columns: Vec<String>,
    pub matrix: Vec<Vec<f64>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VifEntry {
    pub column: String,
    pub vif: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HighCorrPair {
    pub col_a: String,
    pub col_b: String,
    pub pearson: f64,
    pub spearman: f64,
}

// ─── Implementation ─────────────────────────────────────────────────

pub struct CorrelationStats<'a> {
    df: &'a DataFrame,
    schema: &'a DataSchema,
    threshold: f64,
}

impl<'a> CorrelationStats<'a> {
    pub fn new(df: &'a DataFrame, schema: &'a DataSchema, threshold: f64) -> Self {
        Self {
            df,
            schema,
            threshold,
        }
    }

    pub fn compute(&self) -> CorrelationResult {
        let num_cols = self.schema.numeric_columns();
        let (columns, matrix) = self.extract_numeric_matrix(&num_cols);

        let pearson = if matrix.ncols() >= 2 {
            Self::correlation_matrix(&matrix)
        } else {
            vec![vec![1.0]; columns.len().max(1)]
        };

        let spearman = if matrix.ncols() >= 2 {
            let ranked = Self::rank_matrix(&matrix);
            Self::correlation_matrix(&ranked)
        } else {
            vec![vec![1.0]; columns.len().max(1)]
        };

        let high_correlation_pairs =
            Self::find_high_correlations(&columns, &pearson, &spearman, self.threshold);

        let vif = self.compute_vif(&columns, &matrix);

        let cramers_v = self.compute_cramers_v();

        CorrelationResult {
            columns,
            pearson,
            spearman,
            cramers_v,
            vif,
            high_correlation_pairs,
        }
    }

    // ── Pearson correlation matrix ──────────────────────────────

    fn correlation_matrix(matrix: &Array2<f64>) -> Vec<Vec<f64>> {
        let n_rows = matrix.nrows();
        let n_cols = matrix.ncols();
        if n_rows == 0 || n_cols == 0 {
            return vec![];
        }

        // Column means
        let means: Vec<f64> = (0..n_cols)
            .map(|j| matrix.column(j).mean().unwrap_or(0.0))
            .collect();

        // Column standard deviations
        let stds: Vec<f64> = (0..n_cols)
            .map(|j| {
                let col = matrix.column(j);
                let mean = means[j];
                let var = col.iter().map(|x| (x - mean).powi(2)).sum::<f64>()
                    / (n_rows as f64 - 1.0).max(1.0);
                var.sqrt()
            })
            .collect();

        // Correlation
        let mut result = vec![vec![0.0f64; n_cols]; n_cols];
        for i in 0..n_cols {
            result[i][i] = 1.0;
            for j in (i + 1)..n_cols {
                if stds[i] < f64::EPSILON || stds[j] < f64::EPSILON {
                    result[i][j] = 0.0;
                    result[j][i] = 0.0;
                    continue;
                }
                let cov: f64 = (0..n_rows)
                    .map(|k| (matrix[[k, i]] - means[i]) * (matrix[[k, j]] - means[j]))
                    .sum::<f64>()
                    / (n_rows as f64 - 1.0).max(1.0);
                let r = cov / (stds[i] * stds[j]);
                let r = r.clamp(-1.0, 1.0);
                result[i][j] = r;
                result[j][i] = r;
            }
        }
        result
    }

    // ── Rank transform for Spearman ─────────────────────────────

    fn rank_matrix(matrix: &Array2<f64>) -> Array2<f64> {
        let n_rows = matrix.nrows();
        let n_cols = matrix.ncols();
        let mut ranked = Array2::<f64>::zeros((n_rows, n_cols));

        for j in 0..n_cols {
            let col = matrix.column(j);
            let mut indexed: Vec<(usize, f64)> = col.iter().copied().enumerate().collect();
            indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

            // Average ranks for ties
            let mut i = 0;
            while i < indexed.len() {
                let mut end = i + 1;
                while end < indexed.len() && (indexed[end].1 - indexed[i].1).abs() < f64::EPSILON {
                    end += 1;
                }
                let avg_rank = (i..end).map(|k| k + 1).sum::<usize>() as f64 / (end - i) as f64;
                for k in i..end {
                    ranked[[indexed[k].0, j]] = avg_rank;
                }
                i = end;
            }
        }
        ranked
    }

    // ── VIF (Variance Inflation Factor) ─────────────────────────

    fn compute_vif(&self, columns: &[String], matrix: &Array2<f64>) -> Vec<VifEntry> {
        let n_cols = matrix.ncols();
        if n_cols < 2 {
            return vec![];
        }

        columns
            .iter()
            .enumerate()
            .map(|(target_idx, col_name)| {
                // Regress column target_idx on all other columns
                // VIF = 1 / (1 - R²)
                let r_squared = Self::r_squared_from_others(matrix, target_idx);
                let vif = if (1.0 - r_squared).abs() < f64::EPSILON {
                    f64::INFINITY
                } else {
                    1.0 / (1.0 - r_squared)
                };
                VifEntry {
                    column: col_name.clone(),
                    vif,
                }
            })
            .collect()
    }

    /// Compute R² of column `target` regressed on all other columns via OLS.
    fn r_squared_from_others(matrix: &Array2<f64>, target: usize) -> f64 {
        let n = matrix.nrows();
        let p = matrix.ncols();
        if p < 2 || n <= p {
            return 0.0;
        }

        let y = matrix.column(target).to_owned();
        let y_mean = y.mean().unwrap_or(0.0);

        // Build X matrix with intercept (all other columns + 1s column)
        let other_cols: Vec<usize> = (0..p).filter(|&i| i != target).collect();
        let x_cols = other_cols.len() + 1; // +1 for intercept

        let mut x = Array2::<f64>::ones((n, x_cols));
        for (new_j, &old_j) in other_cols.iter().enumerate() {
            for i in 0..n {
                x[[i, new_j + 1]] = matrix[[i, old_j]];
            }
        }

        // OLS: beta = (X'X)^(-1) X'y
        let xt = x.t();
        let xtx = xt.dot(&x);
        let xty = xt.dot(&y);

        // Simple matrix inversion for small matrices using Gauss-Jordan
        let beta = match Self::solve_linear(&xtx, &xty) {
            Some(b) => b,
            None => return 0.0,
        };

        let y_hat = x.dot(&beta);
        let ss_res: f64 = y
            .iter()
            .zip(y_hat.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum();
        let ss_tot: f64 = y.iter().map(|a| (a - y_mean).powi(2)).sum();

        if ss_tot < f64::EPSILON {
            0.0
        } else {
            1.0 - ss_res / ss_tot
        }
    }

    /// Solve Ax = b via LU-style Gaussian elimination.
    fn solve_linear(a: &Array2<f64>, b: &Array1<f64>) -> Option<Array1<f64>> {
        let n = a.nrows();
        if n != a.ncols() || n != b.len() {
            return None;
        }

        // Augmented matrix [A|b]
        let mut aug = Array2::<f64>::zeros((n, n + 1));
        for i in 0..n {
            for j in 0..n {
                aug[[i, j]] = a[[i, j]];
            }
            aug[[i, n]] = b[i];
        }

        // Forward elimination with partial pivoting
        for col in 0..n {
            // Find pivot
            let mut max_row = col;
            let mut max_val = aug[[col, col]].abs();
            for row in (col + 1)..n {
                if aug[[row, col]].abs() > max_val {
                    max_val = aug[[row, col]].abs();
                    max_row = row;
                }
            }
            if max_val < 1e-12 {
                return None; // Singular
            }

            // Swap rows
            if max_row != col {
                for j in 0..=n {
                    let tmp = aug[[col, j]];
                    aug[[col, j]] = aug[[max_row, j]];
                    aug[[max_row, j]] = tmp;
                }
            }

            // Eliminate below
            let pivot = aug[[col, col]];
            for row in (col + 1)..n {
                let factor = aug[[row, col]] / pivot;
                for j in col..=n {
                    aug[[row, j]] -= factor * aug[[col, j]];
                }
            }
        }

        // Back substitution
        let mut x = Array1::<f64>::zeros(n);
        for i in (0..n).rev() {
            let mut sum = aug[[i, n]];
            for j in (i + 1)..n {
                sum -= aug[[i, j]] * x[j];
            }
            x[i] = sum / aug[[i, i]];
        }

        Some(x)
    }

    // ── Cramér's V ──────────────────────────────────────────────

    fn compute_cramers_v(&self) -> Option<CramersVResult> {
        let cat_cols = self.schema.categorical_columns();
        if cat_cols.len() < 2 {
            return None;
        }

        let n = self.df.height() as f64;
        let n_cats = cat_cols.len();
        let mut matrix = vec![vec![0.0f64; n_cats]; n_cats];
        let columns: Vec<String> = cat_cols.iter().map(|s| s.to_string()).collect();

        for i in 0..n_cats {
            matrix[i][i] = 1.0;
            for j in (i + 1)..n_cats {
                let v = self.cramers_v_pair(cat_cols[i], cat_cols[j], n);
                matrix[i][j] = v;
                matrix[j][i] = v;
            }
        }

        Some(CramersVResult { columns, matrix })
    }

    /// Compute Cramér's V between two categorical columns.
    fn cramers_v_pair(&self, col_a: &str, col_b: &str, n: f64) -> f64 {
        let a = match self.df.column(col_a) {
            Ok(c) => c.cast(&DataType::String).unwrap_or_default(),
            Err(_) => return 0.0,
        };
        let b = match self.df.column(col_b) {
            Ok(c) => c.cast(&DataType::String).unwrap_or_default(),
            Err(_) => return 0.0,
        };

        let a_str = a.str().unwrap();
        let b_str = b.str().unwrap();

        // Build contingency table
        let mut contingency: IndexMap<(String, String), usize> = IndexMap::new();
        let mut a_counts: IndexMap<String, usize> = IndexMap::new();
        let mut b_counts: IndexMap<String, usize> = IndexMap::new();

        for (va, vb) in a_str.into_iter().zip(b_str.into_iter()) {
            let ka = va.unwrap_or("(NA)").to_string();
            let kb = vb.unwrap_or("(NA)").to_string();
            *contingency.entry((ka.clone(), kb.clone())).or_insert(0) += 1;
            *a_counts.entry(ka).or_insert(0) += 1;
            *b_counts.entry(kb).or_insert(0) += 1;
        }

        // Chi-square statistic
        let mut chi2 = 0.0f64;
        for ((ka, kb), &observed) in &contingency {
            let ea = *a_counts.get(ka).unwrap_or(&0) as f64;
            let eb = *b_counts.get(kb).unwrap_or(&0) as f64;
            let expected = ea * eb / n;
            if expected > 0.0 {
                chi2 += (observed as f64 - expected).powi(2) / expected;
            }
        }

        let r = a_counts.len() as f64;
        let k = b_counts.len() as f64;
        let min_dim = (r - 1.0).min(k - 1.0);

        if min_dim < 1.0 || n < 1.0 {
            return 0.0;
        }

        (chi2 / (n * min_dim)).sqrt().clamp(0.0, 1.0)
    }

    // ── High-correlation pair detection ─────────────────────────

    fn find_high_correlations(
        columns: &[String],
        pearson: &[Vec<f64>],
        spearman: &[Vec<f64>],
        threshold: f64,
    ) -> Vec<HighCorrPair> {
        let mut pairs = Vec::new();
        let n = columns.len();
        for i in 0..n {
            for j in (i + 1)..n {
                let p = pearson[i][j];
                let s = spearman[i][j];
                if p.abs() >= threshold || s.abs() >= threshold {
                    pairs.push(HighCorrPair {
                        col_a: columns[i].clone(),
                        col_b: columns[j].clone(),
                        pearson: p,
                        spearman: s,
                    });
                }
            }
        }
        pairs.sort_by(|a, b| {
            b.pearson
                .abs()
                .partial_cmp(&a.pearson.abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        pairs
    }

    // ── Utility: extract numeric columns into ndarray matrix ────

    fn extract_numeric_matrix(&self, num_cols: &[&str]) -> (Vec<String>, Array2<f64>) {
        let mut valid_cols: Vec<String> = Vec::new();
        let mut col_data: Vec<Vec<f64>> = Vec::new();

        for &col_name in num_cols {
            if let Some(vals) = self
                .df
                .column(col_name)
                .ok()
                .and_then(|c| DescriptiveStats::column_to_f64_vec(c))
            {
                valid_cols.push(col_name.to_string());
                col_data.push(vals);
            }
        }

        if valid_cols.is_empty() {
            return (vec![], Array2::zeros((0, 0)));
        }

        // Use the minimum row count across all columns (after dropping nulls)
        let min_len = col_data.iter().map(|v| v.len()).min().unwrap_or(0);
        let n_cols = valid_cols.len();

        let mut matrix = Array2::<f64>::zeros((min_len, n_cols));
        for (j, data) in col_data.iter().enumerate() {
            for i in 0..min_len {
                matrix[[i, j]] = data[i];
            }
        }

        (valid_cols, matrix)
    }
}
