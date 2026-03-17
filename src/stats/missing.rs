use polars::prelude::*;
use serde::{Deserialize, Serialize};

use crate::core::schema::DataSchema;

// ─── Result types ───────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MissingColumnInfo {
    pub column: String,
    pub missing_count: usize,
    pub missing_ratio: f64,
    pub dtype: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MissingResult {
    /// Per-column missing information.
    pub columns: Vec<MissingColumnInfo>,
    /// Number of rows with at least one missing value.
    pub rows_with_missing: usize,
    /// Overall missing ratio (total missing / total cells).
    pub overall_missing_ratio: f64,
    /// Distribution of missing count per row: (n_missing, count_of_rows).
    pub row_missing_distribution: Vec<(usize, usize)>,
    /// Boolean missing matrix (columns × is_missing) for pattern analysis.
    /// Stored as column_name → Vec<bool> for serialization.
    pub missing_matrix: Vec<(String, Vec<bool>)>,
}

// ─── Implementation ─────────────────────────────────────────────────

pub struct MissingStats<'a> {
    df: &'a DataFrame,
    schema: &'a DataSchema,
}

impl<'a> MissingStats<'a> {
    pub fn new(df: &'a DataFrame, schema: &'a DataSchema) -> Self {
        Self { df, schema }
    }

    pub fn compute(&self) -> MissingResult {
        let n_rows = self.df.height();
        let n_cols = self.df.width();
        let total_cells = n_rows * n_cols;

        // ── Per-column info ─────────────────────────────────────
        let columns: Vec<MissingColumnInfo> = self
            .df
            .get_columns()
            .iter()
            .map(|col| {
                let missing_count = col.null_count();
                let missing_ratio = if n_rows > 0 {
                    missing_count as f64 / n_rows as f64
                } else {
                    0.0
                };
                MissingColumnInfo {
                    column: col.name().to_string(),
                    missing_count,
                    missing_ratio,
                    dtype: format!("{:?}", col.dtype()),
                }
            })
            .collect();

        let total_missing: usize = columns.iter().map(|c| c.missing_count).sum();
        let overall_missing_ratio = if total_cells > 0 {
            total_missing as f64 / total_cells as f64
        } else {
            0.0
        };

        // ── Per-row missing count ───────────────────────────────
        let mut row_missing_counts = vec![0usize; n_rows];
        for col in self.df.get_columns() {
            let is_null = col.is_null();
            for (i, val) in is_null.into_iter().enumerate() {
                if val.unwrap_or(false) {
                    row_missing_counts[i] += 1;
                }
            }
        }

        let rows_with_missing = row_missing_counts.iter().filter(|&&c| c > 0).count();

        // Distribution: how many rows have 0 missing, 1 missing, etc.
        let max_missing = *row_missing_counts.iter().max().unwrap_or(&0);
        let mut distribution = vec![0usize; max_missing + 1];
        for &c in &row_missing_counts {
            distribution[c] += 1;
        }
        let row_missing_distribution: Vec<(usize, usize)> = distribution
            .into_iter()
            .enumerate()
            .filter(|(_, count)| *count > 0)
            .collect();

        // ── Missing matrix ──────────────────────────────────────
        // Only generate for columns that have any missing (to save memory)
        let missing_matrix: Vec<(String, Vec<bool>)> = self
            .df
            .get_columns()
            .iter()
            .filter(|col| col.null_count() > 0)
            .map(|col| {
                let mask: Vec<bool> = col
                    .is_null()
                    .into_iter()
                    .map(|v| v.unwrap_or(false))
                    .collect();
                (col.name().to_string(), mask)
            })
            .collect();

        MissingResult {
            columns,
            rows_with_missing,
            overall_missing_ratio,
            row_missing_distribution,
            missing_matrix,
        }
    }
}
