use polars::prelude::*;
use serde::{Deserialize, Serialize};

use crate::core::schema::DataSchema;

// ─── Result types ───────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DuplicateResult {
    pub n_exact_duplicates: usize,
    pub duplicate_ratio: f64,
    /// Per-column uniqueness info.
    pub column_uniqueness: Vec<ColumnUniqueness>,
    /// Columns that could serve as unique keys (100% unique, no nulls).
    pub unique_key_candidates: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ColumnUniqueness {
    pub column: String,
    pub n_unique: usize,
    pub uniqueness_ratio: f64,
    pub n_duplicated_values: usize,
}

// ─── Implementation ─────────────────────────────────────────────────

pub struct DuplicateStats<'a> {
    df: &'a DataFrame,
    #[allow(dead_code)]
    schema: &'a DataSchema,
}

impl<'a> DuplicateStats<'a> {
    pub fn new(df: &'a DataFrame, schema: &'a DataSchema) -> Self {
        Self { df, schema }
    }

    pub fn compute(&self) -> DuplicateResult {
        let n_rows = self.df.height();

        // ── Exact row duplicates ────────────────────────────────
        let n_exact_duplicates = self
            .df
            .is_duplicated()
            .map(|mask| mask.sum().unwrap_or(0) as usize)
            .unwrap_or(0);

        let duplicate_ratio = if n_rows > 0 {
            n_exact_duplicates as f64 / n_rows as f64
        } else {
            0.0
        };

        // ── Per-column uniqueness ───────────────────────────────
        let column_uniqueness: Vec<ColumnUniqueness> = self
            .df
            .get_columns()
            .iter()
            .map(|col| {
                let n_unique = col.n_unique().unwrap_or(0);
                let uniqueness_ratio = if n_rows > 0 {
                    n_unique as f64 / n_rows as f64
                } else {
                    0.0
                };
                // Count of values that appear more than once
                let n_duplicated_values = if n_rows > n_unique {
                    n_rows - n_unique
                } else {
                    0
                };

                ColumnUniqueness {
                    column: col.name().to_string(),
                    n_unique,
                    uniqueness_ratio,
                    n_duplicated_values,
                }
            })
            .collect();

        // ── Unique key candidates ───────────────────────────────
        let unique_key_candidates: Vec<String> = column_uniqueness
            .iter()
            .filter(|cu| {
                cu.uniqueness_ratio >= 1.0 - f64::EPSILON
                    && self
                        .df
                        .column(&cu.column)
                        .map(|c| c.null_count() == 0)
                        .unwrap_or(false)
            })
            .map(|cu| cu.column.clone())
            .collect();

        DuplicateResult {
            n_exact_duplicates,
            duplicate_ratio,
            column_uniqueness,
            unique_key_candidates,
        }
    }
}
