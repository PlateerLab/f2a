use polars::prelude::*;
use serde::{Deserialize, Serialize};

use crate::utils::types::{infer_column_type, ColumnType};

// ─── Per-column metadata ────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ColumnInfo {
    pub name: String,
    pub dtype: String,
    pub inferred_type: ColumnType,
    pub n_unique: usize,
    pub n_missing: usize,
    pub missing_ratio: f64,
}

// ─── Dataset-level schema ───────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataSchema {
    pub n_rows: usize,
    pub n_cols: usize,
    pub columns: Vec<ColumnInfo>,
    pub memory_usage_bytes: usize,
}

impl DataSchema {
    /// Build a `DataSchema` from a Polars `DataFrame`.
    pub fn from_dataframe(df: &DataFrame) -> Self {
        let n_rows = df.height();
        let n_cols = df.width();

        let columns: Vec<ColumnInfo> = df
            .get_columns()
            .iter()
            .map(|col| {
                let name = col.name().to_string();
                let dtype = format!("{:?}", col.dtype());
                let n_missing = col.null_count();
                let missing_ratio = if n_rows > 0 {
                    n_missing as f64 / n_rows as f64
                } else {
                    0.0
                };

                let n_unique = col.n_unique().unwrap_or(0);

                // Compute avg string length for text classification heuristic
                let avg_str_len = if col.dtype() == &DataType::String {
                    let lengths = col.str().ok().map(|ca| {
                        let total: usize =
                            ca.into_iter().filter_map(|opt| opt.map(|s| s.len())).sum();
                        let count = ca.into_iter().filter(|o| o.is_some()).count();
                        if count > 0 {
                            total as f64 / count as f64
                        } else {
                            0.0
                        }
                    });
                    lengths
                } else {
                    None
                };

                let inferred_type = infer_column_type(col.dtype(), n_unique, n_rows, avg_str_len);

                ColumnInfo {
                    name,
                    dtype,
                    inferred_type,
                    n_unique,
                    n_missing,
                    missing_ratio,
                }
            })
            .collect();

        // Rough memory estimation
        let memory_usage_bytes = df.estimated_size();

        DataSchema {
            n_rows,
            n_cols,
            columns,
            memory_usage_bytes,
        }
    }

    // ── Convenience accessors ───────────────────────────────────

    pub fn numeric_columns(&self) -> Vec<&str> {
        self.columns
            .iter()
            .filter(|c| c.inferred_type == ColumnType::Numeric)
            .map(|c| c.name.as_str())
            .collect()
    }

    pub fn categorical_columns(&self) -> Vec<&str> {
        self.columns
            .iter()
            .filter(|c| c.inferred_type == ColumnType::Categorical)
            .map(|c| c.name.as_str())
            .collect()
    }

    pub fn text_columns(&self) -> Vec<&str> {
        self.columns
            .iter()
            .filter(|c| c.inferred_type == ColumnType::Text)
            .map(|c| c.name.as_str())
            .collect()
    }

    pub fn datetime_columns(&self) -> Vec<&str> {
        self.columns
            .iter()
            .filter(|c| c.inferred_type == ColumnType::DateTime)
            .map(|c| c.name.as_str())
            .collect()
    }

    pub fn boolean_columns(&self) -> Vec<&str> {
        self.columns
            .iter()
            .filter(|c| c.inferred_type == ColumnType::Boolean)
            .map(|c| c.name.as_str())
            .collect()
    }

    pub fn column_info(&self, name: &str) -> Option<&ColumnInfo> {
        self.columns.iter().find(|c| c.name == name)
    }
}
