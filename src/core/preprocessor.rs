use polars::prelude::*;
use serde::{Deserialize, Serialize};

use crate::core::schema::DataSchema;

/// Detected preprocessing issues.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PreprocessingResult {
    pub constant_columns: Vec<String>,
    pub duplicate_row_count: usize,
    pub duplicate_row_ratio: f64,
    pub high_missing_columns: Vec<(String, f64)>, // (col, ratio)
    pub id_like_columns: Vec<String>,
    pub mixed_type_columns: Vec<String>,
    pub infinite_value_columns: Vec<(String, usize)>, // (col, count)
    pub rows_before: usize,
    pub rows_after: usize,
    pub cols_before: usize,
    pub cols_after: usize,
}

/// Preprocessor – detects data quality issues and optionally cleans the DataFrame.
pub struct Preprocessor;

impl Preprocessor {
    /// Analyse the DataFrame for issues; return (cleaned_df, issues).
    ///
    /// Cleaning is **non-destructive**: the original DataFrame is not mutated.
    /// The cleaned DataFrame has:
    /// - Constant columns removed
    /// - Duplicate rows removed
    /// - Columns with ≥ `missing_threshold` missing ratio removed
    pub fn process(
        df: &DataFrame,
        schema: &DataSchema,
        missing_threshold: f64,
    ) -> (DataFrame, PreprocessingResult) {
        let mut result = PreprocessingResult {
            rows_before: df.height(),
            cols_before: df.width(),
            ..Default::default()
        };

        // ── 1. Detect constant columns ──────────────────────────
        for col in df.get_columns() {
            let n_unique = col.n_unique().unwrap_or(0);
            // A column is constant if it has 0 or 1 unique non-null values
            if n_unique <= 1 {
                result.constant_columns.push(col.name().to_string());
            }
        }

        // ── 2. Detect duplicate rows ────────────────────────────
        let dup_mask = df
            .is_duplicated()
            .unwrap_or_else(|_| BooleanChunked::new("dup".into(), vec![false; df.height()]));
        let dup_count = dup_mask.sum().unwrap_or(0) as usize;
        result.duplicate_row_count = dup_count;
        result.duplicate_row_ratio = if df.height() > 0 {
            dup_count as f64 / df.height() as f64
        } else {
            0.0
        };

        // ── 3. High-missing columns ─────────────────────────────
        for info in &schema.columns {
            if info.missing_ratio >= missing_threshold {
                result
                    .high_missing_columns
                    .push((info.name.clone(), info.missing_ratio));
            }
        }

        // ── 4. ID-like columns (unique ratio ≥ 0.95, string/int) ─
        for col in df.get_columns() {
            let n_unique = col.n_unique().unwrap_or(0);
            if df.height() > 20 {
                let unique_ratio = n_unique as f64 / df.height() as f64;
                if unique_ratio >= 0.95 {
                    let name_lower = col.name().to_lowercase();
                    let is_id_name = name_lower.contains("id")
                        || name_lower.ends_with("_id")
                        || name_lower == "index"
                        || name_lower == "key";
                    if is_id_name || unique_ratio >= 0.99 {
                        result.id_like_columns.push(col.name().to_string());
                    }
                }
            }
        }

        // ── 5. Infinite value columns (numeric only) ────────────
        for col in df.get_columns() {
            if col.dtype().is_float() {
                if let Ok(float_col) = col.f64() {
                    let inf_count = float_col
                        .into_iter()
                        .filter(|v| v.map_or(false, |x| x.is_infinite()))
                        .count();
                    if inf_count > 0 {
                        result
                            .infinite_value_columns
                            .push((col.name().to_string(), inf_count));
                    }
                }
            }
        }

        // ── Build cleaned DataFrame ─────────────────────────────
        let mut cleaned = df.clone();

        // Drop constant columns
        for col_name in &result.constant_columns {
            let _ = cleaned.drop_in_place(col_name.as_str().into());
        }

        // Drop high-missing columns
        for (col_name, _) in &result.high_missing_columns {
            let _ = cleaned.drop_in_place(col_name.as_str().into());
        }

        // Remove duplicate rows
        if dup_count > 0 {
            if let Ok(deduped) =
                cleaned.unique::<&str, PlSmallStr>(None, UniqueKeepStrategy::First, None)
            {
                cleaned = deduped;
            }
        }

        result.rows_after = cleaned.height();
        result.cols_after = cleaned.width();

        (cleaned, result)
    }
}
