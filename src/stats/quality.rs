use polars::prelude::*;
use serde::{Deserialize, Serialize};

use crate::core::schema::DataSchema;
use crate::utils::types::ColumnType;

// ─── Result types ───────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityDimension {
    pub name: String,
    pub score: f64, // 0.0 – 1.0
    pub details: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ColumnQuality {
    pub column: String,
    pub completeness: f64,
    pub uniqueness: f64,
    pub validity: f64,
    pub overall: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityResult {
    pub overall_score: f64,
    pub dimensions: Vec<QualityDimension>,
    pub by_column: Vec<ColumnQuality>,
}

// ─── Implementation ─────────────────────────────────────────────────

pub struct QualityStats<'a> {
    df: &'a DataFrame,
    schema: &'a DataSchema,
}

impl<'a> QualityStats<'a> {
    pub fn new(df: &'a DataFrame, schema: &'a DataSchema) -> Self {
        Self { df, schema }
    }

    pub fn compute(&self) -> QualityResult {
        let dimensions = vec![
            self.completeness(),
            self.uniqueness(),
            self.consistency(),
            self.validity(),
            self.timeliness(),
            self.conformity(),
        ];

        let by_column = self.per_column_quality();

        // Weighted overall score
        let weights = [0.25, 0.15, 0.15, 0.20, 0.10, 0.15];
        let overall_score: f64 = dimensions
            .iter()
            .zip(weights.iter())
            .map(|(d, w)| d.score * w)
            .sum();

        QualityResult {
            overall_score,
            dimensions,
            by_column,
        }
    }

    // ── Completeness (1 - missing ratio) ────────────────────────

    fn completeness(&self) -> QualityDimension {
        let n_rows = self.df.height();
        let n_cols = self.df.width();
        let total = n_rows * n_cols;
        let missing: usize = self.df.get_columns().iter().map(|c| c.null_count()).sum();

        let score = if total > 0 {
            1.0 - (missing as f64 / total as f64)
        } else {
            1.0
        };

        QualityDimension {
            name: "completeness".into(),
            score,
            details: format!(
                "{} missing values out of {} cells ({:.1}% complete)",
                missing,
                total,
                score * 100.0
            ),
        }
    }

    // ── Uniqueness (ratio of distinct rows) ─────────────────────

    fn uniqueness(&self) -> QualityDimension {
        let n_rows = self.df.height();
        let n_unique = self
            .df
            .unique::<&str, PlSmallStr>(None, UniqueKeepStrategy::First, None)
            .map(|u| u.height())
            .unwrap_or(n_rows);

        let score = if n_rows > 0 {
            n_unique as f64 / n_rows as f64
        } else {
            1.0
        };

        QualityDimension {
            name: "uniqueness".into(),
            score,
            details: format!(
                "{} unique rows out of {} ({:.1}% unique)",
                n_unique,
                n_rows,
                score * 100.0
            ),
        }
    }

    // ── Consistency (type uniformity within columns) ────────────

    fn consistency(&self) -> QualityDimension {
        let mut consistent_cols = 0;
        let total_cols = self.df.width();

        for col in self.df.get_columns() {
            // A column is "consistent" if its physical dtype is clean
            // (i.e. not Object / mixed-type)
            match col.dtype() {
                DataType::Unknown(_) | DataType::Null => {}
                _ => consistent_cols += 1,
            }
        }

        let score = if total_cols > 0 {
            consistent_cols as f64 / total_cols as f64
        } else {
            1.0
        };

        QualityDimension {
            name: "consistency".into(),
            score,
            details: format!(
                "{} of {} columns have consistent types ({:.1}%)",
                consistent_cols,
                total_cols,
                score * 100.0
            ),
        }
    }

    // ── Validity (values within ±4σ for numeric) ────────────────

    fn validity(&self) -> QualityDimension {
        let num_cols = self.schema.numeric_columns();
        if num_cols.is_empty() {
            return QualityDimension {
                name: "validity".into(),
                score: 1.0,
                details: "No numeric columns to validate".into(),
            };
        }

        let mut total_values = 0usize;
        let mut valid_values = 0usize;

        for &col_name in &num_cols {
            if let Ok(col) = self.df.column(col_name) {
                if let Ok(casted) = col.cast(&DataType::Float64) {
                    if let Ok(ca) = casted.f64() {
                        let vals: Vec<f64> = ca.into_iter().filter_map(|v| v).collect();
                        let n = vals.len();
                        if n == 0 {
                            continue;
                        }
                        let mean = vals.iter().sum::<f64>() / n as f64;
                        let std = (vals.iter().map(|x| (x - mean).powi(2)).sum::<f64>()
                            / (n as f64 - 1.0).max(1.0))
                        .sqrt();

                        total_values += n;
                        if std > f64::EPSILON {
                            valid_values += vals
                                .iter()
                                .filter(|&&x| {
                                    let z = (x - mean).abs() / std;
                                    z <= 4.0
                                })
                                .count();
                        } else {
                            valid_values += n;
                        }
                    }
                }
            }
        }

        let score = if total_values > 0 {
            valid_values as f64 / total_values as f64
        } else {
            1.0
        };

        QualityDimension {
            name: "validity".into(),
            score,
            details: format!(
                "{} of {} numeric values within ±4σ ({:.1}%)",
                valid_values,
                total_values,
                score * 100.0
            ),
        }
    }

    // ── Timeliness (datetime column recency / range) ────────────

    fn timeliness(&self) -> QualityDimension {
        let dt_cols = self.schema.datetime_columns();
        if dt_cols.is_empty() {
            return QualityDimension {
                name: "timeliness".into(),
                score: 1.0,
                details: "No datetime columns – timeliness check skipped".into(),
            };
        }

        // For now, score based on whether datetime columns parse correctly
        // Full implementation would check recency, gaps, etc.
        QualityDimension {
            name: "timeliness".into(),
            score: 0.8,
            details: format!("{} datetime columns detected", dt_cols.len()),
        }
    }

    // ── Conformity (no control characters, valid ranges) ────────

    fn conformity(&self) -> QualityDimension {
        let str_cols: Vec<&str> = self
            .schema
            .columns
            .iter()
            .filter(|c| {
                c.inferred_type == ColumnType::Categorical || c.inferred_type == ColumnType::Text
            })
            .map(|c| c.name.as_str())
            .collect();

        if str_cols.is_empty() {
            return QualityDimension {
                name: "conformity".into(),
                score: 1.0,
                details: "No string columns to check".into(),
            };
        }

        let mut total_strings = 0usize;
        let mut conforming_strings = 0usize;

        for &col_name in &str_cols {
            if let Ok(col) = self.df.column(col_name) {
                if let Ok(str_col) = col.str() {
                    for opt_val in str_col.into_iter() {
                        if let Some(s) = opt_val {
                            total_strings += 1;
                            // Check for control characters (except \n, \r, \t)
                            let has_control = s
                                .chars()
                                .any(|c| c.is_control() && c != '\n' && c != '\r' && c != '\t');
                            if !has_control {
                                conforming_strings += 1;
                            }
                        }
                    }
                }
            }
        }

        let score = if total_strings > 0 {
            conforming_strings as f64 / total_strings as f64
        } else {
            1.0
        };

        QualityDimension {
            name: "conformity".into(),
            score,
            details: format!(
                "{} of {} strings conform to standards ({:.1}%)",
                conforming_strings,
                total_strings,
                score * 100.0
            ),
        }
    }

    // ── Per-column quality ──────────────────────────────────────

    fn per_column_quality(&self) -> Vec<ColumnQuality> {
        let n_rows = self.df.height();

        self.df
            .get_columns()
            .iter()
            .map(|col| {
                let name = col.name().to_string();
                let completeness = if n_rows > 0 {
                    1.0 - (col.null_count() as f64 / n_rows as f64)
                } else {
                    1.0
                };
                let n_unique = col.n_unique().unwrap_or(0);
                let uniqueness = if n_rows > 0 {
                    n_unique as f64 / n_rows as f64
                } else {
                    1.0
                };

                // Validity: for numeric, ratio within ±4σ; for others, non-null = valid
                let validity = if col.dtype().is_float() || col.dtype().is_integer() {
                    Self::column_validity_numeric(col)
                } else {
                    completeness
                };

                let overall = (completeness + uniqueness.min(1.0) + validity) / 3.0;

                ColumnQuality {
                    column: name,
                    completeness,
                    uniqueness: uniqueness.min(1.0),
                    validity,
                    overall,
                }
            })
            .collect()
    }

    fn column_validity_numeric(col: &Column) -> f64 {
        if let Ok(casted) = col.cast(&DataType::Float64) {
            if let Ok(ca) = casted.f64() {
                let vals: Vec<f64> = ca.into_iter().filter_map(|v| v).collect();
                let n = vals.len();
                if n == 0 {
                    return 1.0;
                }
                let mean = vals.iter().sum::<f64>() / n as f64;
                let std = (vals.iter().map(|x| (x - mean).powi(2)).sum::<f64>()
                    / (n as f64 - 1.0).max(1.0))
                .sqrt();

                if std < f64::EPSILON {
                    return 1.0;
                }

                let valid = vals
                    .iter()
                    .filter(|&&x| ((x - mean) / std).abs() <= 4.0)
                    .count();
                return valid as f64 / n as f64;
            }
        }
        1.0
    }
}
