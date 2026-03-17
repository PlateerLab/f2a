use polars::prelude::*;
use serde::{Deserialize, Serialize};
use statrs::statistics::{Data, OrderStatistics};

pub use crate::core::config::OutlierMethod;
use crate::core::schema::DataSchema;
use crate::stats::descriptive::DescriptiveStats;

// ─── Result types ───────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutlierColumnResult {
    pub column: String,
    pub method: String,
    pub n_outliers: usize,
    pub outlier_ratio: f64,
    pub lower_bound: f64,
    pub upper_bound: f64,
    pub n_below: usize,
    pub n_above: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutlierResult {
    pub columns: Vec<OutlierColumnResult>,
    /// Per-column boolean mask: true = outlier row.
    pub masks: Vec<(String, Vec<bool>)>,
    pub total_outlier_cells: usize,
    pub total_cells: usize,
    pub overall_outlier_ratio: f64,
}

// ─── Implementation ─────────────────────────────────────────────────

pub struct OutlierStats<'a> {
    df: &'a DataFrame,
    schema: &'a DataSchema,
    method: OutlierMethod,
    threshold: f64,
}

impl<'a> OutlierStats<'a> {
    pub fn new(
        df: &'a DataFrame,
        schema: &'a DataSchema,
        method: OutlierMethod,
        threshold: f64,
    ) -> Self {
        Self {
            df,
            schema,
            method,
            threshold,
        }
    }

    pub fn compute(&self) -> OutlierResult {
        let num_cols = self.schema.numeric_columns();
        let n_rows = self.df.height();
        let mut columns = Vec::new();
        let mut masks = Vec::new();
        let mut total_outlier_cells = 0usize;

        for &col_name in &num_cols {
            let col = match self.df.column(col_name) {
                Ok(c) => c,
                Err(_) => continue,
            };

            // Get all values including NaN positions
            let (result, mask) = match self.method {
                OutlierMethod::Iqr => self.iqr_method(col_name, col),
                OutlierMethod::Zscore => self.zscore_method(col_name, col),
            };

            total_outlier_cells += result.n_outliers;
            columns.push(result);
            masks.push((col_name.to_string(), mask));
        }

        let total_cells = n_rows * num_cols.len();
        let overall_outlier_ratio = if total_cells > 0 {
            total_outlier_cells as f64 / total_cells as f64
        } else {
            0.0
        };

        OutlierResult {
            columns,
            masks,
            total_outlier_cells,
            total_cells,
            overall_outlier_ratio,
        }
    }

    /// IQR (Interquartile Range) method.
    fn iqr_method(&self, name: &str, col: &Column) -> (OutlierColumnResult, Vec<bool>) {
        let n_rows = self.df.height();
        let mut mask = vec![false; n_rows];

        // Get non-null values for percentile computation
        let values = match DescriptiveStats::column_to_f64_vec(col) {
            Some(v) => v,
            None => {
                return (
                    OutlierColumnResult {
                        column: name.to_string(),
                        method: "iqr".to_string(),
                        n_outliers: 0,
                        outlier_ratio: 0.0,
                        lower_bound: f64::NEG_INFINITY,
                        upper_bound: f64::INFINITY,
                        n_below: 0,
                        n_above: 0,
                    },
                    mask,
                );
            }
        };

        let mut data = Data::new(values.clone());
        let q1 = data.percentile(25);
        let q3 = data.percentile(75);
        let iqr = q3 - q1;
        let lower = q1 - self.threshold * iqr;
        let upper = q3 + self.threshold * iqr;

        // Apply to all rows (including original null positions)
        let casted = col.cast(&DataType::Float64).unwrap_or_default();
        let ca = casted.f64().unwrap();
        let mut n_below = 0usize;
        let mut n_above = 0usize;

        for (i, opt_val) in ca.into_iter().enumerate() {
            if let Some(v) = opt_val {
                if v < lower {
                    mask[i] = true;
                    n_below += 1;
                } else if v > upper {
                    mask[i] = true;
                    n_above += 1;
                }
            }
        }

        let n_outliers = n_below + n_above;
        let outlier_ratio = if n_rows > 0 {
            n_outliers as f64 / n_rows as f64
        } else {
            0.0
        };

        (
            OutlierColumnResult {
                column: name.to_string(),
                method: "iqr".to_string(),
                n_outliers,
                outlier_ratio,
                lower_bound: lower,
                upper_bound: upper,
                n_below,
                n_above,
            },
            mask,
        )
    }

    /// Z-score method.
    fn zscore_method(&self, name: &str, col: &Column) -> (OutlierColumnResult, Vec<bool>) {
        let n_rows = self.df.height();
        let mut mask = vec![false; n_rows];

        let values = match DescriptiveStats::column_to_f64_vec(col) {
            Some(v) => v,
            None => {
                return (
                    OutlierColumnResult {
                        column: name.to_string(),
                        method: "zscore".to_string(),
                        n_outliers: 0,
                        outlier_ratio: 0.0,
                        lower_bound: f64::NEG_INFINITY,
                        upper_bound: f64::INFINITY,
                        n_below: 0,
                        n_above: 0,
                    },
                    mask,
                );
            }
        };

        let n = values.len() as f64;
        let mean = values.iter().sum::<f64>() / n;
        let std =
            (values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0).max(1.0)).sqrt();

        if std < f64::EPSILON {
            return (
                OutlierColumnResult {
                    column: name.to_string(),
                    method: "zscore".to_string(),
                    n_outliers: 0,
                    outlier_ratio: 0.0,
                    lower_bound: mean,
                    upper_bound: mean,
                    n_below: 0,
                    n_above: 0,
                },
                mask,
            );
        }

        let lower = mean - self.threshold * std;
        let upper = mean + self.threshold * std;

        let casted = col.cast(&DataType::Float64).unwrap_or_default();
        let ca = casted.f64().unwrap();
        let mut n_below = 0usize;
        let mut n_above = 0usize;

        for (i, opt_val) in ca.into_iter().enumerate() {
            if let Some(v) = opt_val {
                let z = (v - mean) / std;
                if z < -self.threshold {
                    mask[i] = true;
                    n_below += 1;
                } else if z > self.threshold {
                    mask[i] = true;
                    n_above += 1;
                }
            }
        }

        let n_outliers = n_below + n_above;
        let outlier_ratio = if n_rows > 0 {
            n_outliers as f64 / n_rows as f64
        } else {
            0.0
        };

        (
            OutlierColumnResult {
                column: name.to_string(),
                method: "zscore".to_string(),
                n_outliers,
                outlier_ratio,
                lower_bound: lower,
                upper_bound: upper,
                n_below,
                n_above,
            },
            mask,
        )
    }
}
