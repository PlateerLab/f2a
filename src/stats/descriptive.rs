use indexmap::IndexMap;
use polars::prelude::*;
use serde::{Deserialize, Serialize};
use statrs::statistics::{Data, Distribution, Max, Min, OrderStatistics};

use crate::core::schema::DataSchema;

// ─── Result types ───────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NumericColumnStats {
    pub column: String,
    pub count: usize,
    pub mean: f64,
    pub std: f64,
    pub se: f64,  // standard error of the mean
    pub cv: f64,  // coefficient of variation
    pub mad: f64, // median absolute deviation
    pub min: f64,
    pub p5: f64,
    pub q1: f64,
    pub median: f64,
    pub q3: f64,
    pub p95: f64,
    pub max: f64,
    pub range: f64,
    pub iqr: f64,
    pub skewness: f64,
    pub kurtosis: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CategoricalColumnStats {
    pub column: String,
    pub count: usize,
    pub unique: usize,
    pub top: String,
    pub freq: usize,
    pub frequencies: Vec<(String, usize)>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DescriptiveResult {
    pub numeric: Vec<NumericColumnStats>,
    pub categorical: Vec<CategoricalColumnStats>,
    pub summary: IndexMap<String, serde_json::Value>,
}

// ─── Implementation ─────────────────────────────────────────────────

pub struct DescriptiveStats<'a> {
    df: &'a DataFrame,
    schema: &'a DataSchema,
}

impl<'a> DescriptiveStats<'a> {
    pub fn new(df: &'a DataFrame, schema: &'a DataSchema) -> Self {
        Self { df, schema }
    }

    /// Compute descriptive statistics for all columns.
    pub fn compute(&self) -> DescriptiveResult {
        let numeric = self.compute_numeric();
        let categorical = self.compute_categorical();
        let summary = self.build_summary(&numeric, &categorical);

        DescriptiveResult {
            numeric,
            categorical,
            summary,
        }
    }

    /// Numeric column statistics.
    fn compute_numeric(&self) -> Vec<NumericColumnStats> {
        let num_cols = self.schema.numeric_columns();

        num_cols
            .iter()
            .filter_map(|&col_name| {
                let col = self.df.column(col_name).ok()?;
                let values = Self::column_to_f64_vec(col)?;

                if values.is_empty() {
                    return None;
                }

                Some(Self::compute_numeric_stats(col_name, &values))
            })
            .collect()
    }

    /// Compute stats for a single numeric column.
    fn compute_numeric_stats(name: &str, values: &[f64]) -> NumericColumnStats {
        let n = values.len();
        let mut data = Data::new(values.to_vec());

        let mean = data.mean().unwrap_or(f64::NAN);
        let std = data.std_dev().unwrap_or(f64::NAN);
        let variance = data.variance().unwrap_or(f64::NAN);
        let median = data.median();
        let min = data.min();
        let max = data.max();

        // Percentiles
        let p5 = data.percentile(5);
        let q1 = data.percentile(25);
        let q3 = data.percentile(75);
        let p95 = data.percentile(95);
        let iqr = q3 - q1;

        // Standard error of the mean
        let se = if n > 0 {
            std / (n as f64).sqrt()
        } else {
            f64::NAN
        };

        // Coefficient of variation
        let cv = if mean.abs() > f64::EPSILON {
            std / mean.abs()
        } else {
            f64::NAN
        };

        // Median absolute deviation
        let mad = {
            let mut deviations: Vec<f64> = values.iter().map(|x| (x - median).abs()).collect();
            deviations.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            if deviations.is_empty() {
                f64::NAN
            } else {
                let mid = deviations.len() / 2;
                if deviations.len() % 2 == 0 {
                    (deviations[mid - 1] + deviations[mid]) / 2.0
                } else {
                    deviations[mid]
                }
            }
        };

        // Skewness (Fisher's definition)
        let skewness = if n >= 3 && variance > f64::EPSILON {
            let m3: f64 = values
                .iter()
                .map(|x| ((x - mean) / std).powi(3))
                .sum::<f64>()
                / n as f64;
            let adjustment = ((n * (n - 1)) as f64).sqrt() / (n - 2) as f64;
            m3 * adjustment
        } else {
            f64::NAN
        };

        // Excess kurtosis (Fisher's definition)
        let kurtosis = if n >= 4 && variance > f64::EPSILON {
            let m4: f64 = values
                .iter()
                .map(|x| ((x - mean) / std).powi(4))
                .sum::<f64>()
                / n as f64;
            // Adjusted Fisher kurtosis
            let nf = n as f64;
            let excess =
                (nf - 1.0) / ((nf - 2.0) * (nf - 3.0)) * ((nf + 1.0) * m4 - 3.0 * (nf - 1.0));
            excess
        } else {
            f64::NAN
        };

        NumericColumnStats {
            column: name.to_string(),
            count: n,
            mean,
            std,
            se,
            cv,
            mad,
            min,
            p5,
            q1,
            median,
            q3,
            p95,
            max,
            range: max - min,
            iqr,
            skewness,
            kurtosis,
        }
    }

    /// Categorical column statistics.
    fn compute_categorical(&self) -> Vec<CategoricalColumnStats> {
        let cat_cols = self.schema.categorical_columns();

        cat_cols
            .iter()
            .filter_map(|&col_name| {
                let col = self.df.column(col_name).ok()?;
                let str_col = col.cast(&DataType::String).ok()?;
                let ca = str_col.str().ok()?;

                let mut freq_map: IndexMap<String, usize> = IndexMap::new();
                for opt_val in ca.into_iter() {
                    let key = opt_val.unwrap_or("(missing)").to_string();
                    *freq_map.entry(key).or_insert(0) += 1;
                }

                // Sort by frequency descending
                let mut frequencies: Vec<(String, usize)> = freq_map.into_iter().collect();
                frequencies.sort_by(|a, b| b.1.cmp(&a.1));

                let (top, freq) = frequencies.first().cloned().unwrap_or(("".to_string(), 0));

                let count = ca.len();
                let unique = ca.n_unique().unwrap_or(0);

                Some(CategoricalColumnStats {
                    column: col_name.to_string(),
                    count,
                    unique,
                    top,
                    freq,
                    frequencies,
                })
            })
            .collect()
    }

    /// Build an overall summary.
    fn build_summary(
        &self,
        numeric: &[NumericColumnStats],
        categorical: &[CategoricalColumnStats],
    ) -> IndexMap<String, serde_json::Value> {
        let mut summary = IndexMap::new();
        summary.insert(
            "n_rows".into(),
            serde_json::Value::Number(self.schema.n_rows.into()),
        );
        summary.insert(
            "n_cols".into(),
            serde_json::Value::Number(self.schema.n_cols.into()),
        );
        summary.insert(
            "n_numeric".into(),
            serde_json::Value::Number(numeric.len().into()),
        );
        summary.insert(
            "n_categorical".into(),
            serde_json::Value::Number(categorical.len().into()),
        );
        summary.insert(
            "memory_mb".into(),
            serde_json::json!(self.schema.memory_usage_bytes as f64 / 1_048_576.0),
        );
        summary
    }

    // ── Helpers ─────────────────────────────────────────────────

    /// Extract non-null f64 values from a column.
    pub(crate) fn column_to_f64_vec(col: &Column) -> Option<Vec<f64>> {
        let casted = col.cast(&DataType::Float64).ok()?;
        let ca = casted.f64().ok()?;
        let values: Vec<f64> = ca
            .into_iter()
            .filter_map(|v| v)
            .filter(|v| v.is_finite())
            .collect();
        if values.is_empty() {
            None
        } else {
            Some(values)
        }
    }
}
