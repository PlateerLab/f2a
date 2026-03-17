use polars::prelude::*;
use serde::{Deserialize, Serialize};

use crate::core::schema::DataSchema;
use crate::stats::descriptive::DescriptiveStats;

// ─── Types ──────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum InsightSeverity {
    Info,
    Warning,
    Critical,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum InsightType {
    HighMissing,
    ConstantColumn,
    HighCardinality,
    DuplicateRows,
    SkewedDistribution,
    HighCorrelation,
    OutlierProportion,
    ClassImbalance,
    InfiniteValues,
    LowVariance,
    IdLikeColumn,
    MixedTypes,
    DatetimePattern,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Insight {
    pub insight_type: InsightType,
    pub severity: InsightSeverity,
    pub column: Option<String>,
    pub message: String,
    pub detail: String,
    pub recommendation: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InsightEngineResult {
    pub insights: Vec<Insight>,
    pub summary: InsightSummary,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InsightSummary {
    pub total: usize,
    pub critical: usize,
    pub warning: usize,
    pub info: usize,
}

// ─── Engine ─────────────────────────────────────────────────────────

pub struct InsightEngine<'a> {
    df: &'a DataFrame,
    schema: &'a DataSchema,
    missing_threshold: f64,
    high_card_threshold: f64,
    corr_threshold: f64,
    outlier_threshold: f64,
    skew_threshold: f64,
    variance_threshold: f64,
}

impl<'a> InsightEngine<'a> {
    pub fn new(df: &'a DataFrame, schema: &'a DataSchema) -> Self {
        Self {
            df,
            schema,
            missing_threshold: 0.3,
            high_card_threshold: 0.9,
            corr_threshold: 0.95,
            outlier_threshold: 0.05,
            skew_threshold: 2.0,
            variance_threshold: 0.01,
        }
    }

    pub fn with_thresholds(
        mut self,
        missing: f64,
        high_card: f64,
        corr: f64,
        outlier: f64,
        skew: f64,
        variance: f64,
    ) -> Self {
        self.missing_threshold = missing;
        self.high_card_threshold = high_card;
        self.corr_threshold = corr;
        self.outlier_threshold = outlier;
        self.skew_threshold = skew;
        self.variance_threshold = variance;
        self
    }

    pub fn compute(&self) -> InsightEngineResult {
        let mut insights = Vec::new();

        self.check_high_missing(&mut insights);
        self.check_constant_columns(&mut insights);
        self.check_high_cardinality(&mut insights);
        self.check_duplicate_rows(&mut insights);
        self.check_skewed_distributions(&mut insights);
        self.check_high_correlations(&mut insights);
        self.check_outlier_proportions(&mut insights);
        self.check_infinite_values(&mut insights);
        self.check_low_variance(&mut insights);
        self.check_id_like_columns(&mut insights);
        self.check_class_imbalance(&mut insights);
        self.check_mixed_types(&mut insights);
        self.check_datetime_patterns(&mut insights);

        // Build summary
        let critical = insights
            .iter()
            .filter(|i| i.severity == InsightSeverity::Critical)
            .count();
        let warning = insights
            .iter()
            .filter(|i| i.severity == InsightSeverity::Warning)
            .count();
        let info = insights
            .iter()
            .filter(|i| i.severity == InsightSeverity::Info)
            .count();

        let summary = InsightSummary {
            total: insights.len(),
            critical,
            warning,
            info,
        };

        InsightEngineResult { insights, summary }
    }

    // ── Rule 1: High Missing ────────────────────────────────────

    fn check_high_missing(&self, out: &mut Vec<Insight>) {
        for col_info in &self.schema.columns {
            if col_info.missing_ratio >= self.missing_threshold {
                let severity = if col_info.missing_ratio > 0.7 {
                    InsightSeverity::Critical
                } else {
                    InsightSeverity::Warning
                };
                out.push(Insight {
                    insight_type: InsightType::HighMissing,
                    severity,
                    column: Some(col_info.name.clone()),
                    message: format!(
                        "Column '{}' has {:.1}% missing values",
                        col_info.name,
                        col_info.missing_ratio * 100.0
                    ),
                    detail: format!(
                        "{} out of {} values are missing",
                        col_info.n_missing, self.schema.n_rows
                    ),
                    recommendation: if col_info.missing_ratio > 0.7 {
                        "Consider dropping this column".into()
                    } else {
                        "Consider imputation or investigate the cause".into()
                    },
                });
            }
        }
    }

    // ── Rule 2: Constant Columns ────────────────────────────────

    fn check_constant_columns(&self, out: &mut Vec<Insight>) {
        for col_info in &self.schema.columns {
            if col_info.n_unique <= 1 {
                out.push(Insight {
                    insight_type: InsightType::ConstantColumn,
                    severity: InsightSeverity::Warning,
                    column: Some(col_info.name.clone()),
                    message: format!(
                        "Column '{}' is constant (only {} unique value)",
                        col_info.name, col_info.n_unique
                    ),
                    detail: "Constant columns carry no information".into(),
                    recommendation: "Drop this column as it has no predictive power".into(),
                });
            }
        }
    }

    // ── Rule 3: High Cardinality ─────────────────────────────────

    fn check_high_cardinality(&self, out: &mut Vec<Insight>) {
        let n = self.schema.n_rows.max(1);
        for col_name in self.schema.categorical_columns() {
            if let Some(info) = self.schema.columns.iter().find(|c| c.name == col_name) {
                let ratio = info.n_unique as f64 / n as f64;
                if ratio > self.high_card_threshold {
                    out.push(Insight {
                        insight_type: InsightType::HighCardinality,
                        severity: InsightSeverity::Warning,
                        column: Some(col_name.to_string()),
                        message: format!(
                            "Categorical column '{}' has very high cardinality ({} unique / {} rows = {:.1}%)",
                            col_name, info.n_unique, n, ratio * 100.0
                        ),
                        detail: "High cardinality may cause one-hot encoding explosion".into(),
                        recommendation: "Use target encoding, hashing, or grouping rare categories".into(),
                    });
                }
            }
        }
    }

    // ── Rule 4: Duplicate Rows ──────────────────────────────────

    fn check_duplicate_rows(&self, out: &mut Vec<Insight>) {
        let n = self.df.height();
        if n == 0 {
            return;
        }
        if let Ok(unique) = self.df.unique_stable(None, UniqueKeepStrategy::First, None) {
            let n_dup = n - unique.height();
            if n_dup > 0 {
                let ratio = n_dup as f64 / n as f64;
                let severity = if ratio > 0.1 {
                    InsightSeverity::Warning
                } else {
                    InsightSeverity::Info
                };
                out.push(Insight {
                    insight_type: InsightType::DuplicateRows,
                    severity,
                    column: None,
                    message: format!(
                        "Dataset has {} duplicate rows ({:.1}%)",
                        n_dup,
                        ratio * 100.0
                    ),
                    detail: format!("{} rows are exact duplicates", n_dup),
                    recommendation: "Investigate and remove duplicates if not intentional".into(),
                });
            }
        }
    }

    // ── Rule 5: Skewed Distributions ────────────────────────────

    fn check_skewed_distributions(&self, out: &mut Vec<Insight>) {
        for col_name in self.schema.numeric_columns() {
            if let Some(vals) = self
                .df
                .column(col_name)
                .ok()
                .and_then(|c| DescriptiveStats::column_to_f64_vec(c))
            {
                let skew = Self::compute_skewness(&vals);
                if skew.abs() > self.skew_threshold {
                    out.push(Insight {
                        insight_type: InsightType::SkewedDistribution,
                        severity: InsightSeverity::Info,
                        column: Some(col_name.to_string()),
                        message: format!(
                            "Column '{}' is heavily skewed (skewness = {:.3})",
                            col_name, skew
                        ),
                        detail: if skew > 0.0 {
                            "Right-skewed distribution".into()
                        } else {
                            "Left-skewed distribution".into()
                        },
                        recommendation: "Consider log, Box-Cox, or Yeo-Johnson transform".into(),
                    });
                }
            }
        }
    }

    // ── Rule 6: High Correlations ───────────────────────────────

    fn check_high_correlations(&self, out: &mut Vec<Insight>) {
        let num_cols = self.schema.numeric_columns();
        if num_cols.len() < 2 {
            return;
        }

        let mut col_vals: Vec<(&str, Vec<f64>)> = Vec::new();
        for &col_name in &num_cols {
            if let Some(vals) = self
                .df
                .column(col_name)
                .ok()
                .and_then(|c| DescriptiveStats::column_to_f64_vec(c))
            {
                col_vals.push((col_name, vals));
            }
        }

        let min_len = col_vals.iter().map(|(_, v)| v.len()).min().unwrap_or(0);
        for i in 0..col_vals.len() {
            for j in (i + 1)..col_vals.len() {
                let r = Self::pearson(&col_vals[i].1[..min_len], &col_vals[j].1[..min_len]);
                if r.abs() > self.corr_threshold {
                    out.push(Insight {
                        insight_type: InsightType::HighCorrelation,
                        severity: InsightSeverity::Warning,
                        column: Some(format!("{}, {}", col_vals[i].0, col_vals[j].0)),
                        message: format!(
                            "Columns '{}' and '{}' are highly correlated (r = {:.3})",
                            col_vals[i].0, col_vals[j].0, r
                        ),
                        detail: "Near-perfect correlation may indicate redundancy".into(),
                        recommendation: "Consider removing one to reduce multicollinearity".into(),
                    });
                }
            }
        }
    }

    // ── Rule 7: Outlier Proportions ─────────────────────────────

    fn check_outlier_proportions(&self, out: &mut Vec<Insight>) {
        for col_name in self.schema.numeric_columns() {
            if let Some(vals) = self
                .df
                .column(col_name)
                .ok()
                .and_then(|c| DescriptiveStats::column_to_f64_vec(c))
            {
                let n = vals.len();
                if n < 10 {
                    continue;
                }
                let mut sorted = vals.clone();
                sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                let q1 = sorted[n / 4];
                let q3 = sorted[3 * n / 4];
                let iqr = q3 - q1;
                let lower = q1 - 1.5 * iqr;
                let upper = q3 + 1.5 * iqr;
                let outlier_count = vals.iter().filter(|&&v| v < lower || v > upper).count();
                let ratio = outlier_count as f64 / n as f64;

                if ratio > self.outlier_threshold {
                    out.push(Insight {
                        insight_type: InsightType::OutlierProportion,
                        severity: if ratio > 0.15 {
                            InsightSeverity::Warning
                        } else {
                            InsightSeverity::Info
                        },
                        column: Some(col_name.to_string()),
                        message: format!(
                            "Column '{}' has {:.1}% outliers (IQR method)",
                            col_name,
                            ratio * 100.0
                        ),
                        detail: format!("{} outlier values detected out of {}", outlier_count, n),
                        recommendation:
                            "Investigate outliers — clip, transform, or treat separately".into(),
                    });
                }
            }
        }
    }

    // ── Rule 8: Infinite Values ─────────────────────────────────

    fn check_infinite_values(&self, out: &mut Vec<Insight>) {
        for col_name in self.schema.numeric_columns() {
            if let Ok(col) = self.df.column(col_name) {
                if let Ok(f) = col.cast(&DataType::Float64) {
                    let ca = f.f64().unwrap();
                    let inf_count = ca
                        .into_iter()
                        .filter(|v| matches!(v, Some(x) if x.is_infinite()))
                        .count();
                    if inf_count > 0 {
                        out.push(Insight {
                            insight_type: InsightType::InfiniteValues,
                            severity: InsightSeverity::Critical,
                            column: Some(col_name.to_string()),
                            message: format!(
                                "Column '{}' contains {} infinite values",
                                col_name, inf_count
                            ),
                            detail: "Infinite values will cause issues with most algorithms".into(),
                            recommendation:
                                "Replace infinite values with NaN or clip to a max value".into(),
                        });
                    }
                }
            }
        }
    }

    // ── Rule 9: Low Variance ────────────────────────────────────

    fn check_low_variance(&self, out: &mut Vec<Insight>) {
        for col_name in self.schema.numeric_columns() {
            if let Some(vals) = self
                .df
                .column(col_name)
                .ok()
                .and_then(|c| DescriptiveStats::column_to_f64_vec(c))
            {
                if vals.is_empty() {
                    continue;
                }
                let mean = vals.iter().sum::<f64>() / vals.len() as f64;
                let var = vals.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / vals.len() as f64;
                if var < self.variance_threshold && var > 0.0 {
                    out.push(Insight {
                        insight_type: InsightType::LowVariance,
                        severity: InsightSeverity::Info,
                        column: Some(col_name.to_string()),
                        message: format!(
                            "Column '{}' has very low variance ({:.6})",
                            col_name, var
                        ),
                        detail: "Low variance features may not contribute to model learning".into(),
                        recommendation: "Consider removing or investigating this feature".into(),
                    });
                }
            }
        }
    }

    // ── Rule 10: ID-like Columns ────────────────────────────────

    fn check_id_like_columns(&self, out: &mut Vec<Insight>) {
        let n = self.schema.n_rows.max(1);
        for col_info in &self.schema.columns {
            let ratio = col_info.n_unique as f64 / n as f64;
            if ratio > 0.99 && col_info.n_missing == 0 {
                let name_lower = col_info.name.to_lowercase();
                if name_lower.contains("id")
                    || name_lower.contains("index")
                    || name_lower.contains("key")
                    || name_lower.ends_with("_no")
                    || name_lower.ends_with("_num")
                {
                    out.push(Insight {
                        insight_type: InsightType::IdLikeColumn,
                        severity: InsightSeverity::Warning,
                        column: Some(col_info.name.clone()),
                        message: format!(
                            "Column '{}' appears to be an ID column ({} unique values)",
                            col_info.name, col_info.n_unique
                        ),
                        detail: "ID columns have no predictive value".into(),
                        recommendation: "Drop ID columns before modeling".into(),
                    });
                }
            }
        }
    }

    // ── Rule 11: Class Imbalance ────────────────────────────────

    fn check_class_imbalance(&self, out: &mut Vec<Insight>) {
        let n = self.df.height();
        for col_name in self.schema.categorical_columns() {
            if let Some(info) = self.schema.columns.iter().find(|c| c.name == col_name) {
                if info.n_unique >= 2 && info.n_unique <= 10 {
                    if let Ok(col) = self.df.column(col_name) {
                        let counts: Vec<usize> = col
                            .unique()
                            .ok()
                            .map(|u| {
                                (0..u.len())
                                    .filter_map(|i| {
                                        u.get(i).ok().map(|val| {
                                            let val_str = format!("{:?}", val);
                                            (0..col.len())
                                                .filter(|&j| {
                                                    col.get(j)
                                                        .ok()
                                                        .map(|v| format!("{:?}", v) == val_str)
                                                        .unwrap_or(false)
                                                })
                                                .count()
                                        })
                                    })
                                    .collect()
                            })
                            .unwrap_or_default();

                        if let (Some(&min_c), Some(&max_c)) =
                            (counts.iter().min(), counts.iter().max())
                        {
                            if max_c > 0 && n > 0 {
                                let ratio = min_c as f64 / max_c as f64;
                                if ratio < 0.1 {
                                    out.push(Insight {
                                        insight_type: InsightType::ClassImbalance,
                                        severity: InsightSeverity::Warning,
                                        column: Some(col_name.to_string()),
                                        message: format!(
                                            "Column '{}' shows class imbalance (min/max ratio = {:.3})",
                                            col_name, ratio
                                        ),
                                        detail: format!(
                                            "Smallest class: {} samples, largest: {} samples",
                                            min_c, max_c
                                        ),
                                        recommendation:
                                            "Consider oversampling, undersampling, or class weights"
                                                .into(),
                                    });
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // ── Rule 12: Mixed Types ────────────────────────────────────

    fn check_mixed_types(&self, out: &mut Vec<Insight>) {
        // Detect columns where dtype is string but many values look numeric
        for col_name in self.schema.categorical_columns() {
            if let Ok(col) = self.df.column(col_name) {
                let total = col.len();
                if total == 0 {
                    continue;
                }
                let str_col = col.cast(&DataType::String).ok();
                if let Some(s) = str_col {
                    if let Ok(ca) = s.str() {
                        let numeric_count = ca
                            .into_iter()
                            .filter(|v| v.map(|s| s.parse::<f64>().is_ok()).unwrap_or(false))
                            .count();
                        let ratio = numeric_count as f64 / total as f64;
                        if ratio > 0.5 && ratio < 0.99 {
                            out.push(Insight {
                                insight_type: InsightType::MixedTypes,
                                severity: InsightSeverity::Warning,
                                column: Some(col_name.to_string()),
                                message: format!(
                                    "Column '{}' appears to have mixed types ({:.0}% numeric in string column)",
                                    col_name,
                                    ratio * 100.0
                                ),
                                detail: format!("{} of {} values are parseable as numbers", numeric_count, total),
                                recommendation: "Clean and convert to the appropriate type".into(),
                            });
                        }
                    }
                }
            }
        }
    }

    // ── Rule 13: DateTime Patterns ──────────────────────────────

    fn check_datetime_patterns(&self, out: &mut Vec<Insight>) {
        for col_name in self.schema.datetime_columns() {
            // Just note datetime columns exist — useful for time series
            out.push(Insight {
                insight_type: InsightType::DatetimePattern,
                severity: InsightSeverity::Info,
                column: Some(col_name.to_string()),
                message: format!("Column '{}' contains datetime data", col_name),
                detail: "Datetime columns can be decomposed into year, month, day, etc.".into(),
                recommendation: "Extract temporal features for improved modeling".into(),
            });
        }
    }

    // ── Stat helpers ────────────────────────────────────────────

    fn compute_skewness(vals: &[f64]) -> f64 {
        let n = vals.len() as f64;
        if n < 3.0 {
            return 0.0;
        }
        let mean = vals.iter().sum::<f64>() / n;
        let m2: f64 = vals.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n;
        let m3: f64 = vals.iter().map(|v| (v - mean).powi(3)).sum::<f64>() / n;
        let std = m2.sqrt();
        if std < f64::EPSILON {
            return 0.0;
        }
        m3 / std.powi(3)
    }

    fn pearson(x: &[f64], y: &[f64]) -> f64 {
        let n = x.len().min(y.len()) as f64;
        if n < 2.0 {
            return 0.0;
        }
        let mx = x.iter().sum::<f64>() / n;
        let my = y.iter().sum::<f64>() / n;
        let mut cov = 0.0;
        let mut sx = 0.0;
        let mut sy = 0.0;
        for i in 0..x.len().min(y.len()) {
            let dx = x[i] - mx;
            let dy = y[i] - my;
            cov += dx * dy;
            sx += dx * dx;
            sy += dy * dy;
        }
        if sx < f64::EPSILON || sy < f64::EPSILON {
            return 0.0;
        }
        cov / (sx * sy).sqrt()
    }
}
