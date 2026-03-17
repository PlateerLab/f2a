use polars::prelude::*;
use serde::{Deserialize, Serialize};

use crate::core::schema::DataSchema;
use crate::stats::descriptive::DescriptiveStats;
use crate::utils::types::ColumnType;

// ─── Result types ───────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DimensionScore {
    pub name: String,
    pub score: f64,
    pub detail: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ColumnReadiness {
    pub column: String,
    pub data_type: String,
    pub needs_encoding: bool,
    pub needs_imputation: bool,
    pub needs_scaling: bool,
    pub needs_transform: bool,
    pub warnings: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MlReadinessResult {
    pub overall_score: f64,
    pub grade: String,
    pub dimensions: Vec<DimensionScore>,
    pub column_readiness: Vec<ColumnReadiness>,
    pub recommendations: Vec<String>,
}

// ─── Implementation ─────────────────────────────────────────────────

pub struct MlReadinessStats<'a> {
    df: &'a DataFrame,
    schema: &'a DataSchema,
}

impl<'a> MlReadinessStats<'a> {
    pub fn new(df: &'a DataFrame, schema: &'a DataSchema) -> Self {
        Self { df, schema }
    }

    pub fn compute(&self) -> MlReadinessResult {
        let completeness = self.score_completeness();
        let uniqueness = self.score_uniqueness();
        let consistency = self.score_consistency();
        let balance = self.score_balance();
        let informativeness = self.score_informativeness();
        let independence = self.score_independence();
        let scale_uniformity = self.score_scale();

        let dimensions = vec![
            DimensionScore {
                name: "completeness".into(),
                score: completeness.0,
                detail: completeness.1,
            },
            DimensionScore {
                name: "uniqueness".into(),
                score: uniqueness.0,
                detail: uniqueness.1,
            },
            DimensionScore {
                name: "consistency".into(),
                score: consistency.0,
                detail: consistency.1,
            },
            DimensionScore {
                name: "balance".into(),
                score: balance.0,
                detail: balance.1,
            },
            DimensionScore {
                name: "informativeness".into(),
                score: informativeness.0,
                detail: informativeness.1,
            },
            DimensionScore {
                name: "independence".into(),
                score: independence.0,
                detail: independence.1,
            },
            DimensionScore {
                name: "scale_uniformity".into(),
                score: scale_uniformity.0,
                detail: scale_uniformity.1,
            },
        ];

        // Weighted overall score
        let weights = [0.20, 0.10, 0.15, 0.10, 0.15, 0.15, 0.15];
        let scores: Vec<f64> = dimensions.iter().map(|d| d.score).collect();
        let overall_score: f64 = scores.iter().zip(weights.iter()).map(|(s, w)| s * w).sum();

        let grade = Self::grade_from_score(overall_score);
        let column_readiness = self.column_readiness();
        let recommendations = self.generate_recommendations(&dimensions, &column_readiness);

        MlReadinessResult {
            overall_score,
            grade,
            dimensions,
            column_readiness,
            recommendations,
        }
    }

    // ── Dimension 1: Completeness ───────────────────────────────

    fn score_completeness(&self) -> (f64, String) {
        if self.schema.columns.is_empty() {
            return (1.0, "No columns to evaluate".into());
        }
        let avg_complete: f64 = self
            .schema
            .columns
            .iter()
            .map(|c| 1.0 - c.missing_ratio)
            .sum::<f64>()
            / self.schema.columns.len() as f64;

        let n_high_missing = self
            .schema
            .columns
            .iter()
            .filter(|c| c.missing_ratio > 0.3)
            .count();

        let penalty = (n_high_missing as f64 * 0.05).min(0.3);
        let score = (avg_complete - penalty).clamp(0.0, 1.0);

        (
            score,
            format!(
                "Avg completeness {:.1}%, {} columns with >30% missing",
                avg_complete * 100.0,
                n_high_missing
            ),
        )
    }

    // ── Dimension 2: Uniqueness (no excessive duplicates) ───────

    fn score_uniqueness(&self) -> (f64, String) {
        let n = self.df.height();
        if n == 0 {
            return (1.0, "Empty dataset".into());
        }

        let n_unique = self
            .df
            .unique_stable(None, UniqueKeepStrategy::First, None)
            .map(|u| u.height())
            .unwrap_or(n);

        let dup_ratio = 1.0 - (n_unique as f64 / n as f64);
        let score = (1.0 - dup_ratio * 2.0).clamp(0.0, 1.0);

        (score, format!("{:.1}% duplicate rows", dup_ratio * 100.0))
    }

    // ── Dimension 3: Consistency (no mixed types, no outlier excess)

    fn score_consistency(&self) -> (f64, String) {
        let mut issues = 0;

        // Check numeric columns for extreme outliers
        for col_name in self.schema.numeric_columns() {
            if let Some(vals) = self
                .df
                .column(col_name)
                .ok()
                .and_then(|c| DescriptiveStats::column_to_f64_vec(c))
            {
                if vals.len() < 10 {
                    continue;
                }
                let mut sorted = vals.clone();
                sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                let q1 = sorted[vals.len() / 4];
                let q3 = sorted[3 * vals.len() / 4];
                let iqr = q3 - q1;
                let outlier_count = vals
                    .iter()
                    .filter(|&&v| v < q1 - 3.0 * iqr || v > q3 + 3.0 * iqr)
                    .count();
                if outlier_count as f64 / vals.len() as f64 > 0.05 {
                    issues += 1;
                }
            }
        }

        // Check for infinite values
        for col_name in self.schema.numeric_columns() {
            if let Ok(col) = self.df.column(col_name) {
                if let Ok(f) = col.cast(&DataType::Float64) {
                    let ca = f.f64().unwrap();
                    let inf_count = ca
                        .into_iter()
                        .filter(|v| matches!(v, Some(x) if x.is_infinite()))
                        .count();
                    if inf_count > 0 {
                        issues += 1;
                    }
                }
            }
        }

        let score = (1.0 - issues as f64 * 0.1).clamp(0.0, 1.0);
        (score, format!("{} consistency issues found", issues))
    }

    // ── Dimension 4: Balance (class distribution) ───────────────

    fn score_balance(&self) -> (f64, String) {
        let cat_cols = self.schema.categorical_columns();
        if cat_cols.is_empty() {
            return (0.8, "No categorical columns to evaluate balance".into());
        }

        let mut worst_gini = 0.0f64;
        for &col_name in &cat_cols {
            if let Some(info) = self.schema.columns.iter().find(|c| c.name == col_name) {
                if info.n_unique >= 2 && info.n_unique <= 20 {
                    if let Ok(col) = self.df.column(col_name) {
                        let n = col.len();
                        let groups = Self::count_groups(col);
                        if groups.len() >= 2 {
                            let gini = Self::gini_impurity(&groups, n);
                            if gini > worst_gini {
                                worst_gini = gini;
                            }
                        }
                    }
                }
            }
        }

        // Higher gini = more balanced → better
        let score = worst_gini.clamp(0.0, 1.0);
        (score, format!("Gini impurity: {:.3}", worst_gini))
    }

    // ── Dimension 5: Informativeness ────────────────────────────

    fn score_informativeness(&self) -> (f64, String) {
        let n_cols = self.schema.columns.len().max(1);
        let mut low_info = 0usize;

        // Constant or near-constant columns
        for col_info in &self.schema.columns {
            if col_info.n_unique <= 1 {
                low_info += 1;
            }
        }

        // Low-variance numeric columns
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
                let cv = if mean.abs() > f64::EPSILON {
                    var.sqrt() / mean.abs()
                } else {
                    var.sqrt()
                };
                if cv < 0.01 {
                    low_info += 1;
                }
            }
        }

        let ratio = low_info as f64 / n_cols as f64;
        let score = (1.0 - ratio * 2.0).clamp(0.0, 1.0);
        (
            score,
            format!("{} of {} columns have low information", low_info, n_cols),
        )
    }

    // ── Dimension 6: Independence (low multicollinearity) ───────

    fn score_independence(&self) -> (f64, String) {
        let num_cols = self.schema.numeric_columns();
        if num_cols.len() < 2 {
            return (1.0, "Fewer than 2 numeric columns".into());
        }

        let mut col_vals: Vec<Vec<f64>> = Vec::new();
        for &name in &num_cols {
            if let Some(vals) = self
                .df
                .column(name)
                .ok()
                .and_then(|c| DescriptiveStats::column_to_f64_vec(c))
            {
                col_vals.push(vals);
            }
        }

        let min_len = col_vals.iter().map(|v| v.len()).min().unwrap_or(0);
        let mut high_corr_pairs = 0usize;
        let total_pairs = col_vals.len() * (col_vals.len() - 1) / 2;

        for i in 0..col_vals.len() {
            for j in (i + 1)..col_vals.len() {
                let r = Self::pearson(&col_vals[i][..min_len], &col_vals[j][..min_len]);
                if r.abs() > 0.9 {
                    high_corr_pairs += 1;
                }
            }
        }

        let ratio = if total_pairs > 0 {
            high_corr_pairs as f64 / total_pairs as f64
        } else {
            0.0
        };
        let score = (1.0 - ratio * 3.0).clamp(0.0, 1.0);
        (
            score,
            format!(
                "{} of {} column pairs have |r| > 0.9",
                high_corr_pairs, total_pairs
            ),
        )
    }

    // ── Dimension 7: Scale uniformity ───────────────────────────

    fn score_scale(&self) -> (f64, String) {
        let num_cols = self.schema.numeric_columns();
        if num_cols.is_empty() {
            return (1.0, "No numeric columns".into());
        }

        let mut ranges: Vec<f64> = Vec::new();
        for &col_name in &num_cols {
            if let Some(vals) = self
                .df
                .column(col_name)
                .ok()
                .and_then(|c| DescriptiveStats::column_to_f64_vec(c))
            {
                if vals.is_empty() {
                    continue;
                }
                let min = vals.iter().cloned().fold(f64::INFINITY, f64::min);
                let max = vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                ranges.push((max - min).abs());
            }
        }

        if ranges.len() < 2 {
            return (1.0, "Single numeric column".into());
        }

        let min_range = ranges.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_range = ranges.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        let ratio = if max_range > f64::EPSILON {
            min_range / max_range
        } else {
            1.0
        };

        // If ranges differ by > 1000x, scaling is needed
        let score = if ratio < 0.001 {
            0.3
        } else if ratio < 0.01 {
            0.5
        } else if ratio < 0.1 {
            0.7
        } else {
            0.9
        };

        (
            score,
            format!(
                "Feature range ratio: {:.4} (min range: {:.2}, max range: {:.2})",
                ratio, min_range, max_range
            ),
        )
    }

    // ── Per-column readiness ────────────────────────────────────

    fn column_readiness(&self) -> Vec<ColumnReadiness> {
        self.schema
            .columns
            .iter()
            .map(|col_info| {
                let mut warnings = Vec::new();
                let needs_imputation = col_info.n_missing > 0;
                let needs_encoding = col_info.inferred_type == ColumnType::Categorical
                    || col_info.inferred_type == ColumnType::Text;
                let needs_scaling = col_info.inferred_type == ColumnType::Numeric;

                // Check for skew → transform
                let needs_transform = if col_info.inferred_type == ColumnType::Numeric {
                    if let Some(vals) = self
                        .df
                        .column(&col_info.name)
                        .ok()
                        .and_then(|c| DescriptiveStats::column_to_f64_vec(c))
                    {
                        let skew = Self::compute_skewness(&vals);
                        if skew.abs() > 2.0 {
                            warnings.push(format!("High skewness ({:.2})", skew));
                            true
                        } else {
                            false
                        }
                    } else {
                        false
                    }
                } else {
                    false
                };

                if needs_imputation {
                    warnings.push(format!("{:.1}% missing", col_info.missing_ratio * 100.0));
                }
                if col_info.n_unique <= 1 {
                    warnings.push("Constant column".into());
                }

                ColumnReadiness {
                    column: col_info.name.clone(),
                    data_type: format!("{:?}", col_info.inferred_type),
                    needs_encoding,
                    needs_imputation,
                    needs_scaling,
                    needs_transform,
                    warnings,
                }
            })
            .collect()
    }

    // ── Recommendation generator ────────────────────────────────

    fn generate_recommendations(
        &self,
        dimensions: &[DimensionScore],
        columns: &[ColumnReadiness],
    ) -> Vec<String> {
        let mut recs = Vec::new();

        for dim in dimensions {
            if dim.score < 0.6 {
                match dim.name.as_str() {
                    "completeness" => recs.push("해결: 결측치가 많습니다. 결측 컬럼 제거 또는 대체 전략을 적용하세요.".into()),
                    "uniqueness" => recs.push("해결: 중복 행이 많습니다. 중복 제거를 고려하세요.".into()),
                    "consistency" => recs.push("해결: 이상치나 비정상 값이 많습니다. 클리핑 또는 변환을 고려하세요.".into()),
                    "balance" => recs.push("해결: 클래스 불균형이 발견되었습니다. 오버/언더 샘플링을 고려하세요.".into()),
                    "informativeness" => recs.push("해결: 정보량이 낮은 컬럼이 있습니다. 상수 컬럼 제거를 고려하세요.".into()),
                    "independence" => recs.push("해결: 다중공선성이 높습니다. 상관성 높은 컬럼 중 하나를 제거하세요.".into()),
                    "scale_uniformity" => recs.push("해결: 피처 스케일이 불균일합니다. StandardScaler 또는 MinMaxScaler를 적용하세요.".into()),
                    _ => {}
                }
            }
        }

        let n_needs_encoding = columns.iter().filter(|c| c.needs_encoding).count();
        if n_needs_encoding > 0 {
            recs.push(format!(
                "{}개 범주형 컬럼에 인코딩이 필요합니다 (OneHot, Target, 또는 Label Encoding).",
                n_needs_encoding
            ));
        }

        let n_needs_imputation = columns.iter().filter(|c| c.needs_imputation).count();
        if n_needs_imputation > 0 {
            recs.push(format!(
                "{}개 컬럼에 결측치 대체가 필요합니다 (mean, median, 또는 KNN imputation).",
                n_needs_imputation
            ));
        }

        recs
    }

    // ── Helpers ─────────────────────────────────────────────────

    fn grade_from_score(score: f64) -> String {
        if score >= 0.9 {
            "A".into()
        } else if score >= 0.8 {
            "B".into()
        } else if score >= 0.7 {
            "C".into()
        } else if score >= 0.6 {
            "D".into()
        } else {
            "F".into()
        }
    }

    fn count_groups(col: &Column) -> Vec<usize> {
        let mut map: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
        for i in 0..col.len() {
            if let Ok(val) = col.get(i) {
                let key = format!("{}", val);
                *map.entry(key).or_default() += 1;
            }
        }
        map.values().copied().collect()
    }

    fn gini_impurity(counts: &[usize], total: usize) -> f64 {
        if total == 0 {
            return 0.0;
        }
        let n = total as f64;
        1.0 - counts.iter().map(|&c| (c as f64 / n).powi(2)).sum::<f64>()
    }

    fn compute_skewness(vals: &[f64]) -> f64 {
        let n = vals.len() as f64;
        if n < 3.0 {
            return 0.0;
        }
        let mean = vals.iter().sum::<f64>() / n;
        let m2 = vals.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n;
        let m3 = vals.iter().map(|v| (v - mean).powi(3)).sum::<f64>() / n;
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
