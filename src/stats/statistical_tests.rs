use polars::prelude::*;
use serde::{Deserialize, Serialize};
use statrs::distribution::{ContinuousCDF, StudentsT};

use crate::core::schema::DataSchema;
use crate::stats::descriptive::DescriptiveStats;

// ─── Result types ───────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LeveneTestResult {
    pub col_a: String,
    pub col_b: String,
    pub statistic: f64,
    pub p_value: f64,
    pub log_var_ratio: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KruskalWallisResult {
    pub numeric_col: String,
    pub group_col: String,
    pub h_statistic: f64,
    pub p_value: f64,
    pub eta_squared: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MannWhitneyResult {
    pub col_a: String,
    pub col_b: String,
    pub u_statistic: f64,
    pub p_value: f64,
    pub rank_biserial_r: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GrubbsTestResult {
    pub column: String,
    pub statistic: f64,
    pub critical_value: f64,
    pub outlier_value: f64,
    pub is_outlier: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatisticalTestsResult {
    pub levene: Vec<LeveneTestResult>,
    pub kruskal_wallis: Vec<KruskalWallisResult>,
    pub grubbs: Vec<GrubbsTestResult>,
}

// ─── Implementation ─────────────────────────────────────────────────

pub struct StatisticalTests<'a> {
    df: &'a DataFrame,
    schema: &'a DataSchema,
}

impl<'a> StatisticalTests<'a> {
    pub fn new(df: &'a DataFrame, schema: &'a DataSchema) -> Self {
        Self { df, schema }
    }

    pub fn compute(&self) -> StatisticalTestsResult {
        let levene = self.levene_tests();
        let kruskal_wallis = self.kruskal_wallis_tests();
        let grubbs = self.grubbs_tests();

        StatisticalTestsResult {
            levene,
            kruskal_wallis,
            grubbs,
        }
    }

    /// Pairwise Levene's test for equality of variances between numeric columns.
    fn levene_tests(&self) -> Vec<LeveneTestResult> {
        let num_cols = self.schema.numeric_columns();
        let mut results = Vec::new();

        for i in 0..num_cols.len() {
            for j in (i + 1)..num_cols.len() {
                let a_vals = self
                    .df
                    .column(num_cols[i])
                    .ok()
                    .and_then(|c| DescriptiveStats::column_to_f64_vec(c));
                let b_vals = self
                    .df
                    .column(num_cols[j])
                    .ok()
                    .and_then(|c| DescriptiveStats::column_to_f64_vec(c));

                if let (Some(a), Some(b)) = (a_vals, b_vals) {
                    if a.len() < 2 || b.len() < 2 {
                        continue;
                    }

                    let med_a = Self::median_of(&a);
                    let med_b = Self::median_of(&b);
                    let za: Vec<f64> = a.iter().map(|x| (x - med_a).abs()).collect();
                    let zb: Vec<f64> = b.iter().map(|x| (x - med_b).abs()).collect();

                    let n1 = za.len() as f64;
                    let n2 = zb.len() as f64;
                    let mean_za = za.iter().sum::<f64>() / n1;
                    let mean_zb = zb.iter().sum::<f64>() / n2;
                    let grand_mean = (za.iter().sum::<f64>() + zb.iter().sum::<f64>()) / (n1 + n2);

                    let between =
                        n1 * (mean_za - grand_mean).powi(2) + n2 * (mean_zb - grand_mean).powi(2);
                    let within: f64 = za.iter().map(|z| (z - mean_za).powi(2)).sum::<f64>()
                        + zb.iter().map(|z| (z - mean_zb).powi(2)).sum::<f64>();

                    let dof_between = 1.0;
                    let dof_within = n1 + n2 - 2.0;
                    let f_stat = if within > f64::EPSILON {
                        (between / dof_between) / (within / dof_within)
                    } else {
                        0.0
                    };

                    // F-distribution p-value approximation
                    let p_value = Self::f_distribution_sf(f_stat, dof_between, dof_within);

                    let var_a = a.iter().map(|x| (x - med_a).powi(2)).sum::<f64>() / (n1 - 1.0);
                    let var_b = b.iter().map(|x| (x - med_b).powi(2)).sum::<f64>() / (n2 - 1.0);
                    let log_var_ratio = if var_b > f64::EPSILON {
                        (var_a / var_b).ln()
                    } else {
                        f64::NAN
                    };

                    results.push(LeveneTestResult {
                        col_a: num_cols[i].to_string(),
                        col_b: num_cols[j].to_string(),
                        statistic: f_stat,
                        p_value,
                        log_var_ratio,
                    });
                }
            }
        }
        results
    }

    /// Kruskal-Wallis H test: numeric column grouped by categorical.
    fn kruskal_wallis_tests(&self) -> Vec<KruskalWallisResult> {
        let num_cols = self.schema.numeric_columns();
        let cat_cols = self.schema.categorical_columns();
        let mut results = Vec::new();

        for &num_col in &num_cols {
            for &cat_col in &cat_cols {
                if let Some(r) = self.kruskal_wallis_pair(num_col, cat_col) {
                    results.push(r);
                }
            }
        }
        results
    }

    fn kruskal_wallis_pair(&self, num_col: &str, cat_col: &str) -> Option<KruskalWallisResult> {
        let num_series = self.df.column(num_col).ok()?;
        let cat_series = self.df.column(cat_col).ok()?;
        let cat_str = cat_series.cast(&DataType::String).ok()?;
        let cat_ca = cat_str.str().ok()?;

        // Group numeric values by category
        let mut groups: indexmap::IndexMap<String, Vec<f64>> = indexmap::IndexMap::new();
        let num_f64 = num_series.cast(&DataType::Float64).ok()?;
        let num_ca = num_f64.f64().ok()?;

        for (nv, cv) in num_ca.into_iter().zip(cat_ca.into_iter()) {
            if let (Some(n), Some(c)) = (nv, cv) {
                if n.is_finite() {
                    groups.entry(c.to_string()).or_default().push(n);
                }
            }
        }

        if groups.len() < 2 {
            return None;
        }

        // Rank all values together
        let mut all_vals: Vec<(f64, usize)> = Vec::new(); // (value, group_idx)
        let group_keys: Vec<String> = groups.keys().cloned().collect();
        for (g_idx, key) in group_keys.iter().enumerate() {
            for &val in &groups[key] {
                all_vals.push((val, g_idx));
            }
        }
        all_vals.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        let n_total = all_vals.len() as f64;
        let mut ranks = vec![0.0f64; all_vals.len()];

        // Assign average ranks for ties
        let mut i = 0;
        while i < all_vals.len() {
            let mut j = i + 1;
            while j < all_vals.len() && (all_vals[j].0 - all_vals[i].0).abs() < f64::EPSILON {
                j += 1;
            }
            let avg_rank = (i + 1..=j).sum::<usize>() as f64 / (j - i) as f64;
            for k in i..j {
                ranks[k] = avg_rank;
            }
            i = j;
        }

        // Compute H statistic
        let mut group_rank_sums = vec![0.0f64; groups.len()];
        let mut group_ns = vec![0usize; groups.len()];
        for (idx, (_, g_idx)) in all_vals.iter().enumerate() {
            group_rank_sums[*g_idx] += ranks[idx];
            group_ns[*g_idx] += 1;
        }

        let h: f64 = group_rank_sums
            .iter()
            .zip(group_ns.iter())
            .map(
                |(&sum, &ni)| {
                    if ni > 0 {
                        sum.powi(2) / ni as f64
                    } else {
                        0.0
                    }
                },
            )
            .sum::<f64>();

        let h = (12.0 / (n_total * (n_total + 1.0))) * h - 3.0 * (n_total + 1.0);

        let dof = groups.len() - 1;
        let p_value = crate::stats::categorical::CategoricalStats::chi_square_sf(h.max(0.0), dof);

        let eta_squared = if n_total > 1.0 {
            (h - dof as f64 + 1.0) / (n_total - dof as f64)
        } else {
            0.0
        };

        Some(KruskalWallisResult {
            numeric_col: num_col.to_string(),
            group_col: cat_col.to_string(),
            h_statistic: h.max(0.0),
            p_value,
            eta_squared: eta_squared.clamp(0.0, 1.0),
        })
    }

    /// Grubbs' test for a single outlier per numeric column.
    fn grubbs_tests(&self) -> Vec<GrubbsTestResult> {
        let num_cols = self.schema.numeric_columns();

        num_cols
            .iter()
            .filter_map(|&col_name| {
                let col = self.df.column(col_name).ok()?;
                let values = DescriptiveStats::column_to_f64_vec(col)?;
                let n = values.len();
                if n < 3 {
                    return None;
                }

                let nf = n as f64;
                let mean = values.iter().sum::<f64>() / nf;
                let std =
                    (values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (nf - 1.0)).sqrt();

                if std < f64::EPSILON {
                    return None;
                }

                // Find the value furthest from the mean
                let (outlier_idx, _) = values.iter().enumerate().max_by(|(_, a), (_, b)| {
                    (*a - mean)
                        .abs()
                        .partial_cmp(&(*b - mean).abs())
                        .unwrap_or(std::cmp::Ordering::Equal)
                })?;

                let outlier_value = values[outlier_idx];
                let g = (outlier_value - mean).abs() / std;

                // Critical value using t-distribution
                let alpha = 0.05;
                let t_crit = StudentsT::new(0.0, 1.0, nf - 2.0)
                    .ok()
                    .map(|t| t.inverse_cdf(1.0 - alpha / (2.0 * nf)))
                    .unwrap_or(2.0);

                let critical_value = ((nf - 1.0) / nf.sqrt())
                    * (t_crit.powi(2) / (nf - 2.0 + t_crit.powi(2))).sqrt();

                Some(GrubbsTestResult {
                    column: col_name.to_string(),
                    statistic: g,
                    critical_value,
                    outlier_value,
                    is_outlier: g > critical_value,
                })
            })
            .collect()
    }

    // ── Helpers ─────────────────────────────────────────────────

    fn median_of(vals: &[f64]) -> f64 {
        let mut sorted = vals.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let mid = sorted.len() / 2;
        if sorted.len() % 2 == 0 {
            (sorted[mid - 1] + sorted[mid]) / 2.0
        } else {
            sorted[mid]
        }
    }

    /// F-distribution survival function approximation.
    fn f_distribution_sf(f: f64, d1: f64, d2: f64) -> f64 {
        if f <= 0.0 || d1 <= 0.0 || d2 <= 0.0 {
            return 1.0;
        }
        let x = d2 / (d2 + d1 * f);
        Self::incomplete_beta(d2 / 2.0, d1 / 2.0, x)
    }

    /// Regularized incomplete beta function approximation (simple continued fraction).
    fn incomplete_beta(a: f64, b: f64, x: f64) -> f64 {
        if x <= 0.0 {
            return 0.0;
        }
        if x >= 1.0 {
            return 1.0;
        }
        // Use series expansion for small x
        let mut sum = 1.0f64;
        let mut term = 1.0f64;
        for n in 1..200 {
            let nf = n as f64;
            term *= x * (a + b + nf - 1.0) * (a + nf - 1.0)
                / ((a + 2.0 * nf - 1.0) * (a + 2.0 * nf) * nf);
            term *= (a + 2.0 * nf) * nf / ((a + nf) * (b - nf).max(0.001));
            if term.abs() < 1e-12 {
                break;
            }
            sum += term;
        }
        (x.powf(a) * (1.0 - x).powf(b) * sum / a).clamp(0.0, 1.0)
    }
}
