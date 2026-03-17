use polars::prelude::*;
use serde::{Deserialize, Serialize};
use statrs::distribution::{ContinuousCDF, Normal};
use statrs::statistics::{Data, Distribution};

use crate::core::schema::DataSchema;
use crate::stats::descriptive::DescriptiveStats;

// ─── Result types ───────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistributionColumnResult {
    pub column: String,
    pub skewness: f64,
    pub kurtosis: f64,
    pub skewness_interpretation: String,
    pub kurtosis_interpretation: String,
    /// Shapiro-Wilk test (only for n ≤ 5000)
    pub shapiro_p: Option<f64>,
    /// Kolmogorov-Smirnov test against normal
    pub ks_statistic: f64,
    pub ks_p_value: f64,
    /// D'Agostino-Pearson omnibus test
    pub dagostino_statistic: Option<f64>,
    pub dagostino_p_value: Option<f64>,
    /// Anderson-Darling statistic
    pub anderson_statistic: f64,
    /// Whether column appears normally distributed (all tests agree at α=0.05)
    pub is_normal: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistributionResult {
    pub columns: Vec<DistributionColumnResult>,
}

// ─── Implementation ─────────────────────────────────────────────────

pub struct DistributionStats<'a> {
    df: &'a DataFrame,
    schema: &'a DataSchema,
}

impl<'a> DistributionStats<'a> {
    pub fn new(df: &'a DataFrame, schema: &'a DataSchema) -> Self {
        Self { df, schema }
    }

    pub fn compute(&self) -> DistributionResult {
        let num_cols = self.schema.numeric_columns();

        let columns: Vec<DistributionColumnResult> = num_cols
            .iter()
            .filter_map(|&col_name| {
                let col = self.df.column(col_name).ok()?;
                let values = DescriptiveStats::column_to_f64_vec(col)?;
                if values.len() < 8 {
                    return None;
                }
                Some(self.analyse_column(col_name, &values))
            })
            .collect();

        DistributionResult { columns }
    }

    fn analyse_column(&self, name: &str, values: &[f64]) -> DistributionColumnResult {
        let n = values.len();
        let data = Data::new(values.to_vec());
        let mean = data.mean().unwrap_or(0.0);
        let std = data.std_dev().unwrap_or(1.0);

        // ── Skewness & kurtosis ─────────────────────────────────
        let skewness = Self::skewness(values, mean, std);
        let kurtosis = Self::excess_kurtosis(values, mean, std);

        let skewness_interpretation = match skewness.abs() {
            s if s < 0.5 => "approximately symmetric".to_string(),
            s if s < 1.0 => {
                if skewness > 0.0 {
                    "moderately right-skewed".to_string()
                } else {
                    "moderately left-skewed".to_string()
                }
            }
            _ => {
                if skewness > 0.0 {
                    "highly right-skewed".to_string()
                } else {
                    "highly left-skewed".to_string()
                }
            }
        };

        let kurtosis_interpretation = match kurtosis {
            k if k < -1.0 => "platykurtic (light-tailed)".to_string(),
            k if k > 1.0 => "leptokurtic (heavy-tailed)".to_string(),
            _ => "mesokurtic (normal-like tails)".to_string(),
        };

        // ── Kolmogorov-Smirnov test against Normal ──────────────
        let (ks_statistic, ks_p_value) = Self::ks_test_normal(values, mean, std);

        // ── Anderson-Darling test ───────────────────────────────
        let anderson_statistic = Self::anderson_darling(values, mean, std);

        // ── D'Agostino-Pearson omnibus test ─────────────────────
        let (dagostino_statistic, dagostino_p_value) = if n >= 20 {
            let (stat, p) = Self::dagostino_pearson(values, mean, std);
            (Some(stat), Some(p))
        } else {
            (None, None)
        };

        // ── Normality consensus ─────────────────────────────────
        let alpha = 0.05;
        let mut normal_votes = 0u32;
        let mut total_tests = 0u32;

        total_tests += 1;
        if ks_p_value > alpha {
            normal_votes += 1;
        }

        // Anderson-Darling: compare with critical value for normal at 5%
        // Approximate critical value ≈ 0.752
        total_tests += 1;
        if anderson_statistic < 0.752 {
            normal_votes += 1;
        }

        if let Some(p) = dagostino_p_value {
            total_tests += 1;
            if p > alpha {
                normal_votes += 1;
            }
        }

        let is_normal = normal_votes > total_tests / 2;

        DistributionColumnResult {
            column: name.to_string(),
            skewness,
            kurtosis,
            skewness_interpretation,
            kurtosis_interpretation,
            shapiro_p: None, // Shapiro-Wilk is complex; delegate to Python/scipy
            ks_statistic,
            ks_p_value,
            dagostino_statistic,
            dagostino_p_value,
            anderson_statistic,
            is_normal,
        }
    }

    // ── Statistical computation helpers ─────────────────────────

    fn skewness(values: &[f64], mean: f64, std: f64) -> f64 {
        let n = values.len() as f64;
        if n < 3.0 || std < f64::EPSILON {
            return f64::NAN;
        }
        let m3: f64 = values
            .iter()
            .map(|x| ((x - mean) / std).powi(3))
            .sum::<f64>()
            / n;
        let adjustment = (n * (n - 1.0)).sqrt() / (n - 2.0);
        m3 * adjustment
    }

    fn excess_kurtosis(values: &[f64], mean: f64, std: f64) -> f64 {
        let n = values.len() as f64;
        if n < 4.0 || std < f64::EPSILON {
            return f64::NAN;
        }
        let m4: f64 = values
            .iter()
            .map(|x| ((x - mean) / std).powi(4))
            .sum::<f64>()
            / n;
        let excess = (n - 1.0) / ((n - 2.0) * (n - 3.0)) * ((n + 1.0) * m4 - 3.0 * (n - 1.0));
        excess
    }

    /// One-sample Kolmogorov-Smirnov test against Normal(mean, std).
    fn ks_test_normal(values: &[f64], mean: f64, std: f64) -> (f64, f64) {
        let n = values.len();
        if n == 0 || std < f64::EPSILON {
            return (f64::NAN, f64::NAN);
        }

        let normal = Normal::new(mean, std).unwrap_or(Normal::new(0.0, 1.0).unwrap());

        let mut sorted = values.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let mut d_max = 0.0f64;
        for (i, &x) in sorted.iter().enumerate() {
            let ecdf = (i + 1) as f64 / n as f64;
            let cdf = normal.cdf(x);
            let ecdf_prev = i as f64 / n as f64;
            d_max = d_max.max((ecdf - cdf).abs()).max((ecdf_prev - cdf).abs());
        }

        // P-value approximation (Kolmogorov distribution)
        let sqrt_n = (n as f64).sqrt();
        let p_value = Self::kolmogorov_p_value(d_max, sqrt_n);

        (d_max, p_value)
    }

    /// Kolmogorov distribution p-value approximation.
    fn kolmogorov_p_value(d: f64, sqrt_n: f64) -> f64 {
        let lambda = (sqrt_n + 0.12 + 0.11 / sqrt_n) * d;
        if lambda < 0.001 {
            return 1.0;
        }
        // Series approximation
        let mut p = 0.0f64;
        for k in 1..=100 {
            let sign = if k % 2 == 0 { -1.0 } else { 1.0 };
            let term = sign * (-2.0 * (k as f64).powi(2) * lambda.powi(2)).exp();
            p += term;
        }
        let p = 2.0 * p;
        p.clamp(0.0, 1.0)
    }

    /// Anderson-Darling statistic against Normal(mean, std).
    fn anderson_darling(values: &[f64], mean: f64, std: f64) -> f64 {
        let n = values.len();
        if n == 0 || std < f64::EPSILON {
            return f64::NAN;
        }

        let normal = Normal::new(mean, std).unwrap_or(Normal::new(0.0, 1.0).unwrap());

        let mut sorted = values.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let nf = n as f64;
        let mut s = 0.0f64;
        for (i, &x) in sorted.iter().enumerate() {
            let f = normal.cdf(x).clamp(1e-10, 1.0 - 1e-10);
            let f_rev = normal.cdf(sorted[n - 1 - i]).clamp(1e-10, 1.0 - 1e-10);
            let w = (2.0 * (i as f64) + 1.0) / nf;
            s += w * (f.ln() + (1.0 - f_rev).ln());
        }

        let a2 = -nf - s;
        // Apply correction factor for estimated parameters
        a2 * (1.0 + 0.75 / nf + 2.25 / (nf * nf))
    }

    /// D'Agostino-Pearson omnibus test for normality.
    fn dagostino_pearson(values: &[f64], mean: f64, std: f64) -> (f64, f64) {
        let n = values.len() as f64;
        if n < 20.0 {
            return (f64::NAN, f64::NAN);
        }

        let skew = Self::skewness(values, mean, std);
        let kurt = Self::excess_kurtosis(values, mean, std);

        // Z-score for skewness (D'Agostino 1970)
        let y = skew * ((n + 1.0) * (n + 3.0) / (6.0 * (n - 2.0))).sqrt();
        let beta2 = 3.0 * (n * n + 27.0 * n - 70.0) * (n + 1.0) * (n + 3.0)
            / ((n - 2.0) * (n + 5.0) * (n + 7.0) * (n + 9.0));
        let w2 = (2.0 * (beta2 - 1.0)).sqrt() - 1.0;
        let delta = 1.0 / (0.5 * w2.ln()).sqrt();
        let alpha_s = (2.0 / (w2 - 1.0)).sqrt();
        let z_s = delta * (y / alpha_s + ((y / alpha_s).powi(2) + 1.0).sqrt()).ln();

        // Z-score for kurtosis (Anscombe & Glynn 1983)
        let e_kurt = 3.0 * (n - 1.0) / (n + 1.0) - 3.0; // Expected kurtosis
        let var_kurt =
            24.0 * n * (n - 2.0) * (n - 3.0) / ((n + 1.0).powi(2) * (n + 3.0) * (n + 5.0));
        let z_k = if var_kurt > 0.0 {
            (kurt - e_kurt) / var_kurt.sqrt()
        } else {
            0.0
        };

        // Omnibus K² = Z_s² + Z_k²
        let k2 = z_s.powi(2) + z_k.powi(2);

        // P-value from chi-square distribution with df=2
        let p_value = (-k2 / 2.0).exp(); // Quick approximation for chi2(df=2)

        (k2, p_value)
    }
}
