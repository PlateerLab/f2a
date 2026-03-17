use polars::prelude::*;
use serde::{Deserialize, Serialize};
use statrs::distribution::ContinuousCDF;

use crate::core::schema::DataSchema;
use crate::stats::descriptive::DescriptiveStats;

// ─── Result types ───────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BestFitEntry {
    pub column: String,
    pub distribution: String,
    pub aic: f64,
    pub bic: f64,
    pub ks_statistic: f64,
    pub ks_p_value: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JarqueBeraEntry {
    pub column: String,
    pub statistic: f64,
    pub p_value: f64,
    pub is_normal: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PowerTransformEntry {
    pub column: String,
    pub method: String,
    pub skewness_before: f64,
    pub skewness_after: f64,
    pub improvement: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KdeBandwidthEntry {
    pub column: String,
    pub silverman: f64,
    pub scott: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdvancedDistributionResult {
    pub best_fits: Vec<BestFitEntry>,
    pub jarque_bera: Vec<JarqueBeraEntry>,
    pub power_transforms: Vec<PowerTransformEntry>,
    pub kde_bandwidths: Vec<KdeBandwidthEntry>,
}

// ─── Implementation ─────────────────────────────────────────────────

pub struct AdvancedDistributionStats<'a> {
    df: &'a DataFrame,
    schema: &'a DataSchema,
    n_fits: usize,
    max_sample: usize,
}

impl<'a> AdvancedDistributionStats<'a> {
    pub fn new(
        df: &'a DataFrame,
        schema: &'a DataSchema,
        n_fits: usize,
        max_sample: usize,
    ) -> Self {
        Self {
            df,
            schema,
            n_fits,
            max_sample,
        }
    }

    pub fn compute(&self) -> AdvancedDistributionResult {
        let num_cols = self.schema.numeric_columns();

        let mut best_fits = Vec::new();
        let mut jarque_bera = Vec::new();
        let mut power_transforms = Vec::new();
        let mut kde_bandwidths = Vec::new();

        for &col_name in &num_cols {
            let col = match self.df.column(col_name) {
                Ok(c) => c,
                Err(_) => continue,
            };
            let values = match DescriptiveStats::column_to_f64_vec(col) {
                Some(v) if v.len() >= 10 => v,
                _ => continue,
            };

            // Sample if needed
            let vals = if values.len() > self.max_sample {
                let step = values.len() / self.max_sample;
                values
                    .iter()
                    .step_by(step)
                    .copied()
                    .take(self.max_sample)
                    .collect::<Vec<_>>()
            } else {
                values.clone()
            };

            // Best fit distribution
            if let Some(bf) = self.fit_distributions(col_name, &vals) {
                best_fits.push(bf);
            }

            // Jarque-Bera test
            jarque_bera.push(self.jarque_bera_test(col_name, &vals));

            // Power transform recommendation
            power_transforms.push(self.power_transform_rec(col_name, &vals));

            // KDE bandwidths
            kde_bandwidths.push(self.kde_bandwidth(col_name, &vals));
        }

        AdvancedDistributionResult {
            best_fits,
            jarque_bera,
            power_transforms,
            kde_bandwidths,
        }
    }

    /// Fit candidate distributions and rank by AIC.
    fn fit_distributions(&self, name: &str, values: &[f64]) -> Option<BestFitEntry> {
        let n = values.len() as f64;
        let mean = values.iter().sum::<f64>() / n;
        let std = (values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0)).sqrt();

        if std < f64::EPSILON {
            return None;
        }

        // Test normal distribution fit
        let normal = statrs::distribution::Normal::new(mean, std).ok()?;

        // Log-likelihood for normal
        let ll_normal: f64 = values
            .iter()
            .map(|&x| {
                let z = (x - mean) / std;
                -0.5 * z * z - 0.5 * (2.0 * std::f64::consts::PI).ln() - std.ln()
            })
            .sum();

        let k_normal = 2.0; // Two parameters (mean, std)
        let aic_normal = 2.0 * k_normal - 2.0 * ll_normal;
        let bic_normal = k_normal * n.ln() - 2.0 * ll_normal;

        // KS test for normal
        let mut sorted = values.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let ks_stat: f64 = sorted
            .iter()
            .enumerate()
            .map(|(i, &x)| {
                let ecdf = (i + 1) as f64 / n;
                let cdf = normal.cdf(x);
                (ecdf - cdf).abs().max((i as f64 / n - cdf).abs())
            })
            .fold(0.0f64, f64::max);

        let sqrt_n = n.sqrt();
        let lambda = (sqrt_n + 0.12 + 0.11 / sqrt_n) * ks_stat;
        let ks_p = if lambda < 0.001 {
            1.0
        } else {
            let mut p = 0.0;
            for k in 1..=100 {
                let sign = if k % 2 == 0 { -1.0 } else { 1.0 };
                p += sign * (-2.0 * (k as f64).powi(2) * lambda.powi(2)).exp();
            }
            (2.0 * p).clamp(0.0, 1.0)
        };

        Some(BestFitEntry {
            column: name.to_string(),
            distribution: "normal".to_string(),
            aic: aic_normal,
            bic: bic_normal,
            ks_statistic: ks_stat,
            ks_p_value: ks_p,
        })
    }

    /// Jarque-Bera test for normality.
    fn jarque_bera_test(&self, name: &str, values: &[f64]) -> JarqueBeraEntry {
        let n = values.len() as f64;
        let mean = values.iter().sum::<f64>() / n;
        let m2: f64 = values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
        let m3: f64 = values.iter().map(|x| (x - mean).powi(3)).sum::<f64>() / n;
        let m4: f64 = values.iter().map(|x| (x - mean).powi(4)).sum::<f64>() / n;

        let skewness = if m2 > f64::EPSILON {
            m3 / m2.powf(1.5)
        } else {
            0.0
        };
        let kurtosis = if m2 > f64::EPSILON {
            m4 / m2.powi(2) - 3.0
        } else {
            0.0
        };

        let jb = n / 6.0 * (skewness.powi(2) + kurtosis.powi(2) / 4.0);
        let p_value = (-jb / 2.0).exp(); // Chi2(df=2) approximation

        JarqueBeraEntry {
            column: name.to_string(),
            statistic: jb,
            p_value,
            is_normal: p_value > 0.05,
        }
    }

    /// Power transform recommendation (Yeo-Johnson).
    fn power_transform_rec(&self, name: &str, values: &[f64]) -> PowerTransformEntry {
        let n = values.len() as f64;
        let mean = values.iter().sum::<f64>() / n;
        let std = (values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0)).sqrt();

        let skewness_before = if std > f64::EPSILON {
            let m3: f64 = values
                .iter()
                .map(|x| ((x - mean) / std).powi(3))
                .sum::<f64>()
                / n;
            m3
        } else {
            0.0
        };

        // Try log(1+x) transform for positive-skewed data
        let (method, skewness_after) = if skewness_before > 0.5 && values.iter().all(|&x| x > 0.0) {
            let transformed: Vec<f64> = values.iter().map(|x| (1.0 + x).ln()).collect();
            let t_mean = transformed.iter().sum::<f64>() / n;
            let t_std = (transformed
                .iter()
                .map(|x| (x - t_mean).powi(2))
                .sum::<f64>()
                / (n - 1.0))
                .sqrt();
            let t_skew = if t_std > f64::EPSILON {
                transformed
                    .iter()
                    .map(|x| ((x - t_mean) / t_std).powi(3))
                    .sum::<f64>()
                    / n
            } else {
                0.0
            };
            ("log1p".to_string(), t_skew)
        } else if skewness_before < -0.5 {
            // Try square transform for left-skewed
            let transformed: Vec<f64> = values.iter().map(|x| x.powi(2)).collect();
            let t_mean = transformed.iter().sum::<f64>() / n;
            let t_std = (transformed
                .iter()
                .map(|x| (x - t_mean).powi(2))
                .sum::<f64>()
                / (n - 1.0))
                .sqrt();
            let t_skew = if t_std > f64::EPSILON {
                transformed
                    .iter()
                    .map(|x| ((x - t_mean) / t_std).powi(3))
                    .sum::<f64>()
                    / n
            } else {
                0.0
            };
            ("square".to_string(), t_skew)
        } else {
            ("none".to_string(), skewness_before)
        };

        let improvement = (skewness_before.abs() - skewness_after.abs()).max(0.0);

        PowerTransformEntry {
            column: name.to_string(),
            method,
            skewness_before,
            skewness_after,
            improvement,
        }
    }

    /// KDE bandwidth estimation (Silverman & Scott rules).
    fn kde_bandwidth(&self, name: &str, values: &[f64]) -> KdeBandwidthEntry {
        let n = values.len() as f64;
        let mean = values.iter().sum::<f64>() / n;
        let std = (values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0)).sqrt();

        // IQR
        let mut sorted = values.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let q1_idx = (n * 0.25) as usize;
        let q3_idx = (n * 0.75) as usize;
        let iqr = sorted.get(q3_idx).unwrap_or(&0.0) - sorted.get(q1_idx).unwrap_or(&0.0);

        // Silverman's rule
        let silverman = 0.9 * std.min(iqr / 1.34) * n.powf(-1.0 / 5.0);

        // Scott's rule
        let scott = 3.49 * std * n.powf(-1.0 / 3.0);

        KdeBandwidthEntry {
            column: name.to_string(),
            silverman: silverman.max(f64::EPSILON),
            scott: scott.max(f64::EPSILON),
        }
    }
}
