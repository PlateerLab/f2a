use indexmap::IndexMap;
use polars::prelude::*;
use serde::{Deserialize, Serialize};

use crate::core::schema::DataSchema;

// ─── Result types ───────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CategoryFrequency {
    pub value: String,
    pub count: usize,
    pub ratio: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CategoricalColumnResult {
    pub column: String,
    pub n_unique: usize,
    pub entropy: f64,
    pub normalized_entropy: f64,
    pub frequencies: Vec<CategoryFrequency>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChiSquareEntry {
    pub col_a: String,
    pub col_b: String,
    pub chi2: f64,
    pub p_value: f64,
    pub cramers_v: f64,
    pub dof: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CategoricalResult {
    pub columns: Vec<CategoricalColumnResult>,
    pub chi_square_tests: Vec<ChiSquareEntry>,
}

// ─── Implementation ─────────────────────────────────────────────────

pub struct CategoricalStats<'a> {
    df: &'a DataFrame,
    schema: &'a DataSchema,
    max_categories: usize,
}

impl<'a> CategoricalStats<'a> {
    pub fn new(df: &'a DataFrame, schema: &'a DataSchema, max_categories: usize) -> Self {
        Self {
            df,
            schema,
            max_categories,
        }
    }

    pub fn compute(&self) -> CategoricalResult {
        let cat_cols = self.schema.categorical_columns();
        let n_rows = self.df.height();

        // ── Per-column analysis ─────────────────────────────────
        let columns: Vec<CategoricalColumnResult> = cat_cols
            .iter()
            .filter_map(|&col_name| {
                let col = self.df.column(col_name).ok()?;
                let str_col = col.cast(&DataType::String).ok()?;
                let ca = str_col.str().ok()?;
                Some(self.analyse_column(col_name, ca, n_rows))
            })
            .collect();

        // ── Chi-square independence tests ───────────────────────
        let chi_square_tests = self.chi_square_independence(&cat_cols, n_rows);

        CategoricalResult {
            columns,
            chi_square_tests,
        }
    }

    fn analyse_column(
        &self,
        name: &str,
        ca: &StringChunked,
        n_rows: usize,
    ) -> CategoricalColumnResult {
        // Count frequencies
        let mut freq_map: IndexMap<String, usize> = IndexMap::new();
        for opt_val in ca.into_iter() {
            let key = opt_val.unwrap_or("(missing)").to_string();
            *freq_map.entry(key).or_insert(0) += 1;
        }

        let n_unique = freq_map.len();

        // Sort by frequency descending
        let mut sorted: Vec<(String, usize)> = freq_map.into_iter().collect();
        sorted.sort_by(|a, b| b.1.cmp(&a.1));

        // Group low-frequency categories into "Other" if exceeding max_categories
        let frequencies = if sorted.len() > self.max_categories {
            let mut result: Vec<CategoryFrequency> = sorted[..self.max_categories]
                .iter()
                .map(|(val, count)| CategoryFrequency {
                    value: val.clone(),
                    count: *count,
                    ratio: *count as f64 / n_rows as f64,
                })
                .collect();

            let other_count: usize = sorted[self.max_categories..].iter().map(|(_, c)| c).sum();
            result.push(CategoryFrequency {
                value: "(Other)".to_string(),
                count: other_count,
                ratio: other_count as f64 / n_rows as f64,
            });
            result
        } else {
            sorted
                .iter()
                .map(|(val, count)| CategoryFrequency {
                    value: val.clone(),
                    count: *count,
                    ratio: *count as f64 / n_rows as f64,
                })
                .collect()
        };

        // Shannon entropy
        let total = n_rows as f64;
        let entropy: f64 = frequencies
            .iter()
            .filter(|f| f.count > 0)
            .map(|f| {
                let p = f.count as f64 / total;
                if p > 0.0 {
                    -p * p.ln()
                } else {
                    0.0
                }
            })
            .sum();

        let max_entropy = if n_unique > 1 {
            (n_unique as f64).ln()
        } else {
            1.0
        };
        let normalized_entropy = if max_entropy > f64::EPSILON {
            entropy / max_entropy
        } else {
            0.0
        };

        CategoricalColumnResult {
            column: name.to_string(),
            n_unique,
            entropy,
            normalized_entropy,
            frequencies,
        }
    }

    /// Chi-square test of independence between all pairs of categorical columns.
    fn chi_square_independence(&self, cat_cols: &[&str], n_rows: usize) -> Vec<ChiSquareEntry> {
        let n = n_rows as f64;
        if n < 1.0 || cat_cols.len() < 2 {
            return vec![];
        }

        let mut results = Vec::new();

        for i in 0..cat_cols.len() {
            for j in (i + 1)..cat_cols.len() {
                if let Some(entry) = self.chi_square_pair(cat_cols[i], cat_cols[j], n) {
                    results.push(entry);
                }
            }
        }

        results
    }

    fn chi_square_pair(&self, col_a: &str, col_b: &str, n: f64) -> Option<ChiSquareEntry> {
        let a = self.df.column(col_a).ok()?.cast(&DataType::String).ok()?;
        let b = self.df.column(col_b).ok()?.cast(&DataType::String).ok()?;
        let a_str = a.str().ok()?;
        let b_str = b.str().ok()?;

        let mut contingency: IndexMap<(String, String), usize> = IndexMap::new();
        let mut a_counts: IndexMap<String, usize> = IndexMap::new();
        let mut b_counts: IndexMap<String, usize> = IndexMap::new();

        for (va, vb) in a_str.into_iter().zip(b_str.into_iter()) {
            let ka = va.unwrap_or("(NA)").to_string();
            let kb = vb.unwrap_or("(NA)").to_string();
            *contingency.entry((ka.clone(), kb.clone())).or_insert(0) += 1;
            *a_counts.entry(ka).or_insert(0) += 1;
            *b_counts.entry(kb).or_insert(0) += 1;
        }

        let r = a_counts.len();
        let k = b_counts.len();
        if r < 2 || k < 2 {
            return None;
        }

        let dof = (r - 1) * (k - 1);

        // Chi-square statistic
        let mut chi2 = 0.0f64;
        for ((ka, kb), &observed) in &contingency {
            let ea = *a_counts.get(ka).unwrap_or(&0) as f64;
            let eb = *b_counts.get(kb).unwrap_or(&0) as f64;
            let expected = ea * eb / n;
            if expected > 0.0 {
                chi2 += (observed as f64 - expected).powi(2) / expected;
            }
        }

        // P-value approximation using chi-square survival function
        let p_value = Self::chi_square_sf(chi2, dof);

        let min_dim = (r - 1).min(k - 1) as f64;
        let cramers_v = if min_dim >= 1.0 && n > 0.0 {
            (chi2 / (n * min_dim)).sqrt().clamp(0.0, 1.0)
        } else {
            0.0
        };

        Some(ChiSquareEntry {
            col_a: col_a.to_string(),
            col_b: col_b.to_string(),
            chi2,
            p_value,
            cramers_v,
            dof,
        })
    }

    /// Chi-square survival function approximation using Wilson-Hilferty.
    pub(crate) fn chi_square_sf(x: f64, dof: usize) -> f64 {
        if dof == 0 {
            return 1.0;
        }
        let k = dof as f64;
        // Wilson-Hilferty approximation
        let z = ((x / k).powf(1.0 / 3.0) - (1.0 - 2.0 / (9.0 * k))) / (2.0 / (9.0 * k)).sqrt();
        // Standard normal survival
        0.5 * statrs::function::erf::erfc(z / std::f64::consts::SQRT_2)
    }
}
