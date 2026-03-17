use polars::prelude::*;
use serde::{Deserialize, Serialize};

use crate::core::schema::DataSchema;
use crate::utils::types::ColumnType;

// ─── Types ──────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum ColumnRole {
    Target,
    Feature,
    Id,
    Datetime,
    Text,
    Constant,
    HighMissing,
    Unknown,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ColumnRoleEntry {
    pub column: String,
    pub role: ColumnRole,
    pub confidence: f64,
    pub reason: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ColumnRoleResult {
    pub roles: Vec<ColumnRoleEntry>,
    pub target_candidates: Vec<String>,
    pub id_columns: Vec<String>,
    pub droppable_columns: Vec<String>,
}

// ─── Classifier ─────────────────────────────────────────────────────

pub struct ColumnRoleClassifier<'a> {
    df: &'a DataFrame,
    schema: &'a DataSchema,
    target_hint: Option<String>,
}

impl<'a> ColumnRoleClassifier<'a> {
    pub fn new(df: &'a DataFrame, schema: &'a DataSchema, target_hint: Option<String>) -> Self {
        Self {
            df,
            schema,
            target_hint,
        }
    }

    pub fn compute(&self) -> ColumnRoleResult {
        let n = self.schema.n_rows.max(1);
        let mut roles: Vec<ColumnRoleEntry> = Vec::new();

        for col_info in &self.schema.columns {
            let (role, confidence, reason) = self.classify_column(col_info, n);
            roles.push(ColumnRoleEntry {
                column: col_info.name.clone(),
                role,
                confidence,
                reason,
            });
        }

        let target_candidates: Vec<String> = roles
            .iter()
            .filter(|r| r.role == ColumnRole::Target)
            .map(|r| r.column.clone())
            .collect();

        let id_columns: Vec<String> = roles
            .iter()
            .filter(|r| r.role == ColumnRole::Id)
            .map(|r| r.column.clone())
            .collect();

        let droppable_columns: Vec<String> = roles
            .iter()
            .filter(|r| {
                matches!(
                    r.role,
                    ColumnRole::Id | ColumnRole::Constant | ColumnRole::HighMissing
                )
            })
            .map(|r| r.column.clone())
            .collect();

        ColumnRoleResult {
            roles,
            target_candidates,
            id_columns,
            droppable_columns,
        }
    }

    fn classify_column(
        &self,
        col_info: &crate::core::schema::ColumnInfo,
        n_rows: usize,
    ) -> (ColumnRole, f64, String) {
        let name_lower = col_info.name.to_lowercase();

        // ── Explicit target_hint ────────────────────────────
        if let Some(ref hint) = self.target_hint {
            if col_info.name == *hint {
                return (
                    ColumnRole::Target,
                    1.0,
                    "Explicitly specified as target".into(),
                );
            }
        }

        // ── High-missing column (>70%) ──────────────────────
        if col_info.missing_ratio > 0.70 {
            return (
                ColumnRole::HighMissing,
                0.95,
                format!(
                    "{:.1}% missing — likely unusable",
                    col_info.missing_ratio * 100.0
                ),
            );
        }

        // ── Constant column ─────────────────────────────────
        if col_info.n_unique <= 1 {
            return (ColumnRole::Constant, 1.0, "Only 1 unique value".into());
        }

        // ── Datetime column ─────────────────────────────────
        if col_info.inferred_type == ColumnType::DateTime {
            return (ColumnRole::Datetime, 0.95, "Datetime type detected".into());
        }

        // ── ID-like column heuristics ───────────────────────
        let uniqueness_ratio = col_info.n_unique as f64 / n_rows as f64;
        let is_name_id = name_lower.contains("id")
            || name_lower == "index"
            || name_lower.ends_with("_key")
            || name_lower.ends_with("_no")
            || name_lower.ends_with("_num")
            || name_lower == "pk"
            || name_lower == "uid";

        if is_name_id && uniqueness_ratio > 0.95 && col_info.n_missing == 0 {
            return (
                ColumnRole::Id,
                0.90,
                "Name pattern + high uniqueness".into(),
            );
        }
        if uniqueness_ratio > 0.999 && col_info.n_missing == 0 {
            return (
                ColumnRole::Id,
                0.80,
                "Near-unique without missing values".into(),
            );
        }

        // ── Text column ─────────────────────────────────────
        if col_info.inferred_type == ColumnType::Text {
            return (ColumnRole::Text, 0.85, "Inferred as free text".into());
        }

        // ── Target candidate heuristics (common column names) ──
        let target_names = [
            "target", "label", "class", "y", "output", "response", "is_fraud", "churn", "survived",
            "income", "price", "sale", "revenue", "default",
        ];
        for t in &target_names {
            if name_lower == *t || name_lower.ends_with(&format!("_{}", t)) {
                return (
                    ColumnRole::Target,
                    0.60,
                    format!("Column name matches common target pattern '{}'", t),
                );
            }
        }

        // ── Binary column with target-like name ─────────────
        if col_info.n_unique == 2 {
            // Binary columns are often targets in classification
            if name_lower.starts_with("is_") || name_lower.starts_with("has_") {
                return (
                    ColumnRole::Target,
                    0.45,
                    "Binary column with boolean-like name".into(),
                );
            }
        }

        // ── Default: Feature ────────────────────────────────
        (
            ColumnRole::Feature,
            0.50,
            "Default classification as feature".into(),
        )
    }
}
