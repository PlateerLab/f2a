use serde::{Deserialize, Serialize};

/// Semantic column type – mirrors the Python `ColumnType` enum.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ColumnType {
    Numeric,
    Categorical,
    Text,
    DateTime,
    Boolean,
}

impl ColumnType {
    pub fn as_str(&self) -> &'static str {
        match self {
            ColumnType::Numeric => "numeric",
            ColumnType::Categorical => "categorical",
            ColumnType::Text => "text",
            ColumnType::DateTime => "datetime",
            ColumnType::Boolean => "boolean",
        }
    }
}

impl std::fmt::Display for ColumnType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Infer the semantic `ColumnType` from a Polars dtype and column statistics.
///
/// Heuristics:
/// - Boolean → `Boolean`
/// - Numeric with ≤10 unique values (when n_rows > 100) → `Categorical`
/// - String with high unique ratio and long avg length → `Text`
/// - String / low-cardinality string → `Categorical`
/// - Date/Time/Datetime/Duration → `DateTime`
/// - Everything else numeric → `Numeric`
pub fn infer_column_type(
    dtype: &polars::prelude::DataType,
    n_unique: usize,
    n_rows: usize,
    avg_str_len: Option<f64>,
) -> ColumnType {
    use polars::prelude::DataType;

    match dtype {
        DataType::Boolean => ColumnType::Boolean,

        DataType::Date | DataType::Time | DataType::Datetime(_, _) | DataType::Duration(_) => {
            ColumnType::DateTime
        }

        DataType::Int8
        | DataType::Int16
        | DataType::Int32
        | DataType::Int64
        | DataType::UInt8
        | DataType::UInt16
        | DataType::UInt32
        | DataType::UInt64
        | DataType::Float32
        | DataType::Float64 => {
            // Numeric with very low cardinality on a large dataset → treat as categorical
            if n_rows > 100 && n_unique <= 10 {
                ColumnType::Categorical
            } else {
                ColumnType::Numeric
            }
        }

        DataType::String => {
            let unique_ratio = if n_rows > 0 {
                n_unique as f64 / n_rows as f64
            } else {
                0.0
            };
            let avg_len = avg_str_len.unwrap_or(0.0);

            // High-cardinality + long strings → text
            if unique_ratio > 0.5 && avg_len > 50.0 {
                ColumnType::Text
            } else {
                ColumnType::Categorical
            }
        }

        _ => ColumnType::Categorical,
    }
}
