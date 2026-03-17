use thiserror::Error;

/// Top-level error type for the f2a library.
#[derive(Error, Debug)]
pub enum F2aError {
    #[error("Unsupported file format: {0}")]
    UnsupportedFormat(String),

    #[error("Failed to load data: {0}")]
    DataLoadError(String),

    #[error("Empty dataset – no rows or no columns")]
    EmptyData,

    #[error("Column not found: {0}")]
    ColumnNotFound(String),

    #[error("Computation error: {0}")]
    ComputationError(String),

    #[error("Configuration error: {0}")]
    ConfigError(String),

    #[error("Polars error: {0}")]
    PolarsError(#[from] polars::prelude::PolarsError),

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("JSON error: {0}")]
    JsonError(#[from] serde_json::Error),
}

/// Convenience alias.
pub type F2aResult<T> = Result<T, F2aError>;

impl F2aError {
    pub fn computation(msg: impl Into<String>) -> Self {
        F2aError::ComputationError(msg.into())
    }
}

// Convert to PyO3 error
impl From<F2aError> for pyo3::PyErr {
    fn from(err: F2aError) -> pyo3::PyErr {
        pyo3::exceptions::PyRuntimeError::new_err(err.to_string())
    }
}
