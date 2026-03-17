//! PyO3 module entry-point for `f2a._core`.
//!
//! Exposes the Rust analysis engine to Python.
//!
// Many Rust items (fields, methods, structs) are intentionally unused from Rust
// because the consumer is Python via PyO3 bindings, not other Rust code.
// Some coding patterns were generated for consistency across 21 stats modules.
#![allow(dead_code)]
#![allow(
    clippy::useless_conversion,
    clippy::double_parens,
    clippy::filter_map_identity,
    clippy::implicit_saturating_sub,
    clippy::let_and_return,
    clippy::manual_clamp,
    clippy::manual_flatten,
    clippy::manual_is_multiple_of,
    clippy::needless_range_loop,
    clippy::question_mark,
    clippy::redundant_closure,
    clippy::unnecessary_map_or
)]

//! Main entry:
//!   `_core.analyze(source, config_json=None) -> str`  (JSON string)
//!
//! Individual module functions are also exposed for fine-grained usage.

use pyo3::prelude::*;
use pyo3::types::PyDict;

mod core;
mod stats;
mod utils;

use crate::core::analyzer::Analyzer;
use crate::core::config::AnalysisConfig;
use crate::core::loader::DataLoader;
use crate::core::preprocessor::Preprocessor;
use crate::core::schema::DataSchema;

// ─── Helper: Python dict → AnalysisConfig ───────────────────────────

fn config_from_pydict(py: Python<'_>, dict: &Bound<'_, PyDict>) -> PyResult<AnalysisConfig> {
    // Serialize the Python dict → JSON string → Rust struct
    let json_mod = py.import_bound("json")?;
    let json_str: String = json_mod.call_method1("dumps", (dict,))?.extract()?;
    let config: AnalysisConfig = serde_json::from_str(&json_str)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("Invalid config: {}", e)))?;
    Ok(config)
}

fn config_from_json(json_str: &str) -> PyResult<AnalysisConfig> {
    serde_json::from_str(json_str)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("Invalid config JSON: {}", e)))
}

// ─── Core functions ─────────────────────────────────────────────────

/// Run a full analysis on a file and return the results as a JSON string.
///
/// Parameters
/// ----------
/// source : str
///     Path to a data file (CSV, Parquet, JSON, etc.)
/// config_json : str, optional
///     JSON string of AnalysisConfig overrides.
///     Omit to use the default configuration (all analyses enabled).
///
/// Returns
/// -------
/// str
///     JSON string with the full AnalysisReport.
#[pyfunction]
#[pyo3(signature = (source, config_json=None))]
fn analyze(source: &str, config_json: Option<&str>) -> PyResult<String> {
    let config = match config_json {
        Some(json) => config_from_json(json)?,
        None => AnalysisConfig::default(),
    };

    let report = Analyzer::run_from_file(source, &config)?;
    let json = serde_json::to_string(&report).map_err(|e| {
        pyo3::exceptions::PyRuntimeError::new_err(format!("Serialization error: {}", e))
    })?;
    Ok(json)
}

/// Run analysis with a Python dict config and return a Python dict.
///
/// Parameters
/// ----------
/// py : Python
///     GIL token
/// source : str
///     Path to a data file
/// config_dict : dict, optional
///     Config overrides as a Python dict
///
/// Returns
/// -------
/// dict
///     Python dict with the full AnalysisReport.
#[pyfunction]
#[pyo3(signature = (source, config_dict=None))]
fn analyze_to_dict(
    py: Python<'_>,
    source: &str,
    config_dict: Option<&Bound<'_, PyDict>>,
) -> PyResult<PyObject> {
    let config = match config_dict {
        Some(d) => config_from_pydict(py, d)?,
        None => AnalysisConfig::default(),
    };

    let report = Analyzer::run_from_file(source, &config)?;
    let json_str = serde_json::to_string(&report).map_err(|e| {
        pyo3::exceptions::PyRuntimeError::new_err(format!("Serialization error: {}", e))
    })?;

    // Parse JSON → Python dict via json.loads
    let json_mod = py.import_bound("json")?;
    let result = json_mod.call_method1("loads", (json_str,))?;
    Ok(result.into_py(py))
}

/// Load a data file and return schema info as JSON.
#[pyfunction]
fn load_schema(source: &str) -> PyResult<String> {
    let df = DataLoader::load(source)?;
    let schema = DataSchema::from_dataframe(&df);
    let json = serde_json::to_string(&schema)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
    Ok(json)
}

/// Return the default config as a JSON string.
#[pyfunction]
fn default_config() -> PyResult<String> {
    let config = AnalysisConfig::default();
    serde_json::to_string(&config)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
}

/// Return a minimal config (only descriptive) as JSON.
#[pyfunction]
fn minimal_config() -> PyResult<String> {
    let config = AnalysisConfig::minimal();
    serde_json::to_string(&config)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
}

/// Return a fast config (basic analyses only) as JSON.
#[pyfunction]
fn fast_config() -> PyResult<String> {
    let config = AnalysisConfig::fast();
    serde_json::to_string(&config)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
}

/// Return the basic-only config as JSON.
#[pyfunction]
fn basic_only_config() -> PyResult<String> {
    let config = AnalysisConfig::basic_only();
    serde_json::to_string(&config)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
}

/// Preprocess a data file and return preprocessing report as JSON.
#[pyfunction]
#[pyo3(signature = (source, missing_threshold=0.95))]
fn preprocess(source: &str, missing_threshold: f64) -> PyResult<String> {
    let df = DataLoader::load(source)?;
    let schema = DataSchema::from_dataframe(&df);
    let (_, result) = Preprocessor::process(&df, &schema, missing_threshold);
    serde_json::to_string(&result)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
}

/// Library version.
#[pyfunction]
fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

// ─── Individual stat functions (for fine-grained Python access) ─────

#[pyfunction]
fn compute_descriptive(source: &str) -> PyResult<String> {
    let df = DataLoader::load(source)?;
    let schema = DataSchema::from_dataframe(&df);
    let result = crate::stats::descriptive::DescriptiveStats::new(&df, &schema).compute();
    serde_json::to_string(&result)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
}

#[pyfunction]
#[pyo3(signature = (source, threshold=0.9))]
fn compute_correlation(source: &str, threshold: f64) -> PyResult<String> {
    let df = DataLoader::load(source)?;
    let schema = DataSchema::from_dataframe(&df);
    let result =
        crate::stats::correlation::CorrelationStats::new(&df, &schema, threshold).compute();
    serde_json::to_string(&result)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
}

#[pyfunction]
fn compute_distribution(source: &str) -> PyResult<String> {
    let df = DataLoader::load(source)?;
    let schema = DataSchema::from_dataframe(&df);
    let result = crate::stats::distribution::DistributionStats::new(&df, &schema).compute();
    serde_json::to_string(&result)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
}

#[pyfunction]
fn compute_missing(source: &str) -> PyResult<String> {
    let df = DataLoader::load(source)?;
    let schema = DataSchema::from_dataframe(&df);
    let result = crate::stats::missing::MissingStats::new(&df, &schema).compute();
    serde_json::to_string(&result)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
}

#[pyfunction]
#[pyo3(signature = (source, method="iqr", threshold=1.5))]
fn compute_outlier(source: &str, method: &str, threshold: f64) -> PyResult<String> {
    let df = DataLoader::load(source)?;
    let schema = DataSchema::from_dataframe(&df);
    let m = match method {
        "zscore" | "z" => crate::stats::outlier::OutlierMethod::Zscore,
        _ => crate::stats::outlier::OutlierMethod::Iqr,
    };
    let result = crate::stats::outlier::OutlierStats::new(&df, &schema, m, threshold).compute();
    serde_json::to_string(&result)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
}

#[pyfunction]
fn compute_quality(source: &str) -> PyResult<String> {
    let df = DataLoader::load(source)?;
    let schema = DataSchema::from_dataframe(&df);
    let result = crate::stats::quality::QualityStats::new(&df, &schema).compute();
    serde_json::to_string(&result)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
}

#[pyfunction]
fn compute_insight_engine(source: &str) -> PyResult<String> {
    let df = DataLoader::load(source)?;
    let schema = DataSchema::from_dataframe(&df);
    let result = crate::stats::insight_engine::InsightEngine::new(&df, &schema).compute();
    serde_json::to_string(&result)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
}

#[pyfunction]
fn compute_ml_readiness(source: &str) -> PyResult<String> {
    let df = DataLoader::load(source)?;
    let schema = DataSchema::from_dataframe(&df);
    let result = crate::stats::ml_readiness::MlReadinessStats::new(&df, &schema).compute();
    serde_json::to_string(&result)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
}

// ─── Module definition ──────────────────────────────────────────────

/// The `_core` native extension module.
#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Core functions
    m.add_function(wrap_pyfunction!(analyze, m)?)?;
    m.add_function(wrap_pyfunction!(analyze_to_dict, m)?)?;
    m.add_function(wrap_pyfunction!(load_schema, m)?)?;
    m.add_function(wrap_pyfunction!(preprocess, m)?)?;

    // Config helpers
    m.add_function(wrap_pyfunction!(default_config, m)?)?;
    m.add_function(wrap_pyfunction!(minimal_config, m)?)?;
    m.add_function(wrap_pyfunction!(fast_config, m)?)?;
    m.add_function(wrap_pyfunction!(basic_only_config, m)?)?;

    // Individual stats
    m.add_function(wrap_pyfunction!(compute_descriptive, m)?)?;
    m.add_function(wrap_pyfunction!(compute_correlation, m)?)?;
    m.add_function(wrap_pyfunction!(compute_distribution, m)?)?;
    m.add_function(wrap_pyfunction!(compute_missing, m)?)?;
    m.add_function(wrap_pyfunction!(compute_outlier, m)?)?;
    m.add_function(wrap_pyfunction!(compute_quality, m)?)?;
    m.add_function(wrap_pyfunction!(compute_insight_engine, m)?)?;
    m.add_function(wrap_pyfunction!(compute_ml_readiness, m)?)?;

    // Meta
    m.add_function(wrap_pyfunction!(version, m)?)?;

    Ok(())
}
