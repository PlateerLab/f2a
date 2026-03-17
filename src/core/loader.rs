use std::path::Path;

use polars::prelude::*;

use crate::utils::errors::{F2aError, F2aResult};

/// Supported file formats.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FileFormat {
    Csv,
    Tsv,
    Parquet,
    Json,
    JsonLines,
    Excel,
    Feather, // Arrow IPC
}

impl FileFormat {
    /// Detect format from file extension.
    pub fn from_path(path: &Path) -> F2aResult<Self> {
        let ext = path
            .extension()
            .and_then(|e| e.to_str())
            .map(|e| e.to_lowercase())
            .unwrap_or_default();

        match ext.as_str() {
            "csv" => Ok(FileFormat::Csv),
            "tsv" | "tab" => Ok(FileFormat::Tsv),
            "parquet" | "pq" => Ok(FileFormat::Parquet),
            "json" => Ok(FileFormat::Json),
            "jsonl" | "ndjson" => Ok(FileFormat::JsonLines),
            "xlsx" | "xls" | "xlsm" | "xlsb" => Ok(FileFormat::Excel),
            "feather" | "arrow" | "ipc" => Ok(FileFormat::Feather),
            _ => Err(F2aError::UnsupportedFormat(ext)),
        }
    }
}

// ─── DataLoader ─────────────────────────────────────────────────────

/// Fast data loader backed by Polars.
pub struct DataLoader;

impl DataLoader {
    /// Load a file into a Polars `DataFrame`.
    ///
    /// Automatically detects format from the file extension.
    pub fn load(source: &str) -> F2aResult<DataFrame> {
        let path = Path::new(source);

        if !path.exists() {
            return Err(F2aError::DataLoadError(format!(
                "File not found: {}",
                source
            )));
        }

        let fmt = FileFormat::from_path(path)?;

        let df = match fmt {
            FileFormat::Csv => Self::load_csv(path)?,
            FileFormat::Tsv => Self::load_tsv(path)?,
            FileFormat::Parquet => Self::load_parquet(path)?,
            FileFormat::Json => Self::load_json(path)?,
            FileFormat::JsonLines => Self::load_jsonlines(path)?,
            FileFormat::Feather => Self::load_feather(path)?,
            FileFormat::Excel => {
                return Err(F2aError::UnsupportedFormat(
                    "Excel loading requires the Python layer (openpyxl)".into(),
                ));
            }
        };

        if df.height() == 0 || df.width() == 0 {
            return Err(F2aError::EmptyData);
        }

        Ok(df)
    }

    fn load_csv(path: &Path) -> F2aResult<DataFrame> {
        let df = CsvReadOptions::default()
            .with_has_header(true)
            .with_infer_schema_length(Some(10000))
            .try_into_reader_with_file_path(Some(path.into()))?
            .finish()?;
        Ok(df)
    }

    fn load_tsv(path: &Path) -> F2aResult<DataFrame> {
        let df = CsvReadOptions::default()
            .with_has_header(true)
            .with_parse_options(CsvParseOptions::default().with_separator(b'\t'))
            .with_infer_schema_length(Some(10000))
            .try_into_reader_with_file_path(Some(path.into()))?
            .finish()?;
        Ok(df)
    }

    fn load_parquet(path: &Path) -> F2aResult<DataFrame> {
        let file = std::fs::File::open(path)?;
        let df = ParquetReader::new(file).finish()?;
        Ok(df)
    }

    fn load_json(path: &Path) -> F2aResult<DataFrame> {
        let file = std::fs::File::open(path)?;
        let df = JsonReader::new(file).finish()?;
        Ok(df)
    }

    fn load_jsonlines(path: &Path) -> F2aResult<DataFrame> {
        let file = std::fs::File::open(path)?;
        let df = JsonLineReader::new(file).finish()?;
        Ok(df)
    }

    fn load_feather(path: &Path) -> F2aResult<DataFrame> {
        let file = std::fs::File::open(path)?;
        let df = IpcReader::new(file).finish()?;
        Ok(df)
    }
}
