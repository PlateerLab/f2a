# f2a - File to Analysis

> **One line of code to analyze any data file.**
> Automatic descriptive statistics, visualizations, and interactive HTML reports from 24+ file formats and HuggingFace datasets.

[![PyPI](https://img.shields.io/pypi/v/f2a?color=blue)](https://pypi.org/project/f2a/)
[![Python](https://img.shields.io/pypi/pyversions/f2a)](https://pypi.org/project/f2a/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Installation

```bash
pip install f2a
```

All formats (HuggingFace, Excel, Parquet, SPSS, DuckDB, etc.) are supported out of the box. No extras needed.

---

## Quick Start

```python
import f2a

# Local file
report = f2a.analyze("data/sales.csv")
report.show()                    # Print summary to console
report.to_html("output/")       # Save interactive HTML report

# HuggingFace dataset (multiple input styles)
report = f2a.analyze("https://huggingface.co/datasets/imdb")
report = f2a.analyze("hf://imdb")
report = f2a.analyze("imdb")    # org/dataset pattern auto-detected

# Access results programmatically
report.stats.summary             # Summary statistics (DataFrame)
report.stats.correlation_matrix  # Correlation matrix
report.schema.columns            # Column type information
report.to_dict()                 # Everything as a dictionary
```

---

## Multi-Subset HuggingFace Datasets

Datasets with multiple subsets (configs) and splits are **automatically discovered and analyzed individually**.

```python
# Auto-discover all subsets x splits
report = f2a.analyze("https://huggingface.co/datasets/FINAL-Bench/ALL-Bench-Leaderboard")
print(f"Total: {report.shape[0]} rows across {len(report.subsets)} subsets")

for s in report.subsets:
    print(f"  {s.subset}/{s.split}: {s.shape[0]} rows x {s.shape[1]} cols")

# Load specific subset via URL path
report = f2a.analyze("https://huggingface.co/datasets/FINAL-Bench/ALL-Bench-Leaderboard/viewer/agent")

# Or via explicit parameters
report = f2a.analyze("FINAL-Bench/ALL-Bench-Leaderboard", config="agent", split="train")
```

The HTML report includes **tabbed navigation** so each subset/split gets its own analysis page.

---

## HTML Report

`report.to_html()` generates a self-contained HTML file with:

- **Overview cards** - row count, column count, type breakdown, memory usage
- **Summary statistics table** - horizontally scrollable with drag support and sticky first column
- **Visualizations** - distribution histograms, boxplots, correlation heatmap, missing data matrix
- **Warnings** - high correlation alerts, high missing ratio alerts
- **Tabbed UI** for multi-subset datasets

---

## Supported Formats (24+)

| Category | Formats |
|---|---|
| Delimited | `.csv` `.tsv` `.txt` `.dat` `.tab` `.fwf` |
| JSON | `.json` `.jsonl` `.ndjson` |
| Spreadsheet | `.xlsx` `.xls` `.xlsm` `.xlsb` |
| OpenDocument | `.ods` |
| Columnar | `.parquet` `.pq` `.feather` `.ftr` `.arrow` `.ipc` `.orc` |
| HDF5 | `.hdf` `.hdf5` `.h5` |
| Statistical | `.dta` (Stata) `.sas7bdat` `.xpt` (SAS) `.sav` `.zsav` (SPSS) |
| Database | `.sqlite` `.sqlite3` `.db` `.duckdb` |
| Pickle | `.pkl` `.pickle` |
| Markup | `.xml` `.html` `.htm` |
| HuggingFace | `hf://` / URL / `org/dataset` |

---

## Analysis Features

| Feature | Details |
|---|---|
| **Descriptive Statistics** | Mean, median, std, min/max, quartiles, unique count, mode |
| **Distribution Analysis** | Skewness, kurtosis, normality tests |
| **Correlation Analysis** | Pearson, Spearman, multicollinearity warnings |
| **Missing Data** | Per-column missing ratio, overall missing alerts |
| **Type Inference** | Auto-detect numeric, categorical, text, datetime, boolean |
| **Visualization** | Histograms, boxplots, correlation heatmaps, missing data matrix |

---

## API Reference

### `f2a.analyze(source, **kwargs)`

| Parameter | Description |
|---|---|
| `source` | File path, URL, or HuggingFace dataset identifier |
| `config` | HuggingFace dataset config/subset name (optional) |
| `split` | HuggingFace dataset split name (optional) |
| `**kwargs` | Additional arguments passed to the data loader |

**Returns:** `AnalysisReport`

### `AnalysisReport`

| Attribute / Method | Description |
|---|---|
| `.shape` | `(rows, columns)` tuple |
| `.schema` | Column types and metadata |
| `.stats` | Statistical analysis results |
| `.viz` | Visualization access |
| `.subsets` | List of `SubsetReport` (multi-subset HF datasets) |
| `.warnings` | List of warning messages |
| `.show()` | Print summary to console |
| `.to_html(output_dir)` | Save interactive HTML report |
| `.to_dict()` | Export all results as dictionary |

---

## License

MIT License - See [LICENSE](LICENSE) for details.
