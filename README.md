# f2a

> **File to Analysis** — Automatically perform statistical analysis from any data source.

`f2a` is a high-performance data analysis library that provides a simple
Python API while running all compute-heavy operations in native Rust via
[PyO3](https://pyo3.rs) and [maturin](https://www.maturin.rs).

## Architecture

```
┌─────────────────────────────────────────┐
│            Python API Layer             │
│  f2a.analyze()  /  AnalysisConfig       │
│  Report generation (Jinja2 HTML)        │
│  Visualization (matplotlib / seaborn)   │
└──────────────┬──────────────────────────┘
               │  PyO3 FFI
┌──────────────▼──────────────────────────┐
│          Rust Core  (_core)             │
│  Data loading (polars)                  │
│  Schema inference & preprocessing       │
│  21 statistical analysis modules        │
│  Parallel computation (rayon)           │
└─────────────────────────────────────────┘
```

### What runs in Rust

| Layer | Modules |
|---|---|
| **Core** | Loader (CSV/TSV/Parquet/JSON/JSONL), Schema inference, Preprocessor, Analyzer orchestration |
| **Basic Stats** | Descriptive, Correlation, Distribution, Missing, Outlier, Categorical, Duplicates, Quality, Feature Importance, PCA |
| **Advanced Stats** | Statistical Tests, Clustering, Anomaly Detection, Advanced Correlation, Advanced Distribution, Dimensionality Reduction, Feature Insights, Insight Engine, Column Role, Cross Analysis, ML Readiness |

### What stays in Python

| Layer | Reason |
|---|---|
| **Visualization** | matplotlib/seaborn — no Rust equivalent worth the effort |
| **HTML Report** | Jinja2 templating is inherently Python |
| **i18n** | String-heavy, low compute |

## Quick Start

```python
import f2a

report = f2a.analyze("data.csv")
report.show()               # Rich console summary
report.to_html("output/")   # Self-contained HTML report
report.get("quality")       # Dict access to any section
```

## Installation

```bash
pip install f2a
```

### Building from Source

```bash
# Prerequisites: Rust toolchain, Python >=3.10
pip install maturin

# Development build (editable)
maturin develop --release

# Build wheel
maturin build --release
```

## Supported Formats

CSV, TSV, JSON, JSONL, Parquet — plus optional extras for Excel, SPSS, SAS, HDF5, ODF, and more:

```bash
pip install f2a[io]
```

## License

Apache-2.0
