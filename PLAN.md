# f2a (File to Analysis) — Technical Design Document

> **Version**: 0.1.0 (Draft)
> **Date**: 2026-03-13
> **Status**: Design Phase

---

## 1. Project Overview

**f2a** is a Python library that takes various data sources (local files, Hugging Face datasets, etc.)
and automatically performs **Descriptive Statistics** analysis and **Visualization**.

### 1.1 Core Goals
- **One-click Analysis**: Full descriptive statistics + visualization from a single file path or HuggingFace URL
- **Diverse Input Support**: CSV, JSON, Parquet, Excel, TSV, Hugging Face `datasets`
- **Rich Statistics**: Summary statistics, distribution analysis, correlation analysis, missing data analysis
- **Automatic Visualization**: Histograms, boxplots, correlation heatmaps, missing data maps, etc.
- **Report Generation**: Automatically produce HTML reports from analysis results

### 1.2 Usage Scenario

```python
import f2a

# Analyze a local file
report = f2a.analyze("data/sales.csv")
report.show()            # Print summary to console
report.to_html("out/")  # Save HTML report

# Analyze a Hugging Face dataset
report = f2a.analyze("hf://imdb")
report.show()

# Detailed access
report.stats.summary()        # Summary statistics DataFrame
report.stats.correlation()    # Correlation matrix
report.viz.plot_distributions()  # Distribution plots
```

---

## 2. Architecture

### 2.1 Layered Structure

```
┌─────────────────────────────────────────────┐
│                  Public API                 │
│         f2a.analyze() / f2a.load()          │
├─────────────────────────────────────────────┤
│               Core Orchestrator             │
│         Analyzer (pipeline control)         │
├──────────┬──────────┬──────────┬────────────┤
│  Loader  │  Stats   │   Viz    │  Reporter  │
│  Data    │  Stat    │  Visual  │  Report    │
│  Loading │ Analysis │ ization  │ Generation │
├──────────┴──────────┴──────────┴────────────┤
│                  Utilities                  │
│     Type Inference · Validation · Logging   │
└─────────────────────────────────────────────┘
```

### 2.2 Directory Structure

```
f2a/
├── pyproject.toml            # Build config (PEP 621)
├── README.md                 # Project introduction
├── PLAN.md                   # This document
├── LICENSE                   # MIT License
│
├── src/
│   └── f2a/
│       ├── __init__.py       # Public API exports
│       ├── _version.py       # Version management
│       │
│       ├── core/
│       │   ├── __init__.py
│       │   ├── loader.py     # File/HF data loading
│       │   ├── analyzer.py   # Analysis orchestrator
│       │   └── schema.py     # Column type inference & schema
│       │
│       ├── stats/
│       │   ├── __init__.py
│       │   ├── descriptive.py    # Descriptive stats (mean, median, variance, etc.)
│       │   ├── distribution.py   # Distribution analysis (skewness, kurtosis, normality)
│       │   ├── correlation.py    # Correlation analysis
│       │   └── missing.py        # Missing data analysis
│       │
│       ├── viz/
│       │   ├── __init__.py
│       │   ├── theme.py          # Visualization theme/style
│       │   ├── plots.py          # Basic plots (histogram, bar, box)
│       │   ├── dist_plots.py     # Distribution visualization
│       │   ├── corr_plots.py     # Correlation visualization
│       │   └── missing_plots.py  # Missing data visualization
│       │
│       ├── report/
│       │   ├── __init__.py
│       │   ├── generator.py      # Report generation engine
│       │   └── templates/        # HTML templates
│       │       └── base.html
│       │
│       └── utils/
│           ├── __init__.py
│           ├── type_inference.py  # Automatic data type inference
│           ├── validators.py      # Input validation
│           └── logging.py         # Logging configuration
│
├── tests/
│   ├── __init__.py
│   ├── conftest.py           # pytest fixtures
│   ├── test_loader.py
│   ├── test_descriptive.py
│   ├── test_correlation.py
│   ├── test_viz.py
│   └── test_report.py
│
└── examples/
    ├── quickstart.py         # Quick start example
    └── huggingface_demo.py   # HF dataset example
```

---

## 3. Core Module Design

### 3.1 Loader (`core/loader.py`)

Automatically detects the data source and converts it uniformly to a `pandas.DataFrame`.

| Input Type | Detection Method | Conversion Method |
|---|---|---|
| CSV / TSV | Extension `.csv`, `.tsv` | `pd.read_csv()` |
| JSON / JSONL | Extension `.json`, `.jsonl` | `pd.read_json()` |
| Parquet | Extension `.parquet` | `pd.read_parquet()` |
| Excel | Extension `.xlsx`, `.xls` | `pd.read_excel()` |
| HuggingFace | `hf://` prefix or `org/dataset` pattern | `datasets.load_dataset()` → `.to_pandas()` |

**Core Interface:**
```python
class DataLoader:
    def load(self, source: str, **kwargs) -> pd.DataFrame:
        """Analyze source string and route to appropriate loader"""

    def _detect_source_type(self, source: str) -> SourceType:
        """Auto-detect source type"""
```

### 3.2 Stats (`stats/`)

#### 3.2.1 Descriptive Statistics (`descriptive.py`)

| Statistic | Numeric | Categorical |
|---|---|---|
| count / unique | ✅ | ✅ |
| mean / median | ✅ | — |
| std / variance | ✅ | — |
| min / max / range | ✅ | — |
| Q1, Q3, IQR | ✅ | — |
| top / freq | — | ✅ |
| mode | ✅ | ✅ |

#### 3.2.2 Distribution Analysis (`distribution.py`)

- **Skewness** & **Kurtosis**
- **Normality Tests**: Shapiro-Wilk (n ≤ 5000), D'Agostino-Pearson
- **Quantile Table**: 5%, 10%, 25%, 50%, 75%, 90%, 95%

#### 3.2.3 Correlation Analysis (`correlation.py`)

- **Pearson** correlation (numeric-numeric)
- **Spearman** rank correlation (numeric-numeric, nonlinear)
- **Cramér's V** (categorical-categorical)
- Multicollinearity warning (|r| > 0.9)

#### 3.2.4 Missing Data Analysis (`missing.py`)

- Column-wise missing ratio
- Missing pattern analysis (MCAR / MAR hints)
- Row-wise missing distribution

### 3.3 Viz (`viz/`)

| Chart Type | Target | Module |
|---|---|---|
| Histogram + KDE | Numeric columns | `dist_plots.py` |
| Boxplot | Numeric columns | `plots.py` |
| Bar chart (frequency) | Categorical columns | `plots.py` |
| Correlation heatmap | Numeric column pairs | `corr_plots.py` |
| Pairplot | Top N numeric columns | `corr_plots.py` |
| Missing data matrix | All columns | `missing_plots.py` |
| Violin plot | Numeric columns | `dist_plots.py` |

**Visualization Theme**: Unified style management in `viz/theme.py` (color palette, font size, etc.)

### 3.4 Report (`report/`)

Generates comprehensive HTML reports from analysis results.

**Report Structure:**
1. **Overview Section**: Dataset name, row/column counts, memory usage
2. **Variable Summary**: Column types, missing ratios, key statistics
3. **Distribution Section**: Distribution visualization for each column
4. **Correlation Section**: Correlation heatmap + key correlated pairs
5. **Missing Data Section**: Missing pattern visualization
6. **Warnings Section**: Outliers, high correlation, high missing ratios, etc.

---

## 4. Data Flow

```
Input (file path / HF URL)
       │
       ▼
  ┌─────────┐
  │  Loader  │ ──→ pd.DataFrame
  └────┬─────┘
       │
       ▼
  ┌──────────┐
  │  Schema  │ ──→ Column type inference (numeric/categorical/text/datetime)
  └────┬─────┘
       │
       ├──→ Stats.descriptive()  ──→ StatResult
       ├──→ Stats.distribution() ──→ StatResult
       ├──→ Stats.correlation()  ──→ StatResult
       └──→ Stats.missing()      ──→ StatResult
              │
              ▼
       ┌────────────┐
       │  Viz Engine │ ──→ matplotlib Figure objects
       └─────┬──────┘
             │
             ▼
       ┌───────────┐
       │  Reporter  │ ──→ AnalysisReport
       └───────────┘
             │
             ├──→ .show()       (console output)
             ├──→ .to_html()    (HTML file)
             └──→ .to_dict()    (programmatic access)
```

---

## 5. Dependencies

### 5.1 Required (Core)

| Package | Version | Purpose |
|---|---|---|
| `pandas` | ≥ 2.0 | DataFrame core |
| `numpy` | ≥ 1.24 | Numeric operations |
| `matplotlib` | ≥ 3.7 | Basic visualization |
| `seaborn` | ≥ 0.13 | Statistical visualization |
| `scipy` | ≥ 1.11 | Statistical tests |

### 5.2 Optional

| Package | Purpose | Extras Name |
|---|---|---|
| `datasets` | HuggingFace dataset loading | `[hf]` |
| `openpyxl` | Excel file support | `[excel]` |
| `pyarrow` | Parquet file support | `[parquet]` |
| `rich` | Console output formatting | `[rich]` |
| `jinja2` | HTML report templates | `[report]` |

### 5.3 Install Commands

```bash
# Basic install
pip install f2a

# With HuggingFace support
pip install f2a[hf]

# All features
pip install f2a[all]
```

---

## 6. Development Roadmap

### Phase 1 — Foundation (v0.1.0) ← **Current**
- [x] Project structure setup (pyproject.toml, directories)
- [x] Basic Loader (CSV, JSON)
- [x] Descriptive statistics module (descriptive.py)
- [x] Basic visualization (histograms, boxplots)
- [x] Console output (show)

### Phase 2 — Expansion (v0.2.0)
- [ ] HuggingFace dataset loader
- [ ] Correlation analysis & heatmap
- [ ] Missing data analysis & visualization
- [ ] HTML report generation

### Phase 3 — Enhancement (v0.3.0)
- [ ] Distribution analysis (normality tests, etc.)
- [ ] Large dataset support (chunk loading)
- [ ] Interactive visualization (plotly option)
- [ ] CLI interface

### Phase 4 — Stabilization (v1.0.0)
- [ ] API stabilization & documentation
- [ ] Comprehensive test coverage > 80%
- [ ] PyPI deployment
- [ ] Tutorials & example notebooks

---

## 7. Coding Conventions

- **Python**: 3.10+
- **Style**: PEP 8, Black formatter, isort
- **Type Hints**: Required for all public APIs
- **Docstrings**: Google style
- **Testing**: pytest, unit testing principles
- **Linting**: ruff

---

## 8. Core Class Design

### 8.1 AnalysisReport

```python
@dataclass
class AnalysisReport:
    """Top-level container for analysis results"""
    dataset_name: str
    shape: tuple[int, int]
    schema: DataSchema
    stats: StatsResult
    figures: dict[str, Figure]
    warnings: list[str]

    def show(self) -> None: ...
    def to_html(self, output_dir: str) -> Path: ...
    def to_dict(self) -> dict: ...
```

### 8.2 StatsResult

```python
@dataclass
class StatsResult:
    """Container for statistical analysis results"""
    summary: pd.DataFrame          # Summary statistics
    correlation_matrix: pd.DataFrame  # Correlation matrix
    missing_info: pd.DataFrame     # Missing data info
    distribution_info: pd.DataFrame   # Distribution info

    def get_numeric_summary(self) -> pd.DataFrame: ...
    def get_categorical_summary(self) -> pd.DataFrame: ...
```

---

## 9. Error Handling Strategy

| Scenario | Handling |
|---|---|
| File not found | `FileNotFoundError` with clear message |
| Unsupported format | `UnsupportedFormatError` (custom) |
| HF dataset load failure | `DataLoadError` (custom) + cause chaining |
| Empty dataset | `EmptyDataError` (custom) |
| No numeric columns | Warning log + skip relevant analysis |

---

*This document is continuously updated as the project progresses.*
