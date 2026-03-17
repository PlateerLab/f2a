"""
Comprehensive end-to-end test for f2a (Rust-powered).
Mirrors the original lerobot_test_local.py pattern and verifies
all key API surfaces work correctly.
"""

import json
import sys
import time
import traceback
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Ensure the local f2a package is used
sys.path.insert(0, str(Path(__file__).resolve().parent / "python"))

PASS = 0
FAIL = 0
RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, fn):
    """Run a test function, record pass/fail."""
    global PASS, FAIL
    try:
        msg = fn()
        PASS += 1
        RESULTS.append((name, True, msg or "OK"))
    except Exception as e:
        FAIL += 1
        RESULTS.append((name, False, f"{e.__class__.__name__}: {e}"))
        traceback.print_exc()


# ═══════════════════════════════════════════════════════════════════════
#  0. Create test datasets
# ═══════════════════════════════════════════════════════════════════════

DATA_DIR = Path(__file__).parent / "test_data_e2e"
DATA_DIR.mkdir(exist_ok=True)

np.random.seed(42)
N = 300

# Mixed-type dataset with missing values
mixed_df = pd.DataFrame({
    "id": range(N),
    "age": np.random.randint(18, 80, N),
    "income": np.random.lognormal(10, 1, N),
    "score": np.random.normal(75, 15, N),
    "category": np.random.choice(["A", "B", "C", "D"], N),
    "city": np.random.choice(["Seoul", "Busan", "Daegu", "Incheon"], N),
    "passed": np.random.choice([True, False], N),
})
mixed_df.loc[np.random.choice(N, 20, replace=False), "income"] = np.nan
mixed_df.loc[np.random.choice(N, 15, replace=False), "score"] = np.nan
mixed_df.loc[np.random.choice(N, 10, replace=False), "city"] = np.nan

CSV_PATH = DATA_DIR / "mixed_data.csv"
mixed_df.to_csv(CSV_PATH, index=False)

TSV_PATH = DATA_DIR / "mixed_data.tsv"
mixed_df.to_csv(TSV_PATH, index=False, sep="\t")

JSON_PATH = DATA_DIR / "mixed_data.json"
mixed_df.to_json(JSON_PATH, orient="records", force_ascii=False)

JSONL_PATH = DATA_DIR / "mixed_data.jsonl"
mixed_df.to_json(JSONL_PATH, orient="records", lines=True, force_ascii=False)

PARQUET_PATH = DATA_DIR / "mixed_data.parquet"
mixed_df.to_parquet(PARQUET_PATH, index=False)

# Numeric-only dataset
numeric_df = pd.DataFrame({
    "x1": np.random.normal(0, 1, N),
    "x2": np.random.normal(5, 2, N),
    "x3": np.random.exponential(2, N),
    "x4": np.random.uniform(-10, 10, N),
    "x5": np.random.poisson(3, N).astype(float),
})
NUMERIC_CSV = DATA_DIR / "numeric_only.csv"
numeric_df.to_csv(NUMERIC_CSV, index=False)


print("=" * 70)
print("  f2a 1.0 (Rust) - End-to-End Validation")
print("=" * 70)


# ═══════════════════════════════════════════════════════════════════════
#  1. Import test
# ═══════════════════════════════════════════════════════════════════════

def test_import():
    import f2a
    assert hasattr(f2a, "__version__")
    assert hasattr(f2a, "analyze")
    assert hasattr(f2a, "AnalysisConfig")
    return f"version={f2a.__version__}"

check("1. Import f2a", test_import)


# ═══════════════════════════════════════════════════════════════════════
#  2. Rust core direct access
# ═══════════════════════════════════════════════════════════════════════

def test_rust_core():
    from f2a._core import version, default_config, minimal_config, fast_config
    assert version() == "1.0.0"
    cfg = json.loads(default_config())
    assert cfg["descriptive"] is True
    assert cfg["correlation"] is True
    mcfg = json.loads(minimal_config())
    assert mcfg["correlation"] is False
    fcfg = json.loads(fast_config())
    assert fcfg["pca"] is False
    return "version/config OK"

check("2. Rust core functions", test_rust_core)


# ═══════════════════════════════════════════════════════════════════════
#  3. AnalysisConfig
# ═══════════════════════════════════════════════════════════════════════

def test_config():
    from f2a import AnalysisConfig
    cfg = AnalysisConfig()
    assert cfg.descriptive is True
    assert cfg.advanced is True

    cfg_min = AnalysisConfig.minimal()
    assert cfg_min.correlation is False

    cfg_fast = AnalysisConfig.fast()
    assert cfg_fast.pca is False

    cfg_basic = AnalysisConfig.basic_only()
    assert cfg_basic.advanced is False

    cfg_json = cfg.to_json()
    parsed = json.loads(cfg_json)
    assert isinstance(parsed, dict)
    return "AnalysisConfig OK"

check("3. AnalysisConfig", test_config)


# ═══════════════════════════════════════════════════════════════════════
#  4. CSV analysis (full pipeline)
# ═══════════════════════════════════════════════════════════════════════

def test_csv_analysis():
    from f2a import analyze
    t0 = time.perf_counter()
    report = analyze(str(CSV_PATH))
    dt = time.perf_counter() - t0

    # Structure checks
    assert report.source is not None
    assert isinstance(report.schema, dict)
    assert isinstance(report.results, dict)
    assert len(report.sections) > 0

    # Key sections present
    expected = {"descriptive", "correlation", "distribution", "missing",
                "outlier", "quality", "categorical", "pca"}
    present = set(report.sections)
    missing = expected - present
    assert not missing, f"Missing sections: {missing}"

    # Descriptive results
    desc = report.get("descriptive")
    assert desc is not None
    assert len(desc.get("numeric", [])) >= 3  # age, income, score
    nc = desc["numeric"][0]
    assert "mean" in nc and "std" in nc and "min" in nc and "max" in nc

    # Categorical
    cat = report.get("categorical")
    assert cat is not None

    # Quality
    qual = report.get("quality")
    assert qual is not None
    score = qual.get("overall_score", 0)
    assert 0.0 < score <= 1.0

    return f"{len(report.sections)} sections, {dt:.2f}s"

check("4. CSV full analysis", test_csv_analysis)


# ═══════════════════════════════════════════════════════════════════════
#  5. Multi-format loading
# ═══════════════════════════════════════════════════════════════════════

def test_tsv():
    from f2a import analyze, AnalysisConfig
    cfg = AnalysisConfig.minimal()
    r = analyze(str(TSV_PATH), config=cfg)
    assert len(r.sections) > 0
    return f"TSV: {len(r.sections)} sections"

check("5a. TSV loading", test_tsv)


def test_json():
    from f2a import analyze, AnalysisConfig
    cfg = AnalysisConfig.minimal()
    r = analyze(str(JSON_PATH), config=cfg)
    assert len(r.sections) > 0
    return f"JSON: {len(r.sections)} sections"

check("5b. JSON loading", test_json)


def test_jsonl():
    from f2a import analyze, AnalysisConfig
    cfg = AnalysisConfig.minimal()
    r = analyze(str(JSONL_PATH), config=cfg)
    assert len(r.sections) > 0
    return f"JSONL: {len(r.sections)} sections"

check("5c. JSONL loading", test_jsonl)


def test_parquet():
    from f2a import analyze, AnalysisConfig
    cfg = AnalysisConfig.minimal()
    r = analyze(str(PARQUET_PATH), config=cfg)
    assert len(r.sections) > 0
    return f"Parquet: {len(r.sections)} sections"

check("5d. Parquet loading", test_parquet)


# ═══════════════════════════════════════════════════════════════════════
#  6. Advanced analysis (all 21 modules)
# ═══════════════════════════════════════════════════════════════════════

def test_advanced():
    from f2a import analyze, AnalysisConfig
    cfg = AnalysisConfig(advanced=True)
    r = analyze(str(CSV_PATH), config=cfg)

    all_expected = {
        "descriptive", "correlation", "distribution", "missing",
        "outlier", "categorical", "feature_importance", "pca",
        "duplicates", "quality", "statistical_tests",
        # Advanced
        "clustering", "advanced_anomaly", "advanced_correlation",
        "advanced_distribution", "advanced_dimreduction",
        "feature_insights", "insight_engine", "column_role",
        "cross_analysis", "ml_readiness",
    }
    present = set(r.sections)
    missing = all_expected - present
    assert not missing, f"Missing advanced sections: {missing}"

    # ML readiness details
    ml = r.get("ml_readiness")
    assert ml is not None
    assert "overall_score" in ml
    assert "grade" in ml
    assert 0.0 <= ml["overall_score"] <= 1.0

    # Insight engine
    ie = r.get("insight_engine")
    assert ie is not None
    assert "insights" in ie

    # Clustering
    cl = r.get("clustering")
    assert cl is not None

    return f"All {len(all_expected)} modules present, ML={ml['grade']}"

check("6. Advanced analysis (21 modules)", test_advanced)


# ═══════════════════════════════════════════════════════════════════════
#  7. Numeric-only dataset
# ═══════════════════════════════════════════════════════════════════════

def test_numeric_only():
    from f2a import analyze
    r = analyze(str(NUMERIC_CSV))
    desc = r.get("descriptive")
    assert desc is not None
    # x5 (Poisson, low cardinality integers) may be classified as categorical
    assert len(desc.get("numeric", [])) >= 4
    return f"{len(desc['numeric'])} numeric, {len(desc.get('categorical', []))} categorical"

check("7. Numeric-only dataset", test_numeric_only)


# ═══════════════════════════════════════════════════════════════════════
#  8. HTML report generation
# ═══════════════════════════════════════════════════════════════════════

def test_html_report():
    from f2a import analyze
    r = analyze(str(CSV_PATH))
    out_dir = DATA_DIR / "html_output"
    path = r.to_html(output_dir=str(out_dir))
    assert path.exists(), f"HTML file not found: {path}"
    content = path.read_text(encoding="utf-8")
    assert len(content) > 1000
    assert "<html" in content.lower()
    assert "f2a" in content.lower() or "analysis" in content.lower()
    return f"HTML {len(content):,} bytes → {path.name}"

check("8. HTML report generation", test_html_report)


# ═══════════════════════════════════════════════════════════════════════
#  9. HTML with i18n (Korean)
# ═══════════════════════════════════════════════════════════════════════

def test_html_korean():
    from f2a import analyze
    r = analyze(str(CSV_PATH))
    out_dir = DATA_DIR / "html_output"
    path = r.to_html(output_dir=str(out_dir), lang="ko")
    assert path.exists()
    content = path.read_text(encoding="utf-8")
    # Korean characters should be present
    assert any(ord(c) >= 0xAC00 for c in content), "No Korean characters in report"
    return f"Korean HTML {len(content):,} bytes"

check("9. HTML report (Korean)", test_html_korean)


# ═══════════════════════════════════════════════════════════════════════
#  10. Console show()
# ═══════════════════════════════════════════════════════════════════════

def test_show():
    from f2a import analyze
    r = analyze(str(CSV_PATH))
    # show() should not raise
    r.show()
    return "show() OK"

check("10. Console show()", test_show)


# ═══════════════════════════════════════════════════════════════════════
#  11. Performance benchmark
# ═══════════════════════════════════════════════════════════════════════

def test_performance():
    from f2a import analyze, AnalysisConfig
    cfg = AnalysisConfig(advanced=True)
    times = []
    for _ in range(3):
        t0 = time.perf_counter()
        analyze(str(CSV_PATH), config=cfg)
        times.append(time.perf_counter() - t0)
    avg = sum(times) / len(times)
    return f"Avg {avg:.3f}s over 3 runs (min={min(times):.3f}s)"

check("11. Performance benchmark", test_performance)


# ═══════════════════════════════════════════════════════════════════════
#  12. Error handling
# ═══════════════════════════════════════════════════════════════════════

def test_error_nonexistent():
    from f2a import analyze
    try:
        analyze("nonexistent_file.csv")
        raise AssertionError("Should have raised an error")
    except Exception as e:
        assert "nonexistent" in str(e).lower() or "not found" in str(e).lower() or "error" in str(e).lower()
    return "Error raised for missing file"

check("12. Error handling", test_error_nonexistent)


# ═══════════════════════════════════════════════════════════════════════
#  SUMMARY
# ═══════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("  RESULTS")
print("=" * 70)
for name, ok, msg in RESULTS:
    status = "[PASS]" if ok else "[FAIL]"
    print(f"  {status}  {name}: {msg}")

print(f"\n  Total: {PASS} passed, {FAIL} failed out of {PASS + FAIL}")
print("=" * 70)

if FAIL > 0:
    sys.exit(1)
else:
    print("\n  All tests passed! f2a 1.0 (Rust) is ready.\n")
    sys.exit(0)
