"""Tests: file loading, multi-format support, analysis pipeline."""

from __future__ import annotations

from pathlib import Path

import pytest

from f2a import AnalysisConfig, analyze


class TestCSVAnalysis:
    """Full CSV analysis pipeline."""

    def test_sections_present(self, csv_path: Path):
        report = analyze(str(csv_path))
        expected = {
            "descriptive",
            "correlation",
            "distribution",
            "missing",
            "outlier",
            "quality",
            "categorical",
            "pca",
        }
        present = set(report.sections)
        missing = expected - present
        assert not missing, f"Missing sections: {missing}"

    def test_descriptive_stats(self, csv_path: Path):
        report = analyze(str(csv_path))
        desc = report.get("descriptive")
        assert desc is not None
        assert len(desc.get("numeric", [])) >= 3
        nc = desc["numeric"][0]
        for key in ("mean", "std", "min", "max"):
            assert key in nc, f"Missing key '{key}' in numeric stats"

    def test_categorical_stats(self, csv_path: Path):
        report = analyze(str(csv_path))
        cat = report.get("categorical")
        assert cat is not None

    def test_quality_score(self, csv_path: Path):
        report = analyze(str(csv_path))
        qual = report.get("quality")
        assert qual is not None
        score = qual.get("overall_score", 0)
        assert 0.0 < score <= 1.0

    def test_source_and_schema(self, csv_path: Path):
        report = analyze(str(csv_path))
        assert report.source is not None
        assert isinstance(report.schema, dict)
        assert isinstance(report.results, dict)


class TestMultiFormat:
    """Multi-format loading."""

    @pytest.fixture(params=["csv_path", "tsv_path", "json_path", "jsonl_path", "parquet_path"])
    def data_path(self, request) -> Path:
        return request.getfixturevalue(request.param)

    def test_load_and_analyze(self, data_path: Path):
        cfg = AnalysisConfig.minimal()
        report = analyze(str(data_path), config=cfg)
        assert len(report.sections) > 0


class TestNumericOnly:
    """Numeric-only dataset analysis."""

    def test_numeric_detection(self, numeric_csv_path: Path):
        report = analyze(str(numeric_csv_path))
        desc = report.get("descriptive")
        assert desc is not None
        # x5 (Poisson with low cardinality) may be classified categorical
        assert len(desc.get("numeric", [])) >= 4

    def test_all_21_sections(self, csv_path: Path):
        cfg = AnalysisConfig(advanced=True)
        report = analyze(str(csv_path), config=cfg)

        all_expected = {
            "descriptive",
            "correlation",
            "distribution",
            "missing",
            "outlier",
            "categorical",
            "feature_importance",
            "pca",
            "duplicates",
            "quality",
            "statistical_tests",
            "clustering",
            "advanced_anomaly",
            "advanced_correlation",
            "advanced_distribution",
            "advanced_dimreduction",
            "feature_insights",
            "insight_engine",
            "column_role",
            "cross_analysis",
            "ml_readiness",
        }
        present = set(report.sections)
        missing = all_expected - present
        assert not missing, f"Missing advanced sections: {missing}"


class TestMLReadiness:
    """ML readiness scoring."""

    def test_ml_readiness_fields(self, csv_path: Path):
        cfg = AnalysisConfig(advanced=True)
        report = analyze(str(csv_path), config=cfg)
        ml = report.get("ml_readiness")
        assert ml is not None
        assert "overall_score" in ml
        assert "grade" in ml
        assert 0.0 <= ml["overall_score"] <= 1.0

    def test_insight_engine(self, csv_path: Path):
        cfg = AnalysisConfig(advanced=True)
        report = analyze(str(csv_path), config=cfg)
        ie = report.get("insight_engine")
        assert ie is not None
        assert "insights" in ie
