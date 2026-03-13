"""Report generation tests."""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

from pathlib import Path

import pandas as pd
import pytest

from f2a.core.analyzer import analyze


class TestAnalysisReport:
    """Integration analysis report tests."""

    def test_analyze_csv(self, sample_csv_path: Path) -> None:
        report = analyze(str(sample_csv_path))
        assert report.shape[0] > 0
        assert report.shape[1] > 0
        assert not report.stats.summary.empty

    def test_report_show(self, sample_csv_path: Path, capsys: pytest.CaptureFixture) -> None:
        report = analyze(str(sample_csv_path))
        report.show()
        captured = capsys.readouterr()
        assert "f2a Analysis Report" in captured.out

    def test_report_to_html(self, sample_csv_path: Path, tmp_path: Path) -> None:
        report = analyze(str(sample_csv_path))
        html_path = report.to_html(str(tmp_path))
        assert html_path.exists()
        content = html_path.read_text(encoding="utf-8")
        assert "f2a Analysis Report" in content

    def test_report_to_dict(self, sample_csv_path: Path) -> None:
        report = analyze(str(sample_csv_path))
        d = report.to_dict()
        assert "dataset_name" in d
        assert "shape" in d
        assert "schema" in d
