"""Tests: HTML report generation, i18n, console show()."""

from __future__ import annotations

from pathlib import Path

from f2a import analyze


class TestHTMLReport:
    """HTML report generation."""

    def test_generates_html_file(self, csv_path: Path, data_dir: Path):
        report = analyze(str(csv_path))
        out_dir = data_dir / "html_output"
        path = report.to_html(output_dir=str(out_dir))
        assert path.exists(), f"HTML file not found: {path}"
        content = path.read_text(encoding="utf-8")
        assert len(content) > 1000
        assert "<html" in content.lower()

    def test_korean_html(self, csv_path: Path, data_dir: Path):
        report = analyze(str(csv_path))
        out_dir = data_dir / "html_output"
        path = report.to_html(output_dir=str(out_dir), lang="ko")
        assert path.exists()
        content = path.read_text(encoding="utf-8")
        # Korean characters (Hangul syllables) should be present
        assert any(ord(c) >= 0xAC00 for c in content), "No Korean characters found"


class TestConsoleOutput:
    """Console show() method."""

    def test_show_does_not_raise(self, csv_path: Path):
        report = analyze(str(csv_path))
        report.show()  # must not raise


class TestErrorHandling:
    """Error handling for invalid inputs."""

    def test_nonexistent_file(self):
        try:
            analyze("nonexistent_file_12345.csv")
            raise AssertionError("Should have raised an error")
        except AssertionError:
            raise
        except Exception as e:
            err = str(e).lower()
            assert any(w in err for w in ("not found", "error", "nonexistent", "no such"))
