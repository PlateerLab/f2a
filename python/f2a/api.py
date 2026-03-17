"""
Public API for f2a -- mirrors the original f2a interface.

    report = f2a.analyze("data.csv", config=AnalysisConfig(advanced=False))
    report.show()
    report.to_html("./output")
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from rich.console import Console
from rich.table import Table

# Import the Rust _core extension
from f2a import _core


# ─── AnalysisConfig ──────────────────────────────────────────────────

@dataclass
class AnalysisConfig:
    """Configuration for analysis. Mirrors the Rust AnalysisConfig."""

    # Basic toggles
    descriptive: bool = True
    correlation: bool = True
    distribution: bool = True
    missing: bool = True
    outlier: bool = True
    categorical: bool = True
    feature_importance: bool = True
    pca: bool = True
    duplicates: bool = True
    quality: bool = True
    preprocessing: bool = True

    # Advanced toggles
    advanced: bool = True
    advanced_distribution: bool = True
    advanced_correlation: bool = True
    clustering: bool = True
    advanced_dimreduction: bool = True
    feature_insights: bool = True
    advanced_anomaly: bool = True
    statistical_tests: bool = True
    data_profiling: bool = True
    insight_engine: bool = True
    cross_analysis: bool = True
    column_role: bool = True
    ml_readiness: bool = True

    # Parameters
    outlier_threshold: float = 1.5
    outlier_method: str = "iqr"
    correlation_threshold: float = 0.9
    pca_max_components: int = 10
    max_categories: int = 50
    max_plot_columns: int = 20
    max_cluster_k: int = 10
    tsne_perplexity: float = 30.0
    bootstrap_iterations: int = 1000
    max_sample_for_advanced: int = 5000
    n_distribution_fits: int = 7

    def to_json(self) -> str:
        """Serialize to JSON for the Rust core."""
        return json.dumps(asdict(self))

    @classmethod
    def minimal(cls) -> "AnalysisConfig":
        """Only descriptive statistics."""
        cfg_json = _core.minimal_config()
        return cls._from_json(cfg_json)

    @classmethod
    def fast(cls) -> "AnalysisConfig":
        """Skip heavy analyses (PCA, feature importance, all advanced)."""
        cfg_json = _core.fast_config()
        return cls._from_json(cfg_json)

    @classmethod
    def basic_only(cls) -> "AnalysisConfig":
        """All basic on, all advanced off."""
        cfg_json = _core.basic_only_config()
        return cls._from_json(cfg_json)

    @classmethod
    def _from_json(cls, json_str: str) -> "AnalysisConfig":
        d = json.loads(json_str)
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ─── AnalysisReport ──────────────────────────────────────────────────

@dataclass
class AnalysisReport:
    """Result of ``f2a.analyze()``."""

    source: str
    schema: dict
    config: dict
    results: dict
    preprocessing: Optional[dict] = None
    analysis_started_at: str = ""
    analysis_duration_sec: float = 0.0

    # ── Console output ───────────────────────────────────────────

    def show(self) -> None:
        """Print a rich summary to the console."""
        console = Console()

        # Header
        console.print(f"\n[bold cyan]=== f2a Analysis Report ===[/bold cyan]")
        console.print(f"  Source: [green]{self.source}[/green]")
        console.print(
            f"  Shape: {self.schema.get('n_rows', '?')} rows x {self.schema.get('n_cols', '?')} cols"
        )
        console.print(
            f"  Duration: {self.analysis_duration_sec:.2f}s\n"
        )

        # Schema overview
        if "columns" in self.schema:
            table = Table(title="Schema", show_lines=False)
            table.add_column("Column", style="bold")
            table.add_column("DType")
            table.add_column("Inferred")
            table.add_column("Unique", justify="right")
            table.add_column("Missing", justify="right")

            for col in self.schema["columns"][:30]:
                missing_str = f"{col.get('n_missing', 0)} ({col.get('missing_ratio', 0)*100:.1f}%)"
                table.add_row(
                    col["name"],
                    col.get("dtype", ""),
                    col.get("inferred_type", ""),
                    str(col.get("n_unique", "")),
                    missing_str,
                )
            console.print(table)

        # Result sections
        sections = list(self.results.keys())
        console.print(f"\n  [bold]Analysis sections:[/bold] {', '.join(sections)}")

        # Insight summary
        if "insight_engine" in self.results:
            ie = self.results["insight_engine"]
            summary = ie.get("summary", {})
            console.print(
                f"\n  [bold]Insights:[/bold] {summary.get('total', 0)} total "
                f"({summary.get('critical', 0)} critical, "
                f"{summary.get('warning', 0)} warning, "
                f"{summary.get('info', 0)} info)"
            )

        # ML readiness
        if "ml_readiness" in self.results:
            ml = self.results["ml_readiness"]
            console.print(
                f"  [bold]ML Readiness:[/bold] {ml.get('grade', '?')} "
                f"({ml.get('overall_score', 0)*100:.0f}%)"
            )

        # Quality
        if "quality" in self.results:
            q = self.results["quality"]
            console.print(
                f"  [bold]Quality Score:[/bold] {q.get('overall_score', 0)*100:.0f}%"
            )

        console.print()

    # ── HTML report ──────────────────────────────────────────────

    def to_html(self, output_dir: str = ".", lang: str = "en") -> Path:
        """Generate a self-contained HTML report.

        Parameters
        ----------
        output_dir : str
            Directory to write the HTML file (created if needed).
        lang : str
            Language code for the report ('en', 'ko', 'ja', 'zh', 'de', 'fr').

        Returns
        -------
        pathlib.Path
            Path to the generated HTML file.
        """
        from f2a.report.generator import ReportGenerator

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        dataset_name = Path(self.source).stem
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{dataset_name}_{timestamp}_report.html"
        filepath = output_path / filename

        generator = ReportGenerator(lang=lang)
        generator.save_html(
            output_path=filepath,
            report=self,
        )

        return filepath

    # ── Convenience accessors ────────────────────────────────────

    def get(self, section: str) -> Optional[dict]:
        """Get results for a named section."""
        return self.results.get(section)

    @property
    def sections(self) -> list[str]:
        """List of computed analysis sections."""
        return list(self.results.keys())


# ─── analyze() entry-point ───────────────────────────────────────────

def analyze(
    source: str,
    config: Optional[AnalysisConfig] = None,
) -> AnalysisReport:
    """Run a full analysis on a data file.

    Parameters
    ----------
    source : str
        Path to a data file (CSV, TSV, Parquet, JSON, JSONL, Feather).
    config : AnalysisConfig, optional
        Configuration overrides. Defaults to all analyses enabled.

    Returns
    -------
    AnalysisReport
        Rich object with ``.show()``, ``.to_html()``, and dict-like access.

    Examples
    --------
    >>> import f2a
    >>> report = f2a.analyze("data.csv")
    >>> report.show()
    >>> report.to_html("./output")
    """
    if config is None:
        config = AnalysisConfig()

    config_json = config.to_json()

    start = time.perf_counter()
    raw_json = _core.analyze(source, config_json)
    duration = time.perf_counter() - start

    raw = json.loads(raw_json)

    return AnalysisReport(
        source=raw.get("source", source),
        schema=raw.get("schema", {}),
        config=raw.get("config", {}),
        results=raw.get("results", {}),
        preprocessing=raw.get("preprocessing"),
        analysis_started_at=datetime.now().isoformat(),
        analysis_duration_sec=duration,
    )
