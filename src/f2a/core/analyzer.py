"""분석 오케스트레이터 — 전체 분석 파이프라인을 조율합니다."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

from f2a.core.loader import DataLoader
from f2a.core.schema import DataSchema, infer_schema
from f2a.stats.descriptive import DescriptiveStats
from f2a.stats.distribution import DistributionStats
from f2a.stats.correlation import CorrelationStats
from f2a.stats.missing import MissingStats
from f2a.viz.plots import BasicPlotter
from f2a.viz.corr_plots import CorrelationPlotter
from f2a.viz.missing_plots import MissingPlotter
from f2a.report.generator import ReportGenerator
from f2a.utils.logging import get_logger
from f2a.utils.validators import validate_source

logger = get_logger(__name__)


@dataclass
class StatsResult:
    """통계 분석 결과 컨테이너."""

    summary: pd.DataFrame
    numeric_summary: pd.DataFrame
    categorical_summary: pd.DataFrame
    correlation_matrix: pd.DataFrame
    missing_info: pd.DataFrame
    distribution_info: pd.DataFrame

    def get_numeric_summary(self) -> pd.DataFrame:
        """수치형 컬럼 요약을 반환합니다."""
        return self.numeric_summary

    def get_categorical_summary(self) -> pd.DataFrame:
        """범주형 컬럼 요약을 반환합니다."""
        return self.categorical_summary


@dataclass
class VizResult:
    """시각화 결과 컨테이너."""

    _df: pd.DataFrame
    _schema: DataSchema
    _figures: dict[str, plt.Figure] = field(default_factory=dict)

    def plot_distributions(self) -> plt.Figure:
        """수치형 컬럼 분포 히스토그램을 반환합니다."""
        plotter = BasicPlotter(self._df, self._schema)
        fig = plotter.histograms()
        self._figures["distributions"] = fig
        return fig

    def plot_boxplots(self) -> plt.Figure:
        """수치형 컬럼 박스플롯을 반환합니다."""
        plotter = BasicPlotter(self._df, self._schema)
        fig = plotter.boxplots()
        self._figures["boxplots"] = fig
        return fig

    def plot_correlation(self, method: str = "pearson") -> plt.Figure:
        """상관계수 히트맵을 반환합니다."""
        plotter = CorrelationPlotter(self._df, self._schema)
        fig = plotter.heatmap(method=method)
        self._figures["correlation"] = fig
        return fig

    def plot_missing(self) -> plt.Figure:
        """결측치 바 차트를 반환합니다."""
        plotter = MissingPlotter(self._df, self._schema)
        fig = plotter.bar()
        self._figures["missing"] = fig
        return fig


@dataclass
class AnalysisReport:
    """분석 결과를 담는 최상위 컨테이너.

    Attributes:
        dataset_name: 데이터셋 이름.
        shape: ``(행, 열)`` 튜플.
        schema: 데이터 스키마.
        stats: 통계 분석 결과.
        viz: 시각화 접근 객체.
        warnings: 분석 중 발견된 경고 목록.
    """

    dataset_name: str
    shape: tuple[int, int]
    schema: DataSchema
    stats: StatsResult
    viz: VizResult
    warnings: list[str] = field(default_factory=list)

    def show(self) -> None:
        """콘솔에 분석 요약을 출력합니다."""
        sep = "=" * 60
        print(sep)
        print(f"  f2a 분석 리포트: {self.dataset_name}")
        print(sep)
        print(f"\n  행: {self.shape[0]:,}  |  열: {self.shape[1]}")
        print(f"  메모리: {self.schema.memory_usage_mb} MB")
        print(f"\n  수치형: {len(self.schema.numeric_columns)}개")
        print(f"  범주형: {len(self.schema.categorical_columns)}개")
        print(f"  텍스트: {len(self.schema.text_columns)}개")
        print(f"  일시형: {len(self.schema.datetime_columns)}개")

        print(f"\n{'─' * 60}")
        print("  요약 통계:")
        print(self.stats.summary.to_string())

        if self.warnings:
            print(f"\n{'─' * 60}")
            print("  ⚠ 경고:")
            for w in self.warnings:
                print(f"    • {w}")

        print(sep)

    def to_html(self, output_dir: str = ".") -> Path:
        """HTML 리포트를 생성하여 파일로 저장합니다.

        Args:
            output_dir: 출력 디렉토리 경로.

        Returns:
            저장된 HTML 파일 경로.
        """
        # 시각화 생성
        figures: dict[str, plt.Figure] = {}
        try:
            figures["분포 히스토그램"] = self.viz.plot_distributions()
        except Exception:
            pass
        try:
            figures["박스플롯"] = self.viz.plot_boxplots()
        except Exception:
            pass
        try:
            figures["상관 히트맵"] = self.viz.plot_correlation()
        except Exception:
            pass
        try:
            figures["결측치 현황"] = self.viz.plot_missing()
        except Exception:
            pass

        generator = ReportGenerator()
        output_path = Path(output_dir) / f"{self.dataset_name}_report.html"
        generator.save_html(
            output_path=output_path,
            dataset_name=self.dataset_name,
            schema_summary=self.schema.summary_dict(),
            stats_df=self.stats.summary,
            figures=figures,
            warnings=self.warnings,
        )
        return output_path

    def to_dict(self) -> dict[str, Any]:
        """분석 결과를 딕셔너리로 반환합니다."""
        return {
            "dataset_name": self.dataset_name,
            "shape": self.shape,
            "schema": self.schema.summary_dict(),
            "stats_summary": self.stats.summary.to_dict(),
            "correlation_matrix": self.stats.correlation_matrix.to_dict()
            if not self.stats.correlation_matrix.empty
            else {},
            "warnings": self.warnings,
        }


class Analyzer:
    """분석 파이프라인을 오케스트레이션합니다.

    Example:
        >>> analyzer = Analyzer()
        >>> report = analyzer.run("data.csv")
        >>> report.show()
    """

    def __init__(self) -> None:
        self._loader = DataLoader()

    def run(self, source: str, **kwargs: Any) -> AnalysisReport:
        """전체 분석 파이프라인을 실행합니다.

        Args:
            source: 데이터 소스 (파일 경로 또는 HF 주소).
            **kwargs: 로더에 전달할 추가 인자.

        Returns:
            :class:`AnalysisReport` 인스턴스.
        """
        source = validate_source(source)
        logger.info("분석 시작: %s", source)

        # 1. 데이터 로딩
        df = self._loader.load(source, **kwargs)

        # 2. 스키마 추론
        schema = infer_schema(df)
        logger.info("스키마 추론 완료: %s", schema.summary_dict())

        # 3. 통계 분석
        warnings: list[str] = []
        stats = self._compute_stats(df, schema, warnings)

        # 4. 결과 조립
        dataset_name = Path(source).stem if "/" not in source or "://" not in source else source
        viz = VizResult(_df=df, _schema=schema)

        report = AnalysisReport(
            dataset_name=dataset_name,
            shape=(len(df), len(df.columns)),
            schema=schema,
            stats=stats,
            viz=viz,
            warnings=warnings,
        )

        logger.info("분석 완료: %s", source)
        return report

    def _compute_stats(
        self,
        df: pd.DataFrame,
        schema: DataSchema,
        warnings: list[str],
    ) -> StatsResult:
        """모든 통계 분석을 수행합니다."""
        desc = DescriptiveStats(df, schema)
        dist = DistributionStats(df, schema)
        corr = CorrelationStats(df, schema)
        miss = MissingStats(df, schema)

        # 기술 통계
        summary = desc.summary()
        numeric_summary = desc.numeric_summary()
        categorical_summary = desc.categorical_summary()

        # 상관 분석
        correlation_matrix = corr.pearson()
        high_corrs = corr.high_correlations(threshold=0.9)
        for col_a, col_b, val in high_corrs:
            warnings.append(f"높은 상관: {col_a} ↔ {col_b} (r={val})")

        # 결측치
        missing_info = miss.column_summary()
        total_missing = miss.total_missing_ratio()
        if total_missing > 0.1:
            warnings.append(f"전체 결측 비율이 {total_missing * 100:.1f}%로 높습니다.")

        # 분포
        distribution_info = dist.analyze()

        return StatsResult(
            summary=summary,
            numeric_summary=numeric_summary,
            categorical_summary=categorical_summary,
            correlation_matrix=correlation_matrix,
            missing_info=missing_info,
            distribution_info=distribution_info,
        )


def analyze(source: str, **kwargs: Any) -> AnalysisReport:
    """데이터 소스를 분석하여 리포트를 반환합니다.

    이 함수는 ``f2a`` 의 주요 진입점(entry point)입니다.

    Args:
        source: 파일 경로 또는 HuggingFace 데이터셋 주소.
            - 파일: ``"data.csv"``, ``"data.json"``, ``"data.parquet"``
            - HuggingFace: ``"hf://imdb"``, ``"hf://squad"``
        **kwargs: 데이터 로더에 전달할 추가 인자.

    Returns:
        :class:`AnalysisReport` — 통계, 시각화, 리포트 접근 객체.

    Example:
        >>> import f2a
        >>> report = f2a.analyze("sales.csv")
        >>> report.show()
        >>> report.to_html("output/")
    """
    analyzer = Analyzer()
    return analyzer.run(source, **kwargs)
