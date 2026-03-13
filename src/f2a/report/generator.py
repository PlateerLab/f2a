"""HTML 리포트 생성 모듈."""

from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

from f2a.utils.logging import get_logger

logger = get_logger(__name__)


def _fig_to_base64(fig: plt.Figure) -> str:
    """matplotlib Figure를 base64 인코딩된 PNG 문자열로 변환합니다."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return encoded


def _df_to_html(df: pd.DataFrame, max_rows: int = 50) -> str:
    """DataFrame을 HTML 테이블 문자열로 변환합니다."""
    return df.head(max_rows).to_html(classes="table", border=0, float_format="%.4f")


class ReportGenerator:
    """분석 결과를 HTML 리포트로 생성합니다."""

    def generate_html(
        self,
        dataset_name: str,
        schema_summary: dict[str, Any],
        stats_df: pd.DataFrame,
        figures: dict[str, plt.Figure],
        warnings: list[str] | None = None,
    ) -> str:
        """HTML 리포트 문자열을 생성합니다.

        Args:
            dataset_name: 데이터셋 이름.
            schema_summary: 스키마 요약 딕셔너리.
            stats_df: 요약 통계 DataFrame.
            figures: 이름 → Figure 매핑.
            warnings: 경고 메시지 리스트.

        Returns:
            HTML 문자열.
        """
        warnings = warnings or []

        # Figure를 base64로 변환
        figure_html_parts: list[str] = []
        for name, fig in figures.items():
            b64 = _fig_to_base64(fig)
            figure_html_parts.append(
                f'<div class="chart-container">'
                f'<h3>{name}</h3>'
                f'<img src="data:image/png;base64,{b64}" alt="{name}" />'
                f"</div>"
            )

        figures_section = "\n".join(figure_html_parts)
        stats_html = _df_to_html(stats_df) if not stats_df.empty else "<p>통계 데이터 없음</p>"
        warnings_html = "".join(f"<li>{w}</li>" for w in warnings)

        html = f"""<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>f2a 분석 리포트 — {dataset_name}</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
               line-height: 1.6; color: #333; max-width: 1200px; margin: 0 auto; padding: 20px; }}
        h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; margin-bottom: 20px; }}
        h2 {{ color: #34495e; margin-top: 30px; margin-bottom: 15px; }}
        h3 {{ color: #7f8c8d; margin-bottom: 10px; }}
        .overview {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                     gap: 15px; margin: 20px 0; }}
        .stat-card {{ background: #f8f9fa; border-radius: 8px; padding: 15px; text-align: center; }}
        .stat-card .value {{ font-size: 2em; font-weight: bold; color: #3498db; }}
        .stat-card .label {{ font-size: 0.9em; color: #7f8c8d; }}
        .table {{ width: 100%; border-collapse: collapse; margin: 15px 0; font-size: 0.9em; }}
        .table th, .table td {{ padding: 8px 12px; text-align: left; border-bottom: 1px solid #ddd; }}
        .table th {{ background: #f8f9fa; font-weight: 600; }}
        .table tr:hover {{ background: #f1f3f5; }}
        .chart-container {{ margin: 20px 0; text-align: center; }}
        .chart-container img {{ max-width: 100%; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        .warnings {{ background: #fff3cd; border: 1px solid #ffc107; border-radius: 8px;
                     padding: 15px; margin: 20px 0; }}
        .warnings li {{ margin-left: 20px; }}
        footer {{ text-align: center; margin-top: 40px; padding: 20px; color: #aaa; font-size: 0.85em; }}
    </style>
</head>
<body>
    <h1>f2a 분석 리포트</h1>
    <p>데이터셋: <strong>{dataset_name}</strong></p>

    <h2>개요</h2>
    <div class="overview">
        <div class="stat-card">
            <div class="value">{schema_summary.get('rows', 'N/A'):,}</div>
            <div class="label">행 (Rows)</div>
        </div>
        <div class="stat-card">
            <div class="value">{schema_summary.get('columns', 'N/A')}</div>
            <div class="label">열 (Columns)</div>
        </div>
        <div class="stat-card">
            <div class="value">{schema_summary.get('numeric', 0)}</div>
            <div class="label">수치형</div>
        </div>
        <div class="stat-card">
            <div class="value">{schema_summary.get('categorical', 0)}</div>
            <div class="label">범주형</div>
        </div>
        <div class="stat-card">
            <div class="value">{schema_summary.get('memory_mb', 0)} MB</div>
            <div class="label">메모리</div>
        </div>
    </div>

    <h2>요약 통계</h2>
    {stats_html}

    <h2>시각화</h2>
    {figures_section}

    {"<h2>경고</h2><div class='warnings'><ul>" + warnings_html + "</ul></div>" if warnings else ""}

    <footer>
        Generated by <strong>f2a</strong> (File to Analysis)
    </footer>
</body>
</html>"""
        return html

    def save_html(
        self,
        output_path: str | Path,
        **kwargs: Any,
    ) -> Path:
        """HTML 리포트를 파일로 저장합니다.

        Args:
            output_path: 출력 파일 경로.
            **kwargs: ``generate_html``에 전달할 인자.

        Returns:
            저장된 파일 경로.
        """
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        html = self.generate_html(**kwargs)
        path.write_text(html, encoding="utf-8")
        logger.info("리포트 저장: %s", path)
        return path
