"""
HTML report generator for f2a.

Produces a self-contained single-file HTML report with embedded charts
(base64 PNG) and interactive navigation.
"""

from __future__ import annotations

import base64
import io
import json
from pathlib import Path
from typing import Any, Optional

from f2a.report.i18n import t

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_VIZ = True
except ImportError:
    HAS_VIZ = False


class ReportGenerator:
    """Generates a self-contained HTML report from an AnalysisReport."""

    def __init__(self, lang: str = "en"):
        self.lang = lang

    @staticmethod
    def _get_version() -> str:
        from f2a._version import __version__
        return __version__

    def save_html(self, output_path: Path, report: Any) -> None:
        """Write the report as a single HTML file."""
        html = self._build_html(report)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html)

    # ── Main HTML builder ────────────────────────────────────────

    def _build_html(self, report: Any) -> str:
        sections_html = []

        # Overview
        sections_html.append(self._section_overview(report))

        # Schema
        if "columns" in report.schema:
            sections_html.append(self._section_schema(report))

        # Preprocessing
        if report.preprocessing:
            sections_html.append(self._section_preprocessing(report))

        # Each analysis section
        section_order = [
            "descriptive", "missing", "distribution", "outlier",
            "correlation", "categorical", "duplicates", "quality",
            "feature_importance", "pca", "statistical_tests",
            "clustering", "advanced_anomaly", "advanced_correlation",
            "advanced_distribution", "advanced_dimreduction",
            "feature_insights", "insight_engine", "column_role",
            "cross_analysis", "ml_readiness",
        ]

        for section_key in section_order:
            if section_key in report.results:
                title = t(section_key, self.lang)
                data = report.results[section_key]
                sections_html.append(
                    self._generic_section(section_key, title, data)
                )

        nav_items = self._build_nav(report, section_order)
        body = "\n".join(sections_html)

        return self._wrap_html(
            title=f"{t('report_title', self.lang)} — {Path(report.source).stem}",
            nav=nav_items,
            body=body,
        )

    # ── Overview section ─────────────────────────────────────────

    def _section_overview(self, report: Any) -> str:
        schema = report.schema
        n_rows = schema.get("n_rows", "?")
        n_cols = schema.get("n_cols", "?")
        mem_bytes = schema.get("memory_usage_bytes", 0)
        mem_str = self._format_bytes(mem_bytes)

        cards = f"""
        <div class="card-grid">
            <div class="card">
                <div class="card-label">{t('rows', self.lang)}</div>
                <div class="card-value">{n_rows:,}</div>
            </div>
            <div class="card">
                <div class="card-label">{t('columns', self.lang)}</div>
                <div class="card-value">{n_cols}</div>
            </div>
            <div class="card">
                <div class="card-label">{t('memory', self.lang)}</div>
                <div class="card-value">{mem_str}</div>
            </div>
            <div class="card">
                <div class="card-label">{t('duration', self.lang)}</div>
                <div class="card-value">{report.analysis_duration_sec:.2f}s</div>
            </div>
        </div>
        """

        # ML readiness badge
        ml_html = ""
        if "ml_readiness" in report.results:
            ml = report.results["ml_readiness"]
            grade = ml.get("grade", "?")
            score = ml.get("overall_score", 0) * 100
            color = {"A": "#22c55e", "B": "#84cc16", "C": "#eab308", "D": "#f97316", "F": "#ef4444"}.get(grade, "#888")
            ml_html = f"""
            <div class="ml-badge" style="background:{color}">
                {t('ml_readiness', self.lang)}: {grade} ({score:.0f}%)
            </div>
            """

        return f"""
        <section id="overview">
            <h2>{t('overview', self.lang)}</h2>
            {cards}
            {ml_html}
        </section>
        """

    # ── Schema section ───────────────────────────────────────────

    def _section_schema(self, report: Any) -> str:
        columns = report.schema.get("columns", [])
        rows = ""
        for col in columns:
            missing_pct = col.get("missing_ratio", 0) * 100
            bar_color = "#22c55e" if missing_pct < 5 else "#eab308" if missing_pct < 30 else "#ef4444"
            rows += f"""
            <tr>
                <td><strong>{col['name']}</strong></td>
                <td><code>{col.get('dtype', '')}</code></td>
                <td>{col.get('inferred_type', '')}</td>
                <td class="right">{col.get('n_unique', '')}</td>
                <td class="right">
                    {col.get('n_missing', 0)}
                    <span class="pct">({missing_pct:.1f}%)</span>
                    <div class="bar" style="width:{min(missing_pct, 100):.0f}%;background:{bar_color}"></div>
                </td>
            </tr>
            """

        return f"""
        <section id="schema">
            <h2>{t('schema', self.lang)}</h2>
            <div class="table-wrap">
                <table>
                    <thead>
                        <tr>
                            <th>Column</th><th>DType</th><th>Inferred</th>
                            <th class="right">Unique</th><th class="right">Missing</th>
                        </tr>
                    </thead>
                    <tbody>{rows}</tbody>
                </table>
            </div>
        </section>
        """

    # ── Preprocessing section ────────────────────────────────────

    def _section_preprocessing(self, report: Any) -> str:
        pp = report.preprocessing
        if not pp:
            return ""

        items = [
            f"Rows: {pp.get('rows_before', '?')} → {pp.get('rows_after', '?')}",
            f"Cols: {pp.get('cols_before', '?')} → {pp.get('cols_after', '?')}",
            f"Duplicates: {pp.get('duplicate_row_count', 0)} ({pp.get('duplicate_row_ratio', 0)*100:.1f}%)",
        ]

        const_cols = pp.get("constant_columns", [])
        if const_cols:
            items.append(f"Constant columns: {', '.join(const_cols)}")

        id_cols = pp.get("id_like_columns", [])
        if id_cols:
            items.append(f"ID-like columns: {', '.join(id_cols)}")

        li = "".join(f"<li>{item}</li>" for item in items)

        return f"""
        <section id="preprocessing">
            <h2>{t('preprocessing', self.lang)}</h2>
            <ul>{li}</ul>
        </section>
        """

    # ── Generic JSON-to-HTML section ─────────────────────────────

    def _generic_section(self, key: str, title: str, data: Any) -> str:
        """Render any analysis section as collapsible JSON + summary tables."""
        content_parts = []

        # Special renderers
        if key == "insight_engine":
            content_parts.append(self._render_insights(data))
        elif key == "ml_readiness":
            content_parts.append(self._render_ml_readiness(data))
        elif key == "quality":
            content_parts.append(self._render_quality(data))
        elif key == "descriptive":
            content_parts.append(self._render_descriptive(data))
        elif key == "missing":
            content_parts.append(self._render_missing(data))
        else:
            # Fallback: render as pretty-printed JSON
            content_parts.append(self._render_json(data))

        content = "\n".join(content_parts)

        return f"""
        <section id="{key}">
            <h2>{title}</h2>
            {content}
        </section>
        """

    # ── Specialized renderers ────────────────────────────────────

    def _render_insights(self, data: dict) -> str:
        summary = data.get("summary", {})
        insights = data.get("insights", [])

        header = f"""
        <div class="insight-summary">
            <span class="badge badge-critical">{summary.get('critical', 0)} {t('critical', self.lang)}</span>
            <span class="badge badge-warning">{summary.get('warning', 0)} {t('warning', self.lang)}</span>
            <span class="badge badge-info">{summary.get('info', 0)} {t('info', self.lang)}</span>
        </div>
        """

        rows = ""
        for ins in insights:
            sev = ins.get("severity", "Info")
            sev_class = sev.lower()
            rows += f"""
            <tr class="insight-{sev_class}">
                <td><span class="badge badge-{sev_class}">{sev}</span></td>
                <td>{ins.get('column', '—')}</td>
                <td>{ins.get('message', '')}</td>
                <td>{ins.get('recommendation', '')}</td>
            </tr>
            """

        table = f"""
        <div class="table-wrap">
            <table>
                <thead><tr><th>Severity</th><th>Column</th><th>Message</th><th>Recommendation</th></tr></thead>
                <tbody>{rows}</tbody>
            </table>
        </div>
        """
        return header + table

    def _render_ml_readiness(self, data: dict) -> str:
        grade = data.get("grade", "?")
        score = data.get("overall_score", 0)
        dims = data.get("dimensions", [])
        recs = data.get("recommendations", [])

        # Dimensions table
        dim_rows = ""
        for d in dims:
            bar_width = d.get("score", 0) * 100
            color = "#22c55e" if bar_width >= 80 else "#eab308" if bar_width >= 60 else "#ef4444"
            dim_rows += f"""
            <tr>
                <td><strong>{d['name']}</strong></td>
                <td>{d.get('score', 0)*100:.0f}%
                    <div class="bar" style="width:{bar_width:.0f}%;background:{color}"></div>
                </td>
                <td>{d.get('detail', '')}</td>
            </tr>
            """

        # Recommendations
        rec_html = ""
        if recs:
            rec_items = "".join(f"<li>{r}</li>" for r in recs)
            rec_html = f"<h3>{t('recommendations', self.lang)}</h3><ul>{rec_items}</ul>"

        return f"""
        <div class="grade-badge grade-{grade.lower()}">{grade} ({score*100:.0f}%)</div>
        <div class="table-wrap">
            <table>
                <thead><tr><th>Dimension</th><th>Score</th><th>Detail</th></tr></thead>
                <tbody>{dim_rows}</tbody>
            </table>
        </div>
        {rec_html}
        """

    def _render_quality(self, data: dict) -> str:
        overall = data.get("overall_score", 0)
        dims = data.get("dimensions", [])

        dim_rows = ""
        for d in dims:
            s = d.get("score", 0)
            bar_width = s * 100
            color = "#22c55e" if bar_width >= 80 else "#eab308" if bar_width >= 60 else "#ef4444"
            dim_rows += f"""
            <tr>
                <td><strong>{d.get('name', '')}</strong></td>
                <td>{s*100:.0f}%
                    <div class="bar" style="width:{bar_width:.0f}%;background:{color}"></div>
                </td>
            </tr>
            """

        return f"""
        <div class="grade-badge">{t('overall_score', self.lang)}: {overall*100:.0f}%</div>
        <div class="table-wrap">
            <table>
                <thead><tr><th>Dimension</th><th>Score</th></tr></thead>
                <tbody>{dim_rows}</tbody>
            </table>
        </div>
        """

    def _render_descriptive(self, data: dict) -> str:
        num = data.get("numeric", [])
        cat = data.get("categorical", [])
        parts = []

        if num:
            rows = ""
            for col_data in num[:30]:
                rows += f"""
                <tr>
                    <td><strong>{col_data.get('column', '')}</strong></td>
                    <td class="right">{col_data.get('count', '')}</td>
                    <td class="right">{self._fmt(col_data.get('mean'))}</td>
                    <td class="right">{self._fmt(col_data.get('std'))}</td>
                    <td class="right">{self._fmt(col_data.get('min'))}</td>
                    <td class="right">{self._fmt(col_data.get('median'))}</td>
                    <td class="right">{self._fmt(col_data.get('max'))}</td>
                    <td class="right">{self._fmt(col_data.get('skewness'))}</td>
                    <td class="right">{self._fmt(col_data.get('kurtosis'))}</td>
                </tr>
                """
            parts.append(f"""
            <h3>Numeric</h3>
            <div class="table-wrap">
                <table>
                    <thead><tr>
                        <th>Column</th><th class="right">Count</th>
                        <th class="right">Mean</th><th class="right">Std</th>
                        <th class="right">Min</th><th class="right">Median</th>
                        <th class="right">Max</th><th class="right">Skew</th>
                        <th class="right">Kurt</th>
                    </tr></thead>
                    <tbody>{rows}</tbody>
                </table>
            </div>
            """)

        if cat:
            rows = ""
            for col_data in cat[:30]:
                rows += f"""
                <tr>
                    <td><strong>{col_data.get('column', '')}</strong></td>
                    <td class="right">{col_data.get('count', '')}</td>
                    <td class="right">{col_data.get('unique', '')}</td>
                    <td>{col_data.get('top', '')}</td>
                    <td class="right">{col_data.get('freq', '')}</td>
                </tr>
                """
            parts.append(f"""
            <h3>Categorical</h3>
            <div class="table-wrap">
                <table>
                    <thead><tr>
                        <th>Column</th><th class="right">Count</th>
                        <th class="right">Unique</th><th>Top</th>
                        <th class="right">Freq</th>
                    </tr></thead>
                    <tbody>{rows}</tbody>
                </table>
            </div>
            """)

        return "\n".join(parts) if parts else self._render_json(data)

    def _render_missing(self, data: dict) -> str:
        per_col = data.get("per_column", [])
        if not per_col:
            return "<p>No missing values detected.</p>"

        rows = ""
        for col_data in per_col:
            n = col_data.get("n_missing", 0)
            ratio = col_data.get("missing_ratio", 0)
            if n > 0:
                bar_width = ratio * 100
                color = "#22c55e" if bar_width < 5 else "#eab308" if bar_width < 30 else "#ef4444"
                rows += f"""
                <tr>
                    <td><strong>{col_data.get('column', '')}</strong></td>
                    <td class="right">{n}</td>
                    <td class="right">{ratio*100:.1f}%
                        <div class="bar" style="width:{min(bar_width, 100):.0f}%;background:{color}"></div>
                    </td>
                </tr>
                """

        if not rows:
            return "<p>No missing values detected.</p>"

        return f"""
        <div class="table-wrap">
            <table>
                <thead><tr><th>Column</th><th class="right">Missing</th><th class="right">Ratio</th></tr></thead>
                <tbody>{rows}</tbody>
            </table>
        </div>
        """

    def _render_json(self, data: Any) -> str:
        """Fallback: render as collapsible JSON."""
        json_str = json.dumps(data, indent=2, ensure_ascii=False, default=str)
        # Truncate very long JSON
        if len(json_str) > 50_000:
            json_str = json_str[:50_000] + "\n... (truncated)"
        return f"""
        <details>
            <summary>Raw JSON data</summary>
            <pre class="json-pre"><code>{self._escape_html(json_str)}</code></pre>
        </details>
        """

    # ── Navigation ───────────────────────────────────────────────

    def _build_nav(self, report: Any, section_order: list[str]) -> str:
        items = [f'<a href="#overview">{t("overview", self.lang)}</a>']

        if "columns" in report.schema:
            items.append(f'<a href="#schema">{t("schema", self.lang)}</a>')

        if report.preprocessing:
            items.append(f'<a href="#preprocessing">{t("preprocessing", self.lang)}</a>')

        for key in section_order:
            if key in report.results:
                title = t(key, self.lang)
                items.append(f'<a href="#{key}">{title}</a>')

        return "\n".join(items)

    # ── Full HTML wrapper ────────────────────────────────────────

    def _wrap_html(self, title: str, nav: str, body: str) -> str:
        return f"""<!DOCTYPE html>
<html lang="{self.lang}">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>{self._escape_html(title)}</title>
    <style>{self._css()}</style>
</head>
<body>
    <nav class="sidebar">{nav}</nav>
    <main class="content">
        <h1>{self._escape_html(title)}</h1>
        <p class="subtitle">{t('generated_by', self.lang)} v{self._get_version()}</p>
        {body}
    </main>
    <script>{self._js()}</script>
</body>
</html>"""

    # ── CSS ──────────────────────────────────────────────────────

    def _css(self) -> str:
        return """
:root {
    --bg: #0d1117; --surface: #161b22; --border: #30363d;
    --text: #e6edf3; --text2: #8b949e; --accent: #58a6ff;
    --success: #22c55e; --warn: #eab308; --danger: #ef4444;
    --font: 'Segoe UI', -apple-system, BlinkMacSystemFont, sans-serif;
    --mono: 'Cascadia Code', 'Fira Code', 'Consolas', monospace;
}
* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: var(--font); background: var(--bg); color: var(--text); display: flex; min-height: 100vh; }
.sidebar {
    position: fixed; top: 0; left: 0; width: 240px; height: 100vh;
    background: var(--surface); border-right: 1px solid var(--border);
    padding: 1rem 0; overflow-y: auto; z-index: 100;
}
.sidebar a {
    display: block; padding: 8px 16px; color: var(--text2);
    text-decoration: none; font-size: 13px; transition: all .15s;
}
.sidebar a:hover { color: var(--accent); background: rgba(88,166,255,.08); }
.content { margin-left: 240px; padding: 2rem 3rem; max-width: 1200px; width: 100%; }
h1 { font-size: 1.8rem; margin-bottom: .25rem; }
h2 { font-size: 1.3rem; margin-top: 2rem; margin-bottom: 1rem; padding-bottom: .5rem; border-bottom: 1px solid var(--border); }
h3 { font-size: 1.1rem; margin-top: 1.5rem; margin-bottom: .75rem; color: var(--text2); }
.subtitle { color: var(--text2); margin-bottom: 2rem; }
section { margin-bottom: 2rem; }
.card-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(180px, 1fr)); gap: 1rem; margin: 1rem 0; }
.card {
    background: var(--surface); border: 1px solid var(--border); border-radius: 8px;
    padding: 1rem; text-align: center;
}
.card-label { font-size: .8rem; color: var(--text2); margin-bottom: .25rem; }
.card-value { font-size: 1.4rem; font-weight: 600; }
.table-wrap { overflow-x: auto; margin: .75rem 0; }
table { width: 100%; border-collapse: collapse; font-size: .85rem; }
thead { background: var(--surface); }
th, td { padding: 8px 12px; text-align: left; border-bottom: 1px solid var(--border); }
th { font-weight: 600; color: var(--text2); font-size: .8rem; text-transform: uppercase; letter-spacing: .5px; }
tr:hover { background: rgba(88,166,255,.04); }
.right { text-align: right; }
.pct { color: var(--text2); font-size: .8em; margin-left: 4px; }
.bar { height: 3px; border-radius: 2px; margin-top: 4px; transition: width .3s; }
code { font-family: var(--mono); font-size: .85em; background: rgba(110,118,129,.15); padding: 2px 6px; border-radius: 4px; }
pre.json-pre { background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 1rem; overflow-x: auto; font-size: .8rem; max-height: 400px; overflow-y: auto; }
details { margin: .75rem 0; }
details summary { cursor: pointer; color: var(--accent); font-size: .9rem; }
ul { padding-left: 1.5rem; margin: .5rem 0; }
li { margin: .25rem 0; }
.badge {
    display: inline-block; padding: 3px 10px; border-radius: 12px; font-size: .75rem; font-weight: 600;
}
.badge-critical { background: rgba(239,68,68,.15); color: var(--danger); }
.badge-warning { background: rgba(234,179,8,.15); color: var(--warn); }
.badge-info { background: rgba(88,166,255,.15); color: var(--accent); }
.insight-summary { display: flex; gap: 1rem; margin-bottom: 1rem; }
.ml-badge { display: inline-block; padding: 6px 16px; border-radius: 8px; color: #fff; font-weight: 700; margin: .5rem 0; }
.grade-badge { font-size: 1.2rem; font-weight: 700; margin: .5rem 0; }
.grade-a { color: var(--success); }
.grade-b { color: #84cc16; }
.grade-c { color: var(--warn); }
.grade-d { color: #f97316; }
.grade-f { color: var(--danger); }
@media (max-width: 768px) {
    .sidebar { width: 100%; height: auto; position: relative; display: flex; flex-wrap: wrap; }
    .sidebar a { padding: 6px 12px; }
    .content { margin-left: 0; padding: 1rem; }
}
"""

    # ── JS ───────────────────────────────────────────────────────

    def _js(self) -> str:
        return """
// Smooth scroll and active state
document.querySelectorAll('.sidebar a').forEach(a => {
    a.addEventListener('click', e => {
        e.preventDefault();
        const target = document.querySelector(a.getAttribute('href'));
        if (target) target.scrollIntoView({ behavior: 'smooth', block: 'start' });
    });
});
"""

    # ── Utilities ────────────────────────────────────────────────

    @staticmethod
    def _escape_html(text: str) -> str:
        return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")

    @staticmethod
    def _format_bytes(n: int) -> str:
        for unit in ("B", "KB", "MB", "GB"):
            if n < 1024:
                return f"{n:.1f} {unit}" if unit != "B" else f"{n} {unit}"
            n /= 1024
        return f"{n:.1f} TB"

    @staticmethod
    def _fmt(val: Any) -> str:
        if val is None:
            return "—"
        if isinstance(val, float):
            if abs(val) >= 1e6:
                return f"{val:.2e}"
            return f"{val:.4f}"
        return str(val)

    @staticmethod
    def fig_to_base64(fig) -> str:
        """Convert a matplotlib figure to a base64-encoded PNG."""
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100, bbox_inches="tight", facecolor="#0d1117")
        buf.seek(0)
        b64 = base64.b64encode(buf.read()).decode("utf-8")
        plt.close(fig)
        return f"data:image/png;base64,{b64}"
