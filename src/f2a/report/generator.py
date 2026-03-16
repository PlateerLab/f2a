"""HTML report generation module.

Generates comprehensive single-page HTML reports with:
- Sticky navigation bar
- Data quality dashboard
- Preprocessing report
- Descriptive / distribution / correlation / missing / outlier / categorical /
  feature-importance / PCA / duplicate analysis sections
- Inline base64 charts
- Drag-to-scroll tables
"""

from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from f2a.core.config import AnalysisConfig
from f2a.utils.logging import get_logger

logger = get_logger(__name__)


# =====================================================================
#  Helpers
# =====================================================================

def _fig_to_base64(fig: plt.Figure) -> str:
    """Convert a matplotlib Figure to a base64 PNG string."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return encoded


# ── Metric tooltip descriptions ──────────────────────────────────────

_METRIC_TIPS: dict[str, str] = {
    # Descriptive
    "type": "Inferred data type of the column (numeric, categorical, text, datetime, boolean).",
    "count": "Number of non-null values in the column.",
    "missing": "Number of missing (null / NaN) values.",
    "missing_%": "Percentage of missing values = (missing / total rows) × 100.",
    "unique": "Number of distinct values in the column.",
    "mean": "Arithmetic mean = sum of values / count.",
    "median": "Middle value when data is sorted (50th percentile).",
    "std": "Standard deviation — measures spread around the mean. Larger = more dispersed.",
    "se": "Standard error of the mean = std / √n. Indicates precision of the sample mean.",
    "cv": "Coefficient of variation = std / |mean|. Unitless relative measure of variability.",
    "mad": "Median Absolute Deviation = median(|xi − median|). Robust measure of spread.",
    "min": "Minimum value in the column.",
    "max": "Maximum value in the column.",
    "range": "Range = max − min. Total spread of the data.",
    "p5": "5th percentile — 5% of data falls below this value.",
    "q1": "1st quartile (25th percentile) — 25% of data falls below this value.",
    "q3": "3rd quartile (75th percentile) — 75% of data falls below this value.",
    "p95": "95th percentile — 95% of data falls below this value.",
    "iqr": "Interquartile Range = Q3 − Q1. Middle 50% spread, used for outlier detection.",
    "skewness": "Skewness measures distribution asymmetry. 0 = symmetric, >0 = right-skewed, <0 = left-skewed.",
    "kurtosis": "Excess kurtosis measures tail heaviness. 0 = normal, >0 = heavy tails, <0 = light tails.",
    "top": "Most frequently occurring value in the column.",
    "freq": "Frequency count of the most common value.",
    # Distribution
    "n": "Number of non-null observations used for the distribution test.",
    "skew_type": "Interpretation of skewness: symmetric (|s|<0.5), moderate skew (0.5–1), high skew (>1).",
    "kurt_type": "Interpretation of kurtosis: mesokurtic (≈0), leptokurtic (>1, heavy tails), platykurtic (<−1, light tails).",
    "normality_test": "Primary normality test used (Shapiro-Wilk for n≤5000, D'Agostino-Pearson for larger).",
    "normality_p": "p-value of the primary normality test. p<0.05 → likely non-normal.",
    "is_normal_0.05": "True if p-value ≥ 0.05, meaning the null hypothesis of normality is not rejected at α=0.05.",
    "shapiro_p": "p-value from Shapiro-Wilk test. Best for small–medium samples (n≤5000).",
    "dagostino_p": "p-value from D'Agostino-Pearson test. Uses skewness + kurtosis, good for n≥20.",
    "ks_p": "p-value from Kolmogorov-Smirnov test vs. normal distribution.",
    "anderson_stat": "Anderson-Darling test statistic. Higher = stronger evidence against normality.",
    "anderson_5pct_cv": "Anderson-Darling 5% critical value. If stat > cv → reject normality at 5%.",
    # Missing
    "missing_count": "Number of missing (null) values in this column.",
    "missing_ratio": "Fraction of missing values = missing_count / total_rows (0 to 1).",
    "dtype": "Pandas dtype of the column.",
    # Outlier
    "lower_bound": "IQR lower fence = Q1 − k × IQR. Values below this are outliers (default k=1.5).",
    "upper_bound": "IQR upper fence = Q3 + k × IQR. Values above this are outliers (default k=1.5).",
    "outlier_count": "Number of values falling outside the outlier bounds.",
    "outlier_%": "Percentage of outlier values = (outlier_count / total) × 100.",
    "min_outlier": "Smallest outlier value detected.",
    "max_outlier": "Largest outlier value detected.",
    "threshold": "Z-score threshold used. Values with |z| > threshold are outliers.",
    "max_zscore": "Maximum absolute z-score found in the column.",
    # Categorical
    "top_value": "The most frequently occurring category value.",
    "top_frequency": "Count of the most frequent category.",
    "top_%": "Percentage of the most frequent category = (top_freq / total) × 100.",
    "entropy": "Shannon entropy (bits). Higher = more uniform distribution among categories.",
    "norm_entropy": "Normalized entropy = entropy / log2(unique). 1.0 = perfectly uniform.",
    "max_entropy": "Maximum possible entropy = log2(unique). Achieved when all categories are equally frequent.",
    "normalized_entropy": "Same as norm_entropy: entropy / max_entropy. 1.0 = uniform.",
    "unique_values": "Number of distinct category values.",
    # Feature importance
    "variance": "Variance of the column = mean of squared deviations from mean.",
    "mean_abs_corr": "Mean absolute Pearson correlation with all other numeric columns.",
    "avg_mutual_info": "Average mutual information with all other columns (uses sklearn).",
    # Correlation
    "VIF": "Variance Inflation Factor. VIF=1 → no multicollinearity, >5 → moderate, >10 → severe.",
    "multicollinearity": "Interpretation of VIF: low (<5), moderate (5–10), or high (≥10).",
    # PCA
    "variance_ratio": "Proportion of total variance explained by this principal component.",
    "cumulative_ratio": "Cumulative proportion of variance explained up to this component.",
    "eigenvalue": "Eigenvalue of the covariance matrix for this component. Higher = more variance.",
    "n_components": "Total number of principal components computed.",
    "total_variance_explained": "Total variance captured by all computed components.",
    "components_for_90pct": "Minimum number of components needed to explain ≥ 90% of variance.",
    "top_component_variance": "Variance ratio of the first (most important) principal component.",
    # Duplicates
    "total_rows": "Total number of rows in the dataset.",
    "duplicate_rows": "Number of exact duplicate rows found.",
    "unique_rows": "Number of unique (non-duplicate) rows.",
    "duplicate_ratio": "Fraction of duplicate rows = duplicate_rows / total_rows.",
    "uniqueness_ratio": "Ratio of unique values = unique / total_non_null. 1.0 = all unique.",
    "total_non_null": "Number of non-null values used for uniqueness calculation.",
    "is_unique_key": "True if every non-null value is unique — potential primary key.",
    # Quality
    "completeness": "Fraction of non-missing values = 1 − (missing / total). 1.0 = no missing data.",
    "uniqueness": "Ratio of unique values to total non-null values. Higher = more diverse.",
    "consistency": "Measures type consistency. 1.0 = all values match the expected data type.",
    "validity": "Fraction of values within expected ranges/formats. 1.0 = all valid.",
    "overall": "Weighted quality score = 0.35×completeness + 0.25×uniqueness + 0.20×consistency + 0.20×validity.",
    "quality_score": "Per-column quality score combining completeness and uniqueness.",
    # Common row-index labels
    "column": "Column name in the dataset.",
    "component": "Principal component identifier (PC1, PC2, …).",
    "value": "Category or discrete value.",
    "percentage": "Percentage share of this value = (count / total) × 100.",
}


def _df_to_html(df: pd.DataFrame, max_rows: int = 100) -> str:
    """Convert a DataFrame to an HTML table with tooltip annotations."""
    if df.empty:
        return "<p>No data available</p>"

    sub = df.head(max_rows)
    # Build table manually to inject data-tip attributes
    parts: list[str] = ['<table class="table" border="0">']

    # Header row
    parts.append("<thead><tr>")
    # Index header
    idx_name = sub.index.name or ""
    tip = _METRIC_TIPS.get(idx_name, "")
    tip_attr = f' data-tip="{tip}"' if tip else ""
    parts.append(f"<th{tip_attr}>{idx_name}</th>")
    for col in sub.columns:
        tip = _METRIC_TIPS.get(str(col), "")
        tip_attr = f' data-tip="{tip}"' if tip else ""
        parts.append(f"<th{tip_attr}>{col}</th>")
    parts.append("</tr></thead>")

    # Body rows
    parts.append("<tbody>")
    for idx_val, row in sub.iterrows():
        parts.append("<tr>")
        # Index cell — row identifier
        parts.append(f"<td>{idx_val}</td>")
        for col in sub.columns:
            val = row[col]
            col_tip = _METRIC_TIPS.get(str(col), "")
            # Format the display value
            if isinstance(val, float):
                display = f"{val:.4f}"
            else:
                display = str(val) if pd.notna(val) else "NaN"
            tip_attr = f' data-tip="{col_tip}"' if col_tip else ""
            parts.append(f"<td{tip_attr}>{display}</td>")
        parts.append("</tr>")
    parts.append("</tbody></table>")
    return "\n".join(parts)


def _dict_to_cards(d: dict[str, Any], fmt: str = ",.0f") -> str:
    """Convert a dict to stat-card HTML elements with tooltips."""
    cards: list[str] = []
    for key, val in d.items():
        if isinstance(val, float):
            display = f"{val * 100:.1f}%" if val <= 1 else f"{val:{fmt}}"
        elif isinstance(val, int):
            display = f"{val:,}"
        else:
            display = str(val)
        label = key.replace("_", " ").title()
        tip = _METRIC_TIPS.get(key, "")
        tip_attr = f' data-tip="{tip}"' if tip else ""
        cards.append(
            f'<div class="card"{tip_attr}><div class="value">{display}</div>'
            f'<div class="label">{label}</div></div>'
        )
    return "\n".join(cards)


# =====================================================================
#  CSS / JS constants
# =====================================================================

_CSS = """
* { margin: 0; padding: 0; box-sizing: border-box; }
body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    line-height: 1.6; color: #333; background: #f5f7fa; margin: 0;
}
/* Header */
.header {
    background: linear-gradient(135deg, #2c3e50, #3498db);
    color: #fff; padding: 30px 40px;
}
.header h1 { font-size: 1.8em; margin-bottom: 4px; }
.header p  { font-size: 1.05em; opacity: 0.9; }
/* Top nav */
.topnav {
    background: #fff; border-bottom: 1px solid #dde; padding: 8px 20px;
    position: sticky; top: 0; z-index: 100;
    display: flex; flex-wrap: wrap; gap: 4px; align-items: center;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
}
.topnav a {
    padding: 5px 14px; border-radius: 20px; text-decoration: none;
    color: #666; font-size: 0.82em; transition: all 0.2s; white-space: nowrap;
}
.topnav a:hover, .topnav a.active {
    background: #3498db; color: #fff;
}
/* Main content */
.main { max-width: 1400px; margin: 0 auto; padding: 20px; }
/* Sections */
section {
    background: #fff; border-radius: 12px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    margin: 20px 0; padding: 25px;
}
.section-title {
    font-size: 1.25em; color: #2c3e50;
    border-bottom: 2px solid #3498db;
    padding-bottom: 8px; margin-bottom: 18px;
}
.section-subtitle { font-size: 1em; color: #555; margin: 18px 0 10px 0; }
/* Cards grid */
.cards {
    display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
    gap: 12px; margin: 15px 0;
}
.card {
    background: #f8f9fa; border-radius: 8px; padding: 14px; text-align: center;
}
.card .value { font-size: 1.7em; font-weight: bold; color: #3498db; }
.card .label { font-size: 0.82em; color: #888; margin-top: 2px; }
/* Tables */
.table-wrapper {
    position: relative; overflow-x: auto; overflow-y: visible;
    margin: 12px 0; border: 1px solid #e0e0e0; border-radius: 8px;
    cursor: grab; -webkit-user-select: none; user-select: none;
}
.table-wrapper.dragging { cursor: grabbing; }
.table-wrapper .scroll-hint {
    position: absolute; top: 0; right: 0; bottom: 0; width: 40px;
    pointer-events: none;
    background: linear-gradient(to right, transparent, rgba(0,0,0,0.06));
    border-radius: 0 8px 8px 0; transition: opacity 0.3s;
}
.table-wrapper .scroll-hint.hidden { opacity: 0; }
.table {
    width: max-content; min-width: 100%; border-collapse: collapse; font-size: 0.85em;
}
.table th, .table td {
    padding: 7px 11px; text-align: left; border-bottom: 1px solid #eee; white-space: nowrap;
}
.table th {
    background: #f8f9fa; font-weight: 600; position: sticky; top: 0; z-index: 1;
}
.table th:first-child { position: sticky; left: 0; z-index: 2; background: #eef2f5; }
.table td:first-child {
    position: sticky; left: 0; background: #fff; z-index: 1;
    font-weight: 500; border-right: 2px solid #e0e0e0;
}
.table tr:hover td { background: #f1f3f5; }
.table tr:hover td:first-child { background: #e8ecf0; }
/* Charts */
.charts-grid {
    display: grid; grid-template-columns: repeat(auto-fit, minmax(420px, 1fr));
    gap: 15px; margin: 15px 0;
}
.chart-card {
    background: #fafafa; border-radius: 8px; padding: 12px; text-align: center;
}
.chart-card img { max-width: 100%; border-radius: 6px; }
.chart-card h4 { font-size: 0.9em; color: #555; margin-bottom: 8px; }
/* Single full-width chart */
.chart-full { text-align: center; margin: 15px 0; }
.chart-full img { max-width: 100%; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); }
/* Warnings */
.warnings {
    background: #fff3cd; border: 1px solid #ffc107; border-radius: 8px;
    padding: 14px; margin: 15px 0;
}
.warnings li { margin: 4px 0 4px 20px; font-size: 0.92em; }
/* Preprocessing log */
.log-list { list-style: none; padding: 0; }
.log-list li { padding: 4px 0; font-size: 0.9em; color: #555; }
.log-list li::before { content: "-> "; color: #3498db; font-weight: bold; }
/* Quality gauge */
.quality-bars { display: flex; flex-wrap: wrap; gap: 20px; margin: 15px 0; }
.qbar { flex: 1; min-width: 120px; }
.qbar-label { font-size: 0.85em; color: #555; margin-bottom: 4px; }
.qbar-track { background: #eee; border-radius: 6px; height: 22px; position: relative; overflow: hidden; }
.qbar-fill { height: 100%; border-radius: 6px; transition: width 0.4s; display: flex; align-items: center; justify-content: flex-end; padding-right: 6px; font-size: 0.75em; color: #fff; font-weight: 600; }
.qbar-fill.good { background: #27ae60; } .qbar-fill.fair { background: #f39c12; } .qbar-fill.poor { background: #e74c3c; }
/* Tabs (multi-subset) */
.tab-bar {
    display: flex; flex-wrap: wrap; gap: 4px;
    border-bottom: 2px solid #e0e0e0; margin: 20px 0 0 0;
}
.tab-btn {
    padding: 10px 20px; border: 1px solid #ddd; border-bottom: none;
    background: #f8f9fa; cursor: pointer; border-radius: 8px 8px 0 0;
    font-size: 0.92em; transition: background 0.15s;
}
.tab-btn:hover { background: #e9ecef; }
.tab-btn.active {
    background: #fff; border-bottom: 2px solid #fff; margin-bottom: -2px;
    font-weight: 600; color: #3498db;
}
.tab-content { padding: 20px 0; }
.summary-bar {
    background: #eaf3fb; border-radius: 8px; padding: 12px 20px;
    margin: 10px 0 20px 0; font-size: 1.05em;
}
/* Footer */
footer { text-align: center; margin-top: 40px; padding: 20px; color: #aaa; font-size: 0.85em; }
/* Tooltip */
.f2a-tooltip {
    position: fixed; z-index: 9999;
    max-width: 340px; padding: 10px 14px;
    background: #2c3e50; color: #fff; font-size: 0.82em; line-height: 1.5;
    border-radius: 8px; pointer-events: none;
    box-shadow: 0 4px 16px rgba(0,0,0,0.25);
    opacity: 0; transition: opacity 0.15s;
}
.f2a-tooltip.visible { opacity: 1; }
.f2a-tooltip .tip-header {
    font-weight: 700; color: #5dade2; margin-bottom: 4px;
    border-bottom: 1px solid rgba(255,255,255,0.15); padding-bottom: 3px;
}
.f2a-tooltip .tip-value { color: #f9e79f; font-weight: 600; }
[data-tip] { cursor: help; }
th[data-tip] { text-decoration: underline dotted rgba(0,0,0,0.25); text-underline-offset: 3px; }
"""

_DRAG_SCROLL_JS = """
(function() {
    document.querySelectorAll('.table-wrapper').forEach(function(wrapper) {
        var isDown = false, startX, scrollLeft, velX = 0, momentumId;
        function updateHint() {
            var hint = wrapper.querySelector('.scroll-hint');
            if (!hint) return;
            hint.classList.toggle('hidden',
                wrapper.scrollLeft + wrapper.clientWidth >= wrapper.scrollWidth - 2);
        }
        wrapper.addEventListener('mousedown', function(e) {
            isDown = true; wrapper.classList.add('dragging');
            startX = e.pageX - wrapper.offsetLeft; scrollLeft = wrapper.scrollLeft;
            velX = 0; cancelAnimationFrame(momentumId); e.preventDefault();
        });
        wrapper.addEventListener('mouseleave', function() {
            if (isDown) { isDown = false; wrapper.classList.remove('dragging'); startMomentum(); }
        });
        wrapper.addEventListener('mouseup', function() {
            if (isDown) { isDown = false; wrapper.classList.remove('dragging'); startMomentum(); }
        });
        wrapper.addEventListener('mousemove', function(e) {
            if (!isDown) return;
            var x = e.pageX - wrapper.offsetLeft;
            var walk = (x - startX) * 1.5;
            velX = wrapper.scrollLeft;
            wrapper.scrollLeft = scrollLeft - walk;
            velX = velX - wrapper.scrollLeft;
            updateHint();
        });
        wrapper.addEventListener('scroll', updateHint);
        function startMomentum() {
            cancelAnimationFrame(momentumId);
            (function step() {
                velX *= 0.92;
                if (Math.abs(velX) > 0.5) {
                    wrapper.scrollLeft -= velX; updateHint();
                    momentumId = requestAnimationFrame(step);
                }
            })();
        }
        var touchStartX, touchScrollLeft;
        wrapper.addEventListener('touchstart', function(e) {
            touchStartX = e.touches[0].pageX; touchScrollLeft = wrapper.scrollLeft;
        }, {passive: true});
        wrapper.addEventListener('touchmove', function(e) {
            wrapper.scrollLeft = touchScrollLeft - (e.touches[0].pageX - touchStartX);
            updateHint();
        }, {passive: true});
        updateHint();
    });
})();
"""

_NAV_SCROLL_JS = """
(function() {
    var links = document.querySelectorAll('.topnav a[href^="#"]');
    var sections = [];
    links.forEach(function(a) {
        var id = a.getAttribute('href').slice(1);
        var el = document.getElementById(id);
        if (el) sections.push({el: el, link: a});
    });
    function highlight() {
        var scrollY = window.scrollY + 120;
        var active = null;
        sections.forEach(function(s) {
            if (s.el.offsetTop <= scrollY) active = s;
        });
        links.forEach(function(a) { a.classList.remove('active'); });
        if (active) active.link.classList.add('active');
    }
    window.addEventListener('scroll', highlight);
    highlight();
})();
"""

_TOOLTIP_JS = """
(function() {
    var tip = document.createElement('div');
    tip.className = 'f2a-tooltip';
    document.body.appendChild(tip);
    var showTimer = null, hideTimer = null;

    function getColHeader(td) {
        var ci = Array.prototype.indexOf.call(td.parentNode.children, td);
        var thead = td.closest('table').querySelector('thead');
        if (!thead) return null;
        var ths = thead.querySelectorAll('tr:first-child th');
        return ci < ths.length ? ths[ci] : null;
    }

    function getRowLabel(td) {
        var first = td.parentNode.children[0];
        return first ? first.textContent.trim() : '';
    }

    function show(el, ev) {
        var desc = el.getAttribute('data-tip');
        if (!desc) return;
        var tagName = el.tagName.toLowerCase();
        var html = '';
        if (tagName === 'th') {
            html = '<div class="tip-header">' + el.textContent.trim() + '</div>' + desc;
        } else {
            var colTh = getColHeader(el);
            var colName = colTh ? colTh.textContent.trim() : '';
            var rowLabel = getRowLabel(el);
            var cellVal = el.textContent.trim();
            html = '';
            if (rowLabel) html += '<div class="tip-header">' + rowLabel + ' → ' + colName + '</div>';
            else if (colName) html += '<div class="tip-header">' + colName + '</div>';
            html += desc;
            if (cellVal && cellVal !== 'NaN') html += '<br><span class="tip-value">Value: ' + cellVal + '</span>';
        }
        tip.innerHTML = html;
        tip.classList.add('visible');
        position(ev);
    }

    function position(ev) {
        var x = ev.clientX + 14, y = ev.clientY + 14;
        var tw = tip.offsetWidth, th2 = tip.offsetHeight;
        var vw = window.innerWidth, vh = window.innerHeight;
        if (x + tw > vw - 10) x = ev.clientX - tw - 10;
        if (y + th2 > vh - 10) y = ev.clientY - th2 - 10;
        if (x < 4) x = 4; if (y < 4) y = 4;
        tip.style.left = x + 'px'; tip.style.top = y + 'px';
    }

    function hide() { tip.classList.remove('visible'); }

    document.addEventListener('mouseover', function(e) {
        var el = e.target.closest('[data-tip]');
        if (!el) return;
        clearTimeout(hideTimer);
        showTimer = setTimeout(function() { show(el, e); }, 250);
    });
    document.addEventListener('mousemove', function(e) {
        if (tip.classList.contains('visible')) position(e);
    });
    document.addEventListener('mouseout', function(e) {
        var el = e.target.closest('[data-tip]');
        if (!el) return;
        clearTimeout(showTimer);
        hideTimer = setTimeout(hide, 120);
    });
})();
"""


# =====================================================================
#  Section builders
# =====================================================================

def _build_quality_bars(scores: dict[str, Any]) -> str:
    """Build quality gauge HTML from quality scores dict."""
    if not scores:
        return ""

    dims = [
        ("Completeness", "completeness", scores.get("completeness", 0)),
        ("Uniqueness", "uniqueness", scores.get("uniqueness", 0)),
        ("Consistency", "consistency", scores.get("consistency", 0)),
        ("Validity", "validity", scores.get("validity", 0)),
        ("Overall", "overall", scores.get("overall", 0)),
    ]
    parts: list[str] = []
    for label, key, val in dims:
        pct = val * 100
        cls = "good" if pct >= 90 else ("fair" if pct >= 70 else "poor")
        tip = _METRIC_TIPS.get(key, "")
        tip_attr = f' data-tip="{tip}"' if tip else ""
        parts.append(
            f'<div class="qbar"{tip_attr}>'
            f'<div class="qbar-label">{label}</div>'
            f'<div class="qbar-track">'
            f'<div class="qbar-fill {cls}" style="width:{pct:.0f}%">{pct:.1f}%</div>'
            f'</div></div>'
        )
    return '<div class="quality-bars">' + "".join(parts) + "</div>"


def _wrap_table(html: str) -> str:
    """Wrap table HTML in a scrollable container."""
    return (
        '<div class="table-wrapper">'
        + html
        + '<div class="scroll-hint"></div></div>'
    )


def _figures_to_html(figures: dict[str, plt.Figure], grid: bool = True) -> str:
    """Convert figure dict to chart HTML (grid or full-width)."""
    parts: list[str] = []
    for name, fig in figures.items():
        b64 = _fig_to_base64(fig)
        if grid:
            parts.append(
                f'<div class="chart-card"><h4>{name}</h4>'
                f'<img src="data:image/png;base64,{b64}" alt="{name}" /></div>'
            )
        else:
            parts.append(
                f'<div class="chart-full"><h4 class="section-subtitle">{name}</h4>'
                f'<img src="data:image/png;base64,{b64}" alt="{name}" /></div>'
            )
    if grid and parts:
        return '<div class="charts-grid">' + "\n".join(parts) + "</div>"
    return "\n".join(parts)


def _build_section(
    section_id: str,
    title: str,
    body: str,
    condition: bool = True,
) -> str:
    """Wrap body content in a <section> element."""
    if not condition or not body.strip():
        return ""
    return (
        f'<section id="{section_id}">'
        f'<h2 class="section-title">{title}</h2>'
        f'{body}</section>'
    )


# =====================================================================
#  Section content builders
# =====================================================================

def _section_overview(schema_summary: dict[str, Any]) -> str:
    return (
        '<div class="cards">'
        + _dict_to_cards({
            "rows": schema_summary.get("rows", 0),
            "columns": schema_summary.get("columns", 0),
            "numeric": schema_summary.get("numeric", 0),
            "categorical": schema_summary.get("categorical", 0),
            "text": schema_summary.get("text", 0),
            "datetime": schema_summary.get("datetime", 0),
            "memory_mb": schema_summary.get("memory_mb", 0),
        })
        + "</div>"
    )


def _section_quality(stats: Any) -> str:
    body = _build_quality_bars(stats.quality_scores)
    if not stats.quality_by_column.empty:
        body += '<h3 class="section-subtitle">Column Quality</h3>'
        body += _wrap_table(_df_to_html(stats.quality_by_column))
    return body


def _section_preprocessing(stats: Any) -> str:
    pp = stats.preprocessing
    if pp is None:
        return ""
    body = '<div class="cards">'
    body += _dict_to_cards({
        "original_rows": pp.original_shape[0],
        "cleaned_rows": pp.cleaned_shape[0],
        "columns_removed": pp.original_shape[1] - pp.cleaned_shape[1],
        "duplicates_removed": pp.duplicate_rows_count,
        "completeness": pp.completeness,
    })
    body += "</div>"
    if pp.cleaning_log:
        body += '<h3 class="section-subtitle">Cleaning Log</h3><ul class="log-list">'
        body += "".join(f"<li>{entry}</li>" for entry in pp.cleaning_log)
        body += "</ul>"
    issues = pp.issues_table()
    if not issues.empty:
        body += '<h3 class="section-subtitle">Detected Issues</h3>'
        body += _wrap_table(_df_to_html(issues))
    return body


def _section_descriptive(stats: Any, figures: dict) -> str:
    body = ""
    if not stats.summary.empty:
        body += _wrap_table(_df_to_html(stats.summary))

    chart_parts: dict[str, plt.Figure] = {}
    for key in ("Distribution Histograms", "Boxplots"):
        if key in figures:
            chart_parts[key] = figures[key]
    if chart_parts:
        body += _figures_to_html(chart_parts, grid=False)
    return body


def _section_distribution(stats: Any, figures: dict) -> str:
    body = ""
    if not stats.distribution_info.empty:
        body += '<h3 class="section-subtitle">Normality Tests & Shape</h3>'
        body += _wrap_table(_df_to_html(stats.distribution_info))

    chart_parts: dict[str, plt.Figure] = {}
    for key in ("Violin Plots", "Q-Q Plots"):
        if key in figures:
            chart_parts[key] = figures[key]
    if chart_parts:
        body += _figures_to_html(chart_parts, grid=False)
    return body


def _section_correlation(stats: Any, figures: dict) -> str:
    body = ""
    chart_parts: dict[str, plt.Figure] = {}
    for key in ("Correlation Heatmap (Pearson)", "Correlation Heatmap (Spearman)"):
        if key in figures:
            chart_parts[key] = figures[key]
    if chart_parts:
        body += _figures_to_html(chart_parts, grid=True)

    if not stats.vif_table.empty:
        body += '<h3 class="section-subtitle">Variance Inflation Factor (VIF)</h3>'
        body += _wrap_table(_df_to_html(stats.vif_table))

    return body


def _section_missing(stats: Any, figures: dict) -> str:
    body = ""
    if not stats.missing_info.empty:
        body += _wrap_table(_df_to_html(stats.missing_info))
    chart_parts: dict[str, plt.Figure] = {}
    for key in ("Missing Data", "Missing Data Matrix"):
        if key in figures:
            chart_parts[key] = figures[key]
    if chart_parts:
        body += _figures_to_html(chart_parts, grid=True)
    return body


def _section_outlier(stats: Any, figures: dict) -> str:
    body = ""
    if not stats.outlier_summary.empty:
        body += _wrap_table(_df_to_html(stats.outlier_summary))
    if "Outlier Detection" in figures:
        body += _figures_to_html({"Outlier Detection": figures["Outlier Detection"]}, grid=False)
    return body


def _section_categorical(stats: Any, figures: dict) -> str:
    body = ""
    if not stats.categorical_analysis.empty:
        body += '<h3 class="section-subtitle">Summary</h3>'
        body += _wrap_table(_df_to_html(stats.categorical_analysis))
    chart_parts: dict[str, plt.Figure] = {}
    for key in ("Categorical Frequency", "Chi-Square Heatmap"):
        if key in figures:
            chart_parts[key] = figures[key]
    if chart_parts:
        body += _figures_to_html(chart_parts, grid=False)
    return body


def _section_feature_importance(stats: Any, figures: dict) -> str:
    body = ""
    if not stats.feature_importance.empty:
        body += _wrap_table(_df_to_html(stats.feature_importance))
    if "Feature Importance" in figures:
        body += _figures_to_html({"Feature Importance": figures["Feature Importance"]}, grid=False)
    return body


def _section_pca(stats: Any, figures: dict) -> str:
    body = ""
    if stats.pca_summary:
        body += '<div class="cards">' + _dict_to_cards(stats.pca_summary) + "</div>"
    if not stats.pca_variance.empty:
        body += '<h3 class="section-subtitle">Variance Explained</h3>'
        body += _wrap_table(_df_to_html(stats.pca_variance))
    if not stats.pca_loadings.empty:
        body += '<h3 class="section-subtitle">Loadings</h3>'
        body += _wrap_table(_df_to_html(stats.pca_loadings))
    chart_parts: dict[str, plt.Figure] = {}
    for key in ("PCA Scree Plot", "PCA Loadings"):
        if key in figures:
            chart_parts[key] = figures[key]
    if chart_parts:
        body += _figures_to_html(chart_parts, grid=True)
    return body


def _section_duplicates(stats: Any) -> str:
    if not stats.duplicate_stats:
        return ""
    return '<div class="cards">' + _dict_to_cards(stats.duplicate_stats) + "</div>"


def _section_warnings(warnings: list[str]) -> str:
    if not warnings:
        return ""
    items = "".join(f"<li>{w}</li>" for w in warnings)
    return f'<div class="warnings"><ul>{items}</ul></div>'


# =====================================================================
#  Navigation links
# =====================================================================

_SECTION_ORDER = [
    ("overview", "Overview"),
    ("quality", "Quality"),
    ("preprocessing", "Preprocessing"),
    ("descriptive", "Descriptive"),
    ("distribution", "Distribution"),
    ("correlation", "Correlation"),
    ("missing", "Missing Data"),
    ("outlier", "Outliers"),
    ("categorical", "Categorical"),
    ("importance", "Feature Importance"),
    ("pca", "PCA"),
    ("duplicates", "Duplicates"),
    ("warnings-section", "Warnings"),
]


# =====================================================================
#  Report Generator
# =====================================================================

class ReportGenerator:
    """Generate comprehensive HTML reports from analysis results."""

    # -- Single partition -------------------------------------------------

    def generate_html(
        self,
        dataset_name: str,
        schema_summary: dict[str, Any],
        stats: Any,
        figures: dict[str, plt.Figure],
        warnings: list[str] | None = None,
        config: AnalysisConfig | None = None,
    ) -> str:
        """Generate a full HTML report string."""
        warnings = warnings or []
        config = config or AnalysisConfig()

        sections_html = ""
        sections_html += _build_section("overview", "Overview", _section_overview(schema_summary))
        sections_html += _build_section("quality", "Data Quality", _section_quality(stats), config.quality_score)
        sections_html += _build_section("preprocessing", "Preprocessing", _section_preprocessing(stats), config.preprocessing)
        sections_html += _build_section("descriptive", "Descriptive Statistics", _section_descriptive(stats, figures), config.descriptive)
        sections_html += _build_section("distribution", "Distribution Analysis", _section_distribution(stats, figures), config.distribution)
        sections_html += _build_section("correlation", "Correlation Analysis", _section_correlation(stats, figures), config.correlation)
        sections_html += _build_section("missing", "Missing Data Analysis", _section_missing(stats, figures))
        sections_html += _build_section("outlier", "Outlier Detection", _section_outlier(stats, figures), config.outlier)
        sections_html += _build_section("categorical", "Categorical Analysis", _section_categorical(stats, figures), config.categorical)
        sections_html += _build_section("importance", "Feature Importance", _section_feature_importance(stats, figures), config.feature_importance)
        sections_html += _build_section("pca", "PCA Analysis", _section_pca(stats, figures), config.pca)
        sections_html += _build_section("duplicates", "Duplicate Analysis", _section_duplicates(stats), config.duplicates)
        sections_html += _build_section("warnings-section", "Warnings", _section_warnings(warnings), bool(warnings))

        nav_links = "".join(
            f'<a href="#{sid}">{label}</a>' for sid, label in _SECTION_ORDER
        )
        rows = schema_summary.get("rows", 0)
        cols = schema_summary.get("columns", 0)

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>f2a Report - {dataset_name}</title>
<style>{_CSS}</style>
</head>
<body>
<div class="header">
    <h1>f2a Analysis Report</h1>
    <p>{dataset_name} &mdash; {rows:,} rows x {cols} columns</p>
</div>
<nav class="topnav">{nav_links}</nav>
<div class="main">
{sections_html}
</div>
<footer>Generated by <strong>f2a</strong> (File to Analysis)</footer>
<script>{_DRAG_SCROLL_JS}</script>
<script>{_NAV_SCROLL_JS}</script>
<script>{_TOOLTIP_JS}</script>
</body>
</html>"""
        return html

    def save_html(self, output_path: str | Path, **kwargs: Any) -> Path:
        """Save single-partition HTML report to file."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        html = self.generate_html(**kwargs)
        path.write_text(html, encoding="utf-8")
        logger.info("Report saved: %s", path)
        return path

    # -- Multi-subset -----------------------------------------------------

    def generate_html_multi(
        self,
        dataset_name: str,
        sections: list[dict[str, Any]],
        config: AnalysisConfig | None = None,
    ) -> str:
        """Generate a multi-subset tabbed HTML report."""
        config = config or AnalysisConfig()

        tab_buttons: list[str] = []
        tab_contents: list[str] = []

        for idx, sec in enumerate(sections):
            tab_id = f"tab-{idx}"
            label = f"{sec['subset']} / {sec['split']}"
            active = "active" if idx == 0 else ""

            tab_buttons.append(
                f'<button class="tab-btn {active}" '
                f"""onclick="openTab(event, '{tab_id}')">{label}</button>"""
            )

            s = sec["stats"]
            figures = sec.get("figures", {})
            schema = sec["schema_summary"]
            sec_warnings = sec.get("warnings", [])

            inner = ""
            inner += _build_section(f"{tab_id}-overview", "Overview", _section_overview(schema))
            inner += _build_section(f"{tab_id}-quality", "Data Quality", _section_quality(s), config.quality_score)
            inner += _build_section(f"{tab_id}-preprocessing", "Preprocessing", _section_preprocessing(s), config.preprocessing)
            inner += _build_section(f"{tab_id}-descriptive", "Descriptive Statistics", _section_descriptive(s, figures), config.descriptive)
            inner += _build_section(f"{tab_id}-distribution", "Distribution Analysis", _section_distribution(s, figures), config.distribution)
            inner += _build_section(f"{tab_id}-correlation", "Correlation Analysis", _section_correlation(s, figures), config.correlation)
            inner += _build_section(f"{tab_id}-missing", "Missing Data", _section_missing(s, figures))
            inner += _build_section(f"{tab_id}-outlier", "Outlier Detection", _section_outlier(s, figures), config.outlier)
            inner += _build_section(f"{tab_id}-categorical", "Categorical Analysis", _section_categorical(s, figures), config.categorical)
            inner += _build_section(f"{tab_id}-importance", "Feature Importance", _section_feature_importance(s, figures), config.feature_importance)
            inner += _build_section(f"{tab_id}-pca", "PCA Analysis", _section_pca(s, figures), config.pca)
            inner += _build_section(f"{tab_id}-duplicates", "Duplicates", _section_duplicates(s), config.duplicates)
            inner += _build_section(f"{tab_id}-warnings", "Warnings", _section_warnings(sec_warnings), bool(sec_warnings))

            display = "block" if idx == 0 else "none"
            tab_contents.append(
                f'<div id="{tab_id}" class="tab-content" style="display:{display};">'
                f"<h2>{label}</h2>{inner}</div>"
            )

        total_rows = sum(s["schema_summary"].get("rows", 0) for s in sections)
        tabs_html = "\n".join(tab_buttons)
        content_html = "\n".join(tab_contents)

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>f2a Report - {dataset_name}</title>
<style>{_CSS}</style>
</head>
<body>
<div class="header">
    <h1>f2a Analysis Report</h1>
    <p>{dataset_name}</p>
</div>
<div class="main">
    <div class="summary-bar">
        Total: <strong>{total_rows:,}</strong> rows across
        <strong>{len(sections)}</strong> subsets / splits
    </div>
    <div class="tab-bar">{tabs_html}</div>
    {content_html}
</div>
<footer>Generated by <strong>f2a</strong> (File to Analysis)</footer>
<script>
function openTab(evt, tabId) {{
    document.querySelectorAll('.tab-content').forEach(function(el) {{ el.style.display = 'none'; }});
    document.querySelectorAll('.tab-btn').forEach(function(el) {{ el.classList.remove('active'); }});
    document.getElementById(tabId).style.display = 'block';
    evt.currentTarget.classList.add('active');
}}
</script>
<script>{_DRAG_SCROLL_JS}</script>
<script>{_TOOLTIP_JS}</script>
</body>
</html>"""
        return html

    def save_html_multi(self, output_path: str | Path, **kwargs: Any) -> Path:
        """Save multi-subset HTML report to file."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        html = self.generate_html_multi(**kwargs)
        path.write_text(html, encoding="utf-8")
        logger.info("Report saved: %s", path)
        return path
