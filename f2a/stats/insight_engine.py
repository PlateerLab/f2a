"""Automatic Insight Engine — generates prioritised natural-language insights.

The engine scans *all* previously computed statistics (basic + advanced) and
applies a comprehensive set of interpretive rules to surface:

* ``FINDING``  — notable data patterns or facts
* ``WARNING``  — data quality or integrity concerns
* ``RECOMMENDATION`` — actionable preprocessing / modelling suggestions
* ``OPPORTUNITY``  — exploitable patterns or segmentation opportunities

Every insight carries a severity (critical / high / medium / low), a
priority score for ranking, related column names, and concrete action items.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd

from f2a.core.schema import DataSchema
from f2a.utils.logging import get_logger

logger = get_logger(__name__)

# =====================================================================
#  Data classes
# =====================================================================

class InsightType(str, Enum):
    FINDING = "finding"
    WARNING = "warning"
    RECOMMENDATION = "recommendation"
    OPPORTUNITY = "opportunity"


class Severity(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


# Numeric weight for priority scoring
_SEV_WEIGHT = {Severity.CRITICAL: 1.0, Severity.HIGH: 0.75, Severity.MEDIUM: 0.5, Severity.LOW: 0.25}


@dataclass
class Insight:
    """A single auto-generated insight."""

    type: InsightType
    severity: Severity
    category: str              # distribution | correlation | cluster | anomaly | missing | quality | feature | general
    title: str
    description: str
    affected_columns: list[str] = field(default_factory=list)
    evidence: dict[str, Any] = field(default_factory=dict)
    action_items: list[str] = field(default_factory=list)
    priority_score: float = 0.0

    # Computed after instantiation
    def __post_init__(self) -> None:
        if self.priority_score == 0.0:
            col_factor = min(len(self.affected_columns) / 5.0, 1.0) if self.affected_columns else 0.3
            actionable = 1.0 if self.action_items else 0.6
            self.priority_score = round(
                _SEV_WEIGHT.get(self.severity, 0.5) * 0.5
                + col_factor * 0.3
                + actionable * 0.2,
                4,
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type.value,
            "severity": self.severity.value,
            "category": self.category,
            "title": self.title,
            "description": self.description,
            "affected_columns": self.affected_columns,
            "evidence": {k: _safe_serialize(v) for k, v in self.evidence.items()},
            "action_items": self.action_items,
            "priority_score": self.priority_score,
        }


def _safe_serialize(v: Any) -> Any:
    """Convert numpy / pandas types to JSON-safe Python primitives."""
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.floating):
        return float(v)
    if isinstance(v, np.bool_):
        return bool(v)
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, pd.DataFrame):
        return v.to_dict()
    if isinstance(v, pd.Series):
        return v.to_dict()
    return v


# =====================================================================
#  Insight Engine
# =====================================================================

class InsightEngine:
    """Generate, rank, and present actionable insights from ``StatsResult``.

    Usage::

        engine = InsightEngine(stats_result, data_schema)
        insights = engine.generate()         # list[Insight]
        executive = engine.executive_summary()  # str
    """

    def __init__(self, stats: Any, schema: DataSchema) -> None:
        self._stats = stats
        self._schema = schema
        self._insights: list[Insight] = []

    # ------------------------------------------------------------------
    #  Public API
    # ------------------------------------------------------------------

    def generate(self) -> list[Insight]:
        """Run all rule sets and return insights sorted by priority (desc)."""
        self._insights.clear()

        try:
            self._distribution_insights()
        except Exception as exc:
            logger.debug("Distribution insight rules failed: %s", exc)

        try:
            self._correlation_insights()
        except Exception as exc:
            logger.debug("Correlation insight rules failed: %s", exc)

        try:
            self._missing_insights()
        except Exception as exc:
            logger.debug("Missing insight rules failed: %s", exc)

        try:
            self._outlier_insights()
        except Exception as exc:
            logger.debug("Outlier insight rules failed: %s", exc)

        try:
            self._quality_insights()
        except Exception as exc:
            logger.debug("Quality insight rules failed: %s", exc)

        try:
            self._clustering_insights()
        except Exception as exc:
            logger.debug("Clustering insight rules failed: %s", exc)

        try:
            self._anomaly_insights()
        except Exception as exc:
            logger.debug("Anomaly insight rules failed: %s", exc)

        try:
            self._feature_insights()
        except Exception as exc:
            logger.debug("Feature insight rules failed: %s", exc)

        try:
            self._pca_insights()
        except Exception as exc:
            logger.debug("PCA insight rules failed: %s", exc)

        try:
            self._duplicate_insights()
        except Exception as exc:
            logger.debug("Duplicate insight rules failed: %s", exc)

        try:
            self._advanced_distribution_insights()
        except Exception as exc:
            logger.debug("Adv distribution insight rules failed: %s", exc)

        try:
            self._advanced_correlation_insights()
        except Exception as exc:
            logger.debug("Adv correlation insight rules failed: %s", exc)

        try:
            self._general_insights()
        except Exception as exc:
            logger.debug("General insight rules failed: %s", exc)

        self._insights.sort(key=lambda i: i.priority_score, reverse=True)
        return self._insights

    def executive_summary(self) -> str:
        """One-paragraph natural-language summary of the dataset."""
        if not self._insights:
            self.generate()

        n = self._schema.n_rows
        d = self._schema.n_cols
        num = len(self._schema.numeric_columns)
        cat = len(self._schema.categorical_columns)

        crit = sum(1 for i in self._insights if i.severity == Severity.CRITICAL)
        high = sum(1 for i in self._insights if i.severity == Severity.HIGH)
        med = sum(1 for i in self._insights if i.severity == Severity.MEDIUM)

        parts = [
            f"Dataset contains {n:,} rows and {d} columns ({num} numeric, {cat} categorical).",
        ]
        if crit:
            parts.append(f"{crit} critical issue(s) require immediate attention.")
        if high:
            parts.append(f"{high} high-priority finding(s) detected.")
        if med:
            parts.append(f"{med} moderate observations noted.")

        # Top 3 headlines
        top3 = self._insights[:3]
        if top3:
            parts.append("Key highlights:")
            for idx, ins in enumerate(top3, 1):
                parts.append(f"  {idx}. {ins.title}")

        return " ".join(parts)

    def summary_dict(self) -> dict[str, Any]:
        """Serialize all insights for storage / HTML rendering."""
        if not self._insights:
            self.generate()
        return {
            "executive_summary": self.executive_summary(),
            "total_count": len(self._insights),
            "by_severity": {
                s.value: sum(1 for i in self._insights if i.severity == s)
                for s in Severity
            },
            "by_type": {
                t.value: sum(1 for i in self._insights if i.type == t)
                for t in InsightType
            },
            "insights": [i.to_dict() for i in self._insights],
        }

    # ------------------------------------------------------------------
    #  Helper
    # ------------------------------------------------------------------

    def _add(self, **kwargs: Any) -> None:
        self._insights.append(Insight(**kwargs))

    # ==================================================================
    #  Rule Sets
    # ==================================================================

    # -- 1. Distribution --------------------------------------------------

    def _distribution_insights(self) -> None:
        summary = self._stats.summary
        dist = self._stats.distribution_info
        if summary.empty:
            return

        numeric_rows = summary[summary.get("type", pd.Series(dtype=str)) == "numeric"] if "type" in summary.columns else summary
        if numeric_rows.empty:
            return

        # Extreme skewness
        if "skewness" in numeric_rows.columns:
            skewed = numeric_rows[numeric_rows["skewness"].abs() > 2.0].dropna(subset=["skewness"])
            if not skewed.empty:
                cols = list(skewed.index)
                worst = skewed["skewness"].abs().idxmax()
                worst_val = skewed.loc[worst, "skewness"]
                self._add(
                    type=InsightType.RECOMMENDATION,
                    severity=Severity.HIGH,
                    category="distribution",
                    title=f"{len(cols)} column(s) with extreme skewness",
                    description=(
                        f"Columns {cols[:5]} have |skewness| > 2, indicating "
                        f"heavily asymmetric distributions. "
                        f"Worst: '{worst}' (skewness={worst_val:.2f})."
                    ),
                    affected_columns=cols,
                    evidence={"worst_column": worst, "worst_skewness": float(worst_val)},
                    action_items=[
                        "Apply log or Box-Cox transform to reduce skewness",
                        "Consider robust statistics (median, MAD) instead of mean/std",
                    ],
                )

        # High kurtosis (heavy tails)
        if "kurtosis" in numeric_rows.columns:
            heavy = numeric_rows[numeric_rows["kurtosis"] > 7.0].dropna(subset=["kurtosis"])
            if not heavy.empty:
                cols = list(heavy.index)
                self._add(
                    type=InsightType.WARNING,
                    severity=Severity.MEDIUM,
                    category="distribution",
                    title=f"{len(cols)} column(s) with extreme kurtosis (heavy tails)",
                    description=(
                        f"Columns {cols[:5]} have kurtosis > 7, meaning very heavy tails. "
                        "Outliers may dominate summary statistics."
                    ),
                    affected_columns=cols,
                    evidence={"kurtosis_values": {c: float(numeric_rows.loc[c, "kurtosis"]) for c in cols[:5]}},
                    action_items=[
                        "Use winsorization or robust estimators",
                        "Check these columns for extreme outliers",
                    ],
                )

        # Normality summary
        if not dist.empty and "is_normal_0.05" in dist.columns:
            normal = dist[dist["is_normal_0.05"] == True]
            non_normal = dist[dist["is_normal_0.05"] == False]
            total = len(dist)
            if len(non_normal) > total * 0.8 and total >= 3:
                self._add(
                    type=InsightType.FINDING,
                    severity=Severity.MEDIUM,
                    category="distribution",
                    title=f"{len(non_normal)}/{total} numeric columns are non-normal",
                    description=(
                        "Most numeric columns fail normality tests (α=0.05). "
                        "Non-parametric methods may be more appropriate."
                    ),
                    affected_columns=list(non_normal.index),
                    action_items=[
                        "Prefer non-parametric tests (Kruskal-Wallis, Mann-Whitney) over t-tests/ANOVA",
                        "Consider power transforms if normality is needed for downstream models",
                    ],
                )

        # Low variability
        if "cv" in numeric_rows.columns:
            low_var = numeric_rows[(numeric_rows["cv"].notna()) & (numeric_rows["cv"].abs() < 0.05)]
            if not low_var.empty:
                cols = list(low_var.index)
                self._add(
                    type=InsightType.FINDING,
                    severity=Severity.LOW,
                    category="distribution",
                    title=f"{len(cols)} column(s) with very low variability (CV < 5%)",
                    description=(
                        f"Columns {cols[:5]} have coefficient of variation < 5%, "
                        "meaning values are tightly clustered. These may be near-constant."
                    ),
                    affected_columns=cols,
                    action_items=["Evaluate whether these columns carry useful information"],
                )

    # -- 2. Correlation ---------------------------------------------------

    def _correlation_insights(self) -> None:
        corr = self._stats.correlation_matrix
        vif = self._stats.vif_table
        spearman = self._stats.spearman_matrix

        # Multicollinearity via VIF
        if not vif.empty and "VIF" in vif.columns:
            severe = vif[vif["VIF"] > 10]
            if not severe.empty:
                cols = list(severe.index)
                worst = severe["VIF"].idxmax()
                self._add(
                    type=InsightType.WARNING,
                    severity=Severity.CRITICAL if len(severe) > 3 else Severity.HIGH,
                    category="correlation",
                    title=f"{len(severe)} column(s) with severe multicollinearity (VIF>10)",
                    description=(
                        f"VIF > 10 detected for: {cols[:5]}. "
                        f"Worst: '{worst}' (VIF={severe.loc[worst, 'VIF']:.1f}). "
                        "Redundant information may cause model instability."
                    ),
                    affected_columns=cols,
                    evidence={"vif_values": {c: float(severe.loc[c, "VIF"]) for c in cols[:5]}},
                    action_items=[
                        "Remove one column from each highly correlated pair",
                        "Apply PCA or regularization (Ridge/Lasso) to handle collinearity",
                    ],
                )

        # High pearson correlation pairs
        if not corr.empty:
            pairs: list[tuple[str, str, float]] = []
            cols_list = corr.columns.tolist()
            for i, c1 in enumerate(cols_list):
                for c2 in cols_list[i + 1:]:
                    v = corr.loc[c1, c2]
                    if abs(v) > 0.9:
                        pairs.append((c1, c2, float(v)))
            if pairs:
                affected = list({c for p in pairs for c in p[:2]})
                self._add(
                    type=InsightType.WARNING,
                    severity=Severity.HIGH,
                    category="correlation",
                    title=f"{len(pairs)} column pair(s) with |r| > 0.9",
                    description=(
                        "Near-perfect linear relationships detected. "
                        f"Top pair: '{pairs[0][0]}' ↔ '{pairs[0][1]}' (r={pairs[0][2]:.3f})."
                    ),
                    affected_columns=affected,
                    evidence={"pairs": [(p[0], p[1], p[2]) for p in pairs[:5]]},
                    action_items=[
                        "Consider dropping one column from each pair to reduce redundancy",
                        "Verify these are not data leakage or duplicate columns",
                    ],
                )

        # No correlations at all (independent features)
        if not corr.empty and corr.shape[0] >= 3:
            upper = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1))
            max_abs = upper.abs().max().max()
            if max_abs < 0.3:
                self._add(
                    type=InsightType.FINDING,
                    severity=Severity.LOW,
                    category="correlation",
                    title="All numeric columns are weakly correlated (max |r| < 0.3)",
                    description=(
                        "No strong linear relationships found between any pair of numeric columns. "
                        "Features appear largely independent."
                    ),
                    affected_columns=list(corr.columns),
                    action_items=["Check for non-linear relationships via MI or distance correlation"],
                )

    # -- 3. Missing -------------------------------------------------------

    def _missing_insights(self) -> None:
        mi = self._stats.missing_info
        if mi.empty or "missing_ratio" not in mi.columns:
            return

        total_ratio = mi["missing_ratio"].mean() if not mi.empty else 0
        high_miss = mi[mi["missing_ratio"] > 0.5]
        moderate_miss = mi[(mi["missing_ratio"] > 0.1) & (mi["missing_ratio"] <= 0.5)]
        no_miss = mi[mi["missing_ratio"] == 0]

        # Columns with >50% missing
        if not high_miss.empty:
            cols = list(high_miss.index)
            self._add(
                type=InsightType.WARNING,
                severity=Severity.CRITICAL,
                category="missing",
                title=f"{len(cols)} column(s) with >50% missing values",
                description=(
                    f"Columns {cols[:5]} are more than half empty. "
                    "These columns may not be usable without strong imputation."
                ),
                affected_columns=cols,
                evidence={"missing_ratios": {c: float(high_miss.loc[c, "missing_ratio"]) for c in cols[:5]}},
                action_items=[
                    "Consider dropping these columns unless domain-critical",
                    "If kept, use model-based imputation (KNN, MICE) rather than simple mean/median",
                ],
            )

        # Moderate missing
        if not moderate_miss.empty:
            cols = list(moderate_miss.index)
            self._add(
                type=InsightType.RECOMMENDATION,
                severity=Severity.MEDIUM,
                category="missing",
                title=f"{len(cols)} column(s) with 10-50% missing values",
                description=(
                    f"Columns {cols[:5]} have noticeable missing rates. "
                    "Imputation strategy should be chosen carefully."
                ),
                affected_columns=cols,
                action_items=[
                    "Check if missingness is random (MCAR) or systematic (MAR/MNAR)",
                    "For numeric columns: median or KNN imputation; for categorical: mode or indicator variable",
                ],
            )

        # Completely clean
        if len(no_miss) == len(mi) and len(mi) > 0:
            self._add(
                type=InsightType.FINDING,
                severity=Severity.LOW,
                category="missing",
                title="No missing values detected in any column",
                description="All columns are fully populated — no imputation needed.",
                affected_columns=[],
            )

    # -- 4. Outlier -------------------------------------------------------

    def _outlier_insights(self) -> None:
        out = self._stats.outlier_summary
        if out.empty or "outlier_%" not in out.columns:
            return

        extreme = out[out["outlier_%"] > 15]
        moderate = out[(out["outlier_%"] > 5) & (out["outlier_%"] <= 15)]

        if not extreme.empty:
            cols = list(extreme.index)
            self._add(
                type=InsightType.WARNING,
                severity=Severity.HIGH,
                category="anomaly",
                title=f"{len(cols)} column(s) with extreme outlier rate (>15%)",
                description=(
                    f"Columns {cols[:5]} have very high outlier percentages. "
                    "This may indicate data quality issues or heavy-tailed distributions."
                ),
                affected_columns=cols,
                evidence={"outlier_rates": {c: float(extreme.loc[c, "outlier_%"]) for c in cols[:5]}},
                action_items=[
                    "Check if the distribution is truly heavy-tailed (in which case outliers are expected)",
                    "Apply winsorization or log-transform if outliers are distorting analysis",
                    "Consider using robust methods (median, MAD, IQR-based)",
                ],
            )

        if not moderate.empty:
            cols = list(moderate.index)
            self._add(
                type=InsightType.FINDING,
                severity=Severity.MEDIUM,
                category="anomaly",
                title=f"{len(cols)} column(s) with notable outlier rate (5-15%)",
                description=f"Columns {cols[:5]} have moderate outlier rates.",
                affected_columns=cols,
                action_items=["Review outlier boundaries and adjust if domain knowledge warrants"],
            )

    # -- 5. Quality -------------------------------------------------------

    def _quality_insights(self) -> None:
        qs = self._stats.quality_scores
        if not qs:
            return

        overall = qs.get("overall", 1.0)
        if overall < 0.5:
            self._add(
                type=InsightType.WARNING,
                severity=Severity.CRITICAL,
                category="quality",
                title=f"Overall data quality is poor ({overall * 100:.0f}%)",
                description=(
                    "The combined quality score across completeness, uniqueness, "
                    "consistency, and validity is below 50%."
                ),
                evidence=qs,
                action_items=[
                    "Address missing values and inconsistencies before analysis",
                    "Review data collection pipeline for systematic issues",
                ],
            )
        elif overall < 0.75:
            self._add(
                type=InsightType.RECOMMENDATION,
                severity=Severity.MEDIUM,
                category="quality",
                title=f"Data quality is moderate ({overall * 100:.0f}%)",
                description="Some quality dimensions need attention before production use.",
                evidence=qs,
                action_items=["Focus on the lowest-scoring quality dimension"],
            )

        # Per-dimension alerts
        for dim, label in [("completeness", "Completeness"), ("uniqueness", "Uniqueness"),
                           ("consistency", "Consistency"), ("validity", "Validity")]:
            score = qs.get(dim, 1.0)
            if score < 0.6:
                self._add(
                    type=InsightType.WARNING,
                    severity=Severity.HIGH,
                    category="quality",
                    title=f"{label} score is low ({score * 100:.0f}%)",
                    description=f"The {label.lower()} dimension scored {score * 100:.0f}%, dragging down overall quality.",
                    evidence={dim: score},
                    action_items=[f"Investigate and improve {label.lower()} issues"],
                )

    # -- 6. Clustering ----------------------------------------------------

    def _clustering_insights(self) -> None:
        adv = self._stats.advanced_stats
        clustering = adv.get("clustering")
        if not clustering:
            return

        km = clustering.get("kmeans")
        if km:
            k = km.get("optimal_k", 0)
            sil = km.get("best_silhouette", 0)
            sizes = km.get("cluster_sizes", {})

            if k >= 2 and sil > 0.4:
                self._add(
                    type=InsightType.OPPORTUNITY,
                    severity=Severity.HIGH,
                    category="cluster",
                    title=f"Clear cluster structure found (k={k}, silhouette={sil:.2f})",
                    description=(
                        f"K-Means identifies {k} well-separated clusters "
                        f"(silhouette={sil:.2f}). Cluster sizes: {sizes}."
                    ),
                    evidence={"optimal_k": k, "silhouette": sil, "sizes": sizes},
                    action_items=[
                        "Profile each cluster to understand segment characteristics",
                        "Use cluster labels as a feature for downstream modelling",
                    ],
                )
            elif k >= 2 and sil > 0.2:
                self._add(
                    type=InsightType.FINDING,
                    severity=Severity.MEDIUM,
                    category="cluster",
                    title=f"Moderate cluster structure (k={k}, silhouette={sil:.2f})",
                    description=(
                        f"Some grouping exists but clusters overlap. "
                        f"Silhouette={sil:.2f} suggests partial separation."
                    ),
                    evidence={"optimal_k": k, "silhouette": sil},
                    action_items=["Consider density-based methods (DBSCAN) for better cluster boundaries"],
                )

            # Check for imbalanced clusters
            if sizes:
                total = sum(sizes.values())
                if total > 0:
                    min_pct = min(sizes.values()) / total
                    max_pct = max(sizes.values()) / total
                    if min_pct < 0.05:
                        tiny_clusters = [k for k, v in sizes.items() if v / total < 0.05]
                        self._add(
                            type=InsightType.FINDING,
                            severity=Severity.MEDIUM,
                            category="cluster",
                            title=f"Highly imbalanced clusters detected",
                            description=(
                                f"Cluster(s) {tiny_clusters} contain <5% of data. "
                                "These may represent anomalous sub-populations."
                            ),
                            evidence={"tiny_clusters": tiny_clusters, "min_pct": min_pct},
                            action_items=["Inspect small clusters — they may be anomalies or niche segments"],
                        )

        dbscan = clustering.get("dbscan")
        if dbscan:
            noise_ratio = dbscan.get("noise_ratio", 0)
            if noise_ratio > 0.2:
                self._add(
                    type=InsightType.WARNING,
                    severity=Severity.MEDIUM,
                    category="cluster",
                    title=f"DBSCAN labels {noise_ratio * 100:.0f}% of data as noise",
                    description=(
                        "A high proportion of data points don't belong to any density cluster. "
                        "This may indicate dispersed data or sub-optimal epsilon."
                    ),
                    evidence={"noise_ratio": noise_ratio, "eps": dbscan.get("eps")},
                    action_items=["Try adjusting eps parameter or use HDBSCAN for adaptive density"],
                )

    # -- 7. Anomaly -------------------------------------------------------

    def _anomaly_insights(self) -> None:
        adv = self._stats.advanced_stats
        anomaly = adv.get("advanced_anomaly", {})
        consensus = anomaly.get("consensus")
        if not consensus:
            return

        ratio = consensus.get("consensus_ratio", 0)
        count = consensus.get("consensus_count", 0)
        n = consensus.get("n_samples", 1)
        agreement = consensus.get("agreement_matrix", {})

        if ratio > 0.05:
            self._add(
                type=InsightType.WARNING,
                severity=Severity.HIGH,
                category="anomaly",
                title=f"Multi-method consensus: {count} anomalies ({ratio * 100:.1f}%)",
                description=(
                    f"{count} rows flagged as anomalous by ≥2 independent methods "
                    f"(IF + LOF + Mahalanobis). "
                    f"All-agree: {agreement.get('all_agree_anomaly', 0)}, "
                    f"majority: {agreement.get('majority_anomaly', 0)}."
                ),
                evidence={"consensus_ratio": ratio, "agreement": agreement},
                action_items=[
                    "Investigate consensus anomalies — they are high-confidence outliers",
                    "Consider removing or winsorizing before modelling",
                ],
            )
        elif ratio > 0.01:
            self._add(
                type=InsightType.FINDING,
                severity=Severity.MEDIUM,
                category="anomaly",
                title=f"Multi-method anomalies: {count} rows ({ratio * 100:.1f}%)",
                description=(
                    f"A small fraction of rows are flagged by multiple anomaly detection methods."
                ),
                evidence={"consensus_ratio": ratio},
                action_items=["Review flagged rows for data entry errors or special cases"],
            )

    # -- 8. Feature Insights ----------------------------------------------

    def _feature_insights(self) -> None:
        adv = self._stats.advanced_stats
        fi = adv.get("feature_insights", {})
        if not fi:
            return

        # Leakage detection
        leakage = fi.get("leakage")
        if leakage is not None and not leakage.empty:
            high_risk = leakage[leakage.get("risk_level", pd.Series()) == "high"] if "risk_level" in leakage.columns else pd.DataFrame()
            if not high_risk.empty:
                cols = list(high_risk.index)
                self._add(
                    type=InsightType.WARNING,
                    severity=Severity.CRITICAL,
                    category="feature",
                    title=f"{len(cols)} column(s) flagged for potential data leakage",
                    description=(
                        f"Columns {cols[:5]} show high leakage risk "
                        "(constant, ID-like, or perfectly correlated with others)."
                    ),
                    affected_columns=cols,
                    action_items=[
                        "Remove these columns before building any ML model",
                        "Verify they are not derived from the target variable",
                    ],
                )

        # Strong interactions
        interactions = fi.get("interactions")
        if interactions is not None and not interactions.empty:
            strong = interactions[interactions.get("interaction_strength", pd.Series(dtype=float)) > 0.7] if "interaction_strength" in interactions.columns else pd.DataFrame()
            if not strong.empty and len(strong) > 0:
                top = strong.iloc[0]
                self._add(
                    type=InsightType.OPPORTUNITY,
                    severity=Severity.MEDIUM,
                    category="feature",
                    title=f"{len(strong)} strong feature interaction(s) detected",
                    description=(
                        f"Top interaction: '{top.get('col_a', '?')}' × '{top.get('col_b', '?')}' "
                        f"(strength={top.get('interaction_strength', 0):.2f}). "
                        "Product features may improve model performance."
                    ),
                    affected_columns=[str(top.get("col_a", "")), str(top.get("col_b", ""))],
                    action_items=["Create interaction (product) features for the top pairs"],
                )

        # Cardinality / encoding
        card = fi.get("cardinality")
        if card is not None and not card.empty and "recommended_encoding" in card.columns:
            hash_cols = card[card["recommended_encoding"] == "hashing"]
            if not hash_cols.empty:
                cols = list(hash_cols.index)
                self._add(
                    type=InsightType.RECOMMENDATION,
                    severity=Severity.MEDIUM,
                    category="feature",
                    title=f"{len(cols)} high-cardinality column(s) need special encoding",
                    description=(
                        f"Columns {cols[:5]} have very high cardinality. "
                        "One-hot encoding would create too many features."
                    ),
                    affected_columns=cols,
                    action_items=[
                        "Use target encoding, hashing, or embedding for these columns",
                        "Consider grouping rare categories into 'Other'",
                    ],
                )

    # -- 9. PCA -----------------------------------------------------------

    def _pca_insights(self) -> None:
        pca_sum = self._stats.pca_summary
        pca_var = self._stats.pca_variance
        if not pca_sum:
            return

        comp90 = pca_sum.get("components_for_90pct", 0)
        n_orig = len(self._schema.numeric_columns)
        if n_orig > 0 and comp90 > 0:
            reduction = 1 - comp90 / n_orig
            if reduction > 0.5:
                self._add(
                    type=InsightType.OPPORTUNITY,
                    severity=Severity.MEDIUM,
                    category="feature",
                    title=f"High dimensionality reduction potential: {n_orig} → {comp90} components for 90% variance",
                    description=(
                        f"PCA shows that {comp90} components explain 90% of variance "
                        f"from {n_orig} original features ({reduction * 100:.0f}% reduction)."
                    ),
                    evidence={"original_features": n_orig, "pca_components": comp90, "reduction": reduction},
                    action_items=[
                        "Consider PCA projection for dimensionality reduction in ML pipelines",
                        "Examine top PCA loadings to understand dominant variance directions",
                    ],
                )

        # First PC dominance
        if not pca_var.empty and "variance_ratio" in pca_var.columns:
            first_pc = pca_var.iloc[0]["variance_ratio"] if len(pca_var) > 0 else 0
            if first_pc > 0.6:
                self._add(
                    type=InsightType.FINDING,
                    severity=Severity.MEDIUM,
                    category="feature",
                    title=f"First principal component explains {first_pc * 100:.0f}% of variance",
                    description=(
                        "A single axis captures most of the data's variability. "
                        "This suggests a dominant latent factor."
                    ),
                    evidence={"pc1_variance_ratio": first_pc},
                    action_items=["Inspect PC1 loadings to identify the driving variables"],
                )

    # -- 10. Duplicates ---------------------------------------------------

    def _duplicate_insights(self) -> None:
        dup = self._stats.duplicate_stats
        if not dup:
            return

        ratio = dup.get("duplicate_ratio", 0)
        count = dup.get("duplicate_rows", 0)

        if ratio > 0.1:
            self._add(
                type=InsightType.WARNING,
                severity=Severity.HIGH,
                category="quality",
                title=f"{count} duplicate rows ({ratio * 100:.1f}% of dataset)",
                description="Significant portion of data is duplicated, which may bias analysis and modelling.",
                evidence={"duplicate_rows": count, "duplicate_ratio": ratio},
                action_items=[
                    "Remove exact duplicates before analysis",
                    "Check if duplicates are legitimate (e.g. repeated measurements) or data errors",
                ],
            )
        elif ratio > 0.01:
            self._add(
                type=InsightType.FINDING,
                severity=Severity.LOW,
                category="quality",
                title=f"{count} duplicate rows ({ratio * 100:.1f}%)",
                description="A small number of duplicate rows exist.",
                evidence={"duplicate_rows": count, "duplicate_ratio": ratio},
                action_items=["Review whether duplicates should be removed for your use case"],
            )

    # -- 11. Advanced Distribution ----------------------------------------

    def _advanced_distribution_insights(self) -> None:
        adv = self._stats.advanced_stats
        adv_dist = adv.get("advanced_distribution", {})
        if not adv_dist:
            return

        # Best-fit distribution
        best_fit = adv_dist.get("best_fit")
        if best_fit is not None and not best_fit.empty and "best_distribution" in best_fit.columns:
            non_normal = best_fit[best_fit["best_distribution"] != "norm"]
            if not non_normal.empty:
                dist_counts: dict[str, int] = {}
                for d in non_normal["best_distribution"]:
                    dist_counts[d] = dist_counts.get(d, 0) + 1
                most_common = max(dist_counts, key=dist_counts.get)
                self._add(
                    type=InsightType.FINDING,
                    severity=Severity.MEDIUM,
                    category="distribution",
                    title=f"{len(non_normal)} column(s) best fit by non-normal distributions",
                    description=(
                        f"Distribution fitting reveals non-Normal best fits. "
                        f"Most common: {most_common} ({dist_counts[most_common]} columns). "
                        f"Others: {dict(list(dist_counts.items())[:5])}."
                    ),
                    affected_columns=list(non_normal.index),
                    evidence={"distribution_counts": dist_counts},
                    action_items=[
                        "Use the identified distributions for parametric modeling or simulation",
                        "Transform columns toward normality if Gaussian assumptions are needed",
                    ],
                )

        # Power transform recommendations
        pt = adv_dist.get("power_transform")
        if pt is not None and not pt.empty and "needs_transform" in pt.columns:
            needs = pt[pt["needs_transform"] == True]
            if not needs.empty:
                cols = list(needs.index)
                self._add(
                    type=InsightType.RECOMMENDATION,
                    severity=Severity.MEDIUM,
                    category="distribution",
                    title=f"{len(cols)} column(s) benefit from power transformation",
                    description=(
                        f"Box-Cox / Yeo-Johnson transforms can significantly reduce skewness "
                        f"for columns: {cols[:5]}."
                    ),
                    affected_columns=cols,
                    action_items=[
                        "Apply the recommended transform (Box-Cox or Yeo-Johnson) in preprocessing",
                    ],
                )

    # -- 12. Advanced Correlation -----------------------------------------

    def _advanced_correlation_insights(self) -> None:
        adv = self._stats.advanced_stats
        adv_corr = adv.get("advanced_correlation", {})
        if not adv_corr:
            return

        # Non-linear dependencies via MI
        mi = adv_corr.get("mutual_information")
        pearson = self._stats.correlation_matrix
        if mi is not None and not mi.empty and not pearson.empty:
            # Find pairs with high MI but low Pearson (non-linear relationship)
            mi_cols = set(mi.columns) & set(pearson.columns)
            nonlinear_pairs = []
            for c1 in mi_cols:
                for c2 in mi_cols:
                    if c1 >= c2:
                        continue
                    mi_val = mi.loc[c1, c2] if c1 in mi.index and c2 in mi.columns else 0
                    p_val = abs(pearson.loc[c1, c2]) if c1 in pearson.index and c2 in pearson.columns else 0
                    if mi_val > 0.3 and p_val < 0.3:
                        nonlinear_pairs.append((c1, c2, float(mi_val), float(p_val)))

            if nonlinear_pairs:
                nonlinear_pairs.sort(key=lambda x: x[2], reverse=True)
                top = nonlinear_pairs[0]
                self._add(
                    type=InsightType.FINDING,
                    severity=Severity.HIGH,
                    category="correlation",
                    title=f"{len(nonlinear_pairs)} non-linear dependency pair(s) detected",
                    description=(
                        f"High mutual information but low Pearson correlation suggests non-linear "
                        f"relationships. Top: '{top[0]}' ↔ '{top[1]}' (MI={top[2]:.2f}, r={top[3]:.2f})."
                    ),
                    affected_columns=[top[0], top[1]],
                    evidence={"nonlinear_pairs": nonlinear_pairs[:5]},
                    action_items=[
                        "Use non-linear models (tree-based, kernel) to capture these relationships",
                        "Consider polynomial or interaction features",
                    ],
                )

        # Confounded correlations (partial vs raw)
        pcorr = adv_corr.get("partial_correlation")
        if pcorr is not None and not pcorr.empty and not pearson.empty:
            confounded = []
            pcorr_cols = set(pcorr.columns) & set(pearson.columns)
            for c1 in pcorr_cols:
                for c2 in pcorr_cols:
                    if c1 >= c2:
                        continue
                    raw = pearson.loc[c1, c2] if c1 in pearson.index and c2 in pearson.columns else 0
                    part = pcorr.loc[c1, c2] if c1 in pcorr.index and c2 in pcorr.columns else 0
                    if abs(raw) > 0.5 and abs(raw - part) > 0.3:
                        confounded.append((c1, c2, float(raw), float(part)))

            if confounded:
                confounded.sort(key=lambda x: abs(x[2] - x[3]), reverse=True)
                top = confounded[0]
                self._add(
                    type=InsightType.FINDING,
                    severity=Severity.HIGH,
                    category="correlation",
                    title=f"{len(confounded)} likely confounded correlation(s) detected",
                    description=(
                        f"Raw correlation differs significantly from partial correlation, "
                        f"suggesting confounding variables. "
                        f"Top: '{top[0]}' ↔ '{top[1]}' (raw r={top[2]:.2f}, partial r={top[3]:.2f})."
                    ),
                    affected_columns=[top[0], top[1]],
                    evidence={"confounded_pairs": confounded[:5]},
                    action_items=[
                        "Do not assume causal relationship from raw correlation for these pairs",
                        "Investigate which variables are confounders",
                    ],
                )

        # Bootstrap CI stability
        bci = adv_corr.get("bootstrap_ci")
        if bci is not None and not bci.empty and "ci_width" in bci.columns:
            unstable = bci[bci["ci_width"] > 0.4]
            if not unstable.empty:
                self._add(
                    type=InsightType.WARNING,
                    severity=Severity.MEDIUM,
                    category="correlation",
                    title=f"{len(unstable)} correlation estimate(s) with wide bootstrap CI",
                    description=(
                        "Correlation confidence intervals wider than 0.4 indicate "
                        "unreliable estimates — possibly due to small sample or outliers."
                    ),
                    evidence={"unstable_count": len(unstable)},
                    action_items=[
                        "Treat these correlations with caution",
                        "Consider collecting more data or removing outliers",
                    ],
                )

    # -- 13. General / Cross-Cutting --------------------------------------

    def _general_insights(self) -> None:
        n_rows = self._schema.n_rows
        n_cols = self._schema.n_cols
        n_num = len(self._schema.numeric_columns)
        n_cat = len(self._schema.categorical_columns)

        # Curse of dimensionality
        if n_cols > 0 and n_rows / n_cols < 10:
            self._add(
                type=InsightType.WARNING,
                severity=Severity.HIGH,
                category="general",
                title=f"Low sample-to-feature ratio ({n_rows / n_cols:.1f}:1)",
                description=(
                    f"With {n_rows} rows and {n_cols} columns, the sample-to-feature ratio is low. "
                    "This raises overfitting risk in ML models."
                ),
                evidence={"n_rows": n_rows, "n_cols": n_cols, "ratio": n_rows / n_cols},
                action_items=[
                    "Apply dimensionality reduction (PCA, feature selection) before modelling",
                    "Use regularization (L1/L2) or simpler models",
                    "Collect more data if possible",
                ],
            )

        # Very small dataset
        if n_rows < 50:
            self._add(
                type=InsightType.WARNING,
                severity=Severity.HIGH,
                category="general",
                title=f"Very small dataset ({n_rows} rows)",
                description=(
                    "Statistical tests and ML models may be unreliable with so few samples. "
                    "Confidence intervals will be wide."
                ),
                evidence={"n_rows": n_rows},
                action_items=[
                    "Use cross-validation with appropriate folds (e.g., leave-one-out for very small n)",
                    "Prefer non-parametric or Bayesian approaches",
                ],
            )

        # All-numeric or all-categorical
        if n_num > 0 and n_cat == 0 and n_cols > 3:
            self._add(
                type=InsightType.FINDING,
                severity=Severity.LOW,
                category="general",
                title="Dataset is fully numeric (no categorical columns)",
                description="All columns are numeric, which simplifies preprocessing but may miss categorical patterns.",
                action_items=["Verify no categorical data was inadvertently coded as integers"],
            )
        elif n_cat > 0 and n_num == 0 and n_cols > 3:
            self._add(
                type=InsightType.FINDING,
                severity=Severity.LOW,
                category="general",
                title="Dataset is fully categorical (no numeric columns)",
                description="All columns are categorical. Numeric encoding will be needed for most ML algorithms.",
                action_items=["Plan encoding strategy (one-hot, target, ordinal) for all columns"],
            )
