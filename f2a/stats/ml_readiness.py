"""ML Readiness Evaluator — multi-dimensional assessment of dataset fitness.

Evaluates a dataset across six dimensions to produce a composite *readiness
score* and letter grade, together with blocking issues that **must** be resolved
and improvement suggestions that **should** be considered before feeding the
data into a machine learning pipeline.

Dimensions
----------
1. **Completeness** — missing value burden
2. **Consistency** — type homogeneity, value-range sanity
3. **Balance** — class / category imbalance, outlier skew
4. **Informativeness** — variance, uniqueness, MI content
5. **Independence** — multicollinearity (VIF / high-r)
6. **Scale** — sample-to-feature ratio, curse of dimensionality
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from f2a.core.schema import DataSchema
from f2a.utils.logging import get_logger

logger = get_logger(__name__)

# Grade thresholds
_GRADES = [
    (95, "A+"), (90, "A"), (85, "B+"), (80, "B"),
    (75, "C+"), (70, "C"), (60, "D"), (0, "F"),
]


def _to_grade(score: float) -> str:
    for threshold, grade in _GRADES:
        if score >= threshold:
            return grade
    return "F"


@dataclass
class ReadinessScore:
    """ML readiness evaluation result."""

    overall: float                     # 0-100
    grade: str                         # A+, A, B+, B, C+, C, D, F
    dimensions: dict[str, float]       # each 0-100
    blocking_issues: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "overall": round(self.overall, 1),
            "grade": self.grade,
            "dimensions": {k: round(v, 1) for k, v in self.dimensions.items()},
            "blocking_issues": self.blocking_issues,
            "suggestions": self.suggestions,
            "details": self.details,
        }


class MLReadinessEvaluator:
    """Evaluate the ML-readiness of a dataset from pre-computed stats.

    Parameters
    ----------
    df : pd.DataFrame
        The (cleaned) analysis DataFrame.
    schema : DataSchema
        Type metadata.
    stats : StatsResult
        All pre-computed statistical results (basic + advanced).
    column_roles : pd.DataFrame | None
        Output of ``ColumnRoleClassifier.summary()`` (optional).
    """

    # Dimension weights — must sum to 1.0
    _WEIGHTS = {
        "completeness": 0.25,
        "consistency": 0.15,
        "balance": 0.15,
        "informativeness": 0.20,
        "independence": 0.15,
        "scale": 0.10,
    }

    def __init__(
        self,
        df: pd.DataFrame,
        schema: DataSchema,
        stats: Any,
        column_roles: pd.DataFrame | None = None,
    ) -> None:
        self._df = df
        self._schema = schema
        self._stats = stats
        self._roles = column_roles
        self._blocking: list[str] = []
        self._suggestions: list[str] = []

    def evaluate(self) -> ReadinessScore:
        """Compute the overall ML readiness score."""
        dims: dict[str, float] = {}
        details: dict[str, Any] = {}

        for name, method in [
            ("completeness", self._completeness),
            ("consistency", self._consistency),
            ("balance", self._balance),
            ("informativeness", self._informativeness),
            ("independence", self._independence),
            ("scale", self._scale),
        ]:
            try:
                score, det = method()
                dims[name] = max(0.0, min(100.0, score))
                details[name] = det
            except Exception as exc:
                logger.debug("ML readiness dimension '%s' failed: %s", name, exc)
                dims[name] = 50.0  # neutral fallback
                details[name] = {"error": str(exc)}

        overall = sum(dims[d] * self._WEIGHTS[d] for d in dims)
        grade = _to_grade(overall)

        return ReadinessScore(
            overall=round(overall, 1),
            grade=grade,
            dimensions=dims,
            blocking_issues=self._blocking,
            suggestions=self._suggestions,
            details=details,
        )

    # ==================================================================
    #  Dimension scorers — each returns (score_0_100, detail_dict)
    # ==================================================================

    def _completeness(self) -> tuple[float, dict]:
        mi = self._stats.missing_info
        detail: dict[str, Any] = {}

        if mi.empty or "missing_ratio" not in mi.columns:
            return 100.0, {"no_missing_info": True}

        ratios = mi["missing_ratio"]
        overall_miss = float(ratios.mean())
        high_miss_cols = list(mi[ratios > 0.5].index)
        mod_miss_cols = list(mi[(ratios > 0.1) & (ratios <= 0.5)].index)

        detail["overall_missing_rate"] = round(overall_miss, 4)
        detail["high_missing_columns"] = high_miss_cols[:10]
        detail["moderate_missing_columns"] = mod_miss_cols[:10]

        if high_miss_cols:
            self._blocking.append(
                f"{len(high_miss_cols)} column(s) have >50% missing — drop or impute: "
                f"{', '.join(high_miss_cols[:5])}"
            )

        if mod_miss_cols:
            self._suggestions.append(
                f"{len(mod_miss_cols)} column(s) have 10-50% missing — plan imputation strategy"
            )

        # Score: 100 if 0 missing, linearly degrade
        score = max(0, 100 * (1 - overall_miss * 2))  # 50% average missing → 0
        return score, detail

    def _consistency(self) -> tuple[float, dict]:
        detail: dict[str, Any] = {}
        penalties = 0.0
        n_cols = self._schema.n_cols

        # Mixed types from preprocessing
        pp = self._stats.preprocessing
        mixed = len(pp.mixed_type_columns) if pp else 0
        inf_cols = len(pp.infinite_value_columns) if pp else 0

        detail["mixed_type_columns"] = mixed
        detail["infinite_value_columns"] = inf_cols

        if mixed > 0:
            penalties += (mixed / max(n_cols, 1)) * 40
            self._suggestions.append(f"{mixed} mixed-type column(s) — cast to consistent types")

        if inf_cols > 0:
            penalties += (inf_cols / max(n_cols, 1)) * 20
            self._suggestions.append(f"{inf_cols} column(s) contain infinity values — replace with NaN or cap")

        # ID-like columns that shouldn't be features
        if self._roles is not None and not self._roles.empty:
            ids = self._roles[self._roles["primary_role"] == "id"]
            if not ids.empty:
                detail["id_columns"] = list(ids.index)
                self._suggestions.append(
                    f"Remove {len(ids)} ID-like column(s) before modelling: "
                    f"{', '.join(list(ids.index)[:5])}"
                )

        # Constants
        if self._roles is not None and not self._roles.empty:
            constants = self._roles[self._roles["primary_role"] == "constant"]
            if not constants.empty:
                penalties += len(constants) / max(n_cols, 1) * 20
                self._blocking.append(
                    f"{len(constants)} constant column(s) — remove before modelling"
                )

        score = max(0, 100 - penalties)
        return score, detail

    def _balance(self) -> tuple[float, dict]:
        detail: dict[str, Any] = {}
        penalties = 0.0

        # Outlier ratio
        out = self._stats.outlier_summary
        if not out.empty and "outlier_%" in out.columns:
            avg_outlier = float(out["outlier_%"].mean())
            detail["avg_outlier_pct"] = round(avg_outlier, 2)
            if avg_outlier > 20:
                penalties += 30
                self._suggestions.append("High average outlier rate — consider winsorization or robust methods")
            elif avg_outlier > 10:
                penalties += 15

        # Categorical imbalance (Gini index)
        cat_cols = self._schema.categorical_columns[:20]
        if cat_cols:
            ginis = []
            for col in cat_cols:
                if col in self._df.columns:
                    vc = self._df[col].value_counts(normalize=True).values
                    gini = 1 - np.sum(vc ** 2)
                    ginis.append(gini)
            if ginis:
                avg_gini = float(np.mean(ginis))
                detail["avg_categorical_gini"] = round(avg_gini, 4)
                # Low Gini means imbalanced
                if avg_gini < 0.3:
                    penalties += 20
                    self._suggestions.append(
                        "Categorical columns are highly imbalanced — consider SMOTE or class weighting"
                    )

        score = max(0, 100 - penalties)
        return score, detail

    def _informativeness(self) -> tuple[float, dict]:
        detail: dict[str, Any] = {}
        penalties = 0.0
        n_cols = self._schema.n_cols

        # Duplicate ratio
        dup = self._stats.duplicate_stats
        dup_ratio = dup.get("duplicate_ratio", 0) if dup else 0
        detail["duplicate_ratio"] = round(dup_ratio, 4)
        if dup_ratio > 0.2:
            penalties += 25
            self._blocking.append(f"{dup_ratio * 100:.0f}% duplicate rows — remove before modelling")
        elif dup_ratio > 0.05:
            penalties += 10
            self._suggestions.append("Some duplicate rows exist — verify they are intentional")

        # Low-variance features (constant or near-constant)
        summary = self._stats.summary
        if not summary.empty and "cv" in summary.columns:
            near_const = summary[(summary["cv"].notna()) & (summary["cv"].abs() < 0.01)]
            if not near_const.empty:
                penalties += (len(near_const) / max(n_cols, 1)) * 30
                detail["near_constant_columns"] = list(near_const.index)[:10]
                self._suggestions.append(
                    f"{len(near_const)} near-constant column(s) carry very little information"
                )

        # PCA compressibility (high reduction = redundancy penalty, but also okay)
        pca_sum = self._stats.pca_summary
        if pca_sum:
            comp90 = pca_sum.get("components_for_90pct", 0)
            n_num = len(self._schema.numeric_columns)
            if n_num > 0 and comp90 > 0:
                compression = comp90 / n_num
                detail["pca_compression"] = round(compression, 3)
                if compression < 0.3:
                    # very compressible → lots of redundancy
                    penalties += 10
                    self._suggestions.append(
                        f"90% variance in just {comp90}/{n_num} PCs — consider PCA for dimensionality reduction"
                    )

        score = max(0, 100 - penalties)
        return score, detail

    def _independence(self) -> tuple[float, dict]:
        detail: dict[str, Any] = {}
        penalties = 0.0

        vif = self._stats.vif_table
        if not vif.empty and "VIF" in vif.columns:
            severe = vif[vif["VIF"] > 10]
            moderate = vif[(vif["VIF"] > 5) & (vif["VIF"] <= 10)]
            detail["severe_vif_columns"] = list(severe.index)[:10]
            detail["moderate_vif_columns"] = list(moderate.index)[:10]

            if not severe.empty:
                worst_vif = float(severe["VIF"].max())
                penalties += min(50, len(severe) * 10)
                if worst_vif > 100:
                    self._blocking.append(
                        f"Extreme multicollinearity: VIF={worst_vif:.0f} for '{severe['VIF'].idxmax()}' — remove or combine"
                    )
                else:
                    self._suggestions.append(
                        f"{len(severe)} column(s) with VIF>10 — consider regularization or PCA"
                    )

            if not moderate.empty:
                penalties += len(moderate) * 3

        # High-correlation pairs
        corr = self._stats.correlation_matrix
        if not corr.empty:
            n_high = 0
            cols_list = corr.columns.tolist()
            for i, c1 in enumerate(cols_list):
                for c2 in cols_list[i + 1:]:
                    if abs(corr.loc[c1, c2]) > 0.95:
                        n_high += 1
            if n_high > 0:
                detail["near_perfect_pairs"] = n_high
                penalties += min(30, n_high * 5)

        score = max(0, 100 - penalties)
        return score, detail

    def _scale(self) -> tuple[float, dict]:
        detail: dict[str, Any] = {}
        n_rows = self._schema.n_rows
        n_features = len(self._schema.numeric_columns) + len(self._schema.categorical_columns)

        ratio = n_rows / max(n_features, 1)
        detail["sample_feature_ratio"] = round(ratio, 1)
        detail["n_rows"] = n_rows
        detail["n_features"] = n_features

        if ratio < 5:
            self._blocking.append(
                f"Sample-to-feature ratio is {ratio:.1f}:1 — very high overfitting risk"
            )
            score = max(0, ratio / 5 * 50)
        elif ratio < 10:
            self._suggestions.append(
                f"Sample-to-feature ratio ({ratio:.0f}:1) is low — use regularization"
            )
            score = 50 + (ratio - 5) / 5 * 30
        elif ratio < 20:
            score = 80 + (ratio - 10) / 10 * 15
        else:
            score = min(100, 95 + min(ratio / 100, 1) * 5)

        return score, detail
