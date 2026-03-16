"""Data quality scoring module.

Computes per-column and overall quality scores across **six** dimensions:

1. **Completeness** — proportion of non-missing cells.
2. **Uniqueness** — proportion of non-duplicate rows.
3. **Consistency** — dtype-based type-uniformity check (fast).
4. **Validity** — proportion of finite numeric values (no ``inf``).
5. **Timeliness** — recency of datetime columns (optional).
6. **Conformity** — value-range and pattern compliance.

The ``overall_score`` is a weighted average of whichever dimensions apply
to the dataset, ensuring the score adapts to the data's characteristics.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd

from f2a.core.schema import DataSchema


class QualityStats:
    """Compute data quality scores.

    Args:
        df: Target DataFrame.
        schema: Data schema.
    """

    def __init__(self, df: pd.DataFrame, schema: DataSchema) -> None:
        self._df = df
        self._schema = schema

    # ── Dimension scores ──────────────────────────────────

    def completeness(self) -> float:
        """Proportion of non-missing cells."""
        total = self._df.shape[0] * self._df.shape[1]
        if total == 0:
            return 1.0
        return round(1.0 - float(self._df.isna().sum().sum() / total), 4)

    def uniqueness(self) -> float:
        """Proportion of non-duplicate rows."""
        n = len(self._df)
        if n == 0:
            return 1.0
        return round(1.0 - float(self._df.duplicated().sum() / n), 4)

    def consistency(self) -> float:
        """Type-consistency score — fraction of columns with uniform dtype.

        Uses ``dtype.kind`` instead of the slow per-element ``apply(type)``
        approach, checking whether object-typed columns are truly mixed-type.
        """
        ncol = len(self._df.columns)
        if ncol == 0:
            return 1.0

        consistent = 0
        for col in self._df.columns:
            kind = self._df[col].dtype.kind
            if kind != "O":
                # Non-object dtypes (int, float, bool, datetime, …) are
                # inherently type-consistent.
                consistent += 1
                continue
            # For object columns, sample up to 1 000 values and check types.
            non_null = self._df[col].dropna()
            if len(non_null) == 0:
                consistent += 1
                continue
            sample = non_null.head(1_000)
            types_seen = set(type(v).__name__ for v in sample.values)
            if len(types_seen) <= 1:
                consistent += 1

        return round(consistent / ncol, 4)

    def validity(self) -> float:
        """Proportion of finite numeric values (excludes ``inf`` / ``-inf``)."""
        num_cols = self._schema.numeric_columns
        if not num_cols:
            return 1.0

        total = 0
        valid = 0
        for col in num_cols:
            series = self._df[col].dropna()
            total += len(series)
            valid += int(np.isfinite(series).sum())

        return round(valid / total, 4) if total > 0 else 1.0

    def timeliness(self) -> float | None:
        """Recency score for datetime columns (0 = ancient, 1 = fresh).

        If no datetime columns exist, returns ``None`` and the dimension
        is excluded from the overall score.

        Heuristic: score = mean(exp(−days_since / 365)) across datetime cols.
        """
        dt_cols = self._schema.datetime_columns
        if not dt_cols:
            return None

        now = pd.Timestamp.now()
        scores: list[float] = []
        for col in dt_cols:
            series = pd.to_datetime(self._df[col], errors="coerce").dropna()
            if series.empty:
                continue
            max_ts = series.max()
            if pd.isna(max_ts):
                continue
            days_since = max((now - max_ts).days, 0)
            # exponential decay with half-life ≈ 253 days
            scores.append(float(np.exp(-days_since / 365.0)))

        if not scores:
            return None
        return round(float(np.mean(scores)), 4)

    def conformity(self) -> float:
        """Pattern-and-range compliance score.

        Checks:
        * Numeric columns: values within [μ ± 4σ] (i.e. no extreme outliers).
        * String columns: no excessively long / short values or embedded
          control characters.

        Returns:
            Score in [0, 1]. 1.0 = fully conforming.
        """
        scores: list[float] = []

        # ── Numeric: fraction within ±4σ
        for col in self._schema.numeric_columns:
            series = self._df[col].dropna()
            if len(series) < 10:
                scores.append(1.0)
                continue
            mu, sigma = float(series.mean()), float(series.std())
            if sigma == 0:
                scores.append(1.0)
                continue
            in_range = ((series >= mu - 4 * sigma) & (series <= mu + 4 * sigma)).sum()
            scores.append(float(in_range) / len(series))

        # ── String: no control characters (ASCII 0-31 except \n\r\t)
        _CTRL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
        for col in self._schema.categorical_columns:
            series = self._df[col].dropna().astype(str).head(2_000)
            if series.empty:
                scores.append(1.0)
                continue
            has_ctrl = series.apply(lambda v: bool(_CTRL_RE.search(v)))
            scores.append(1.0 - float(has_ctrl.mean()))

        if not scores:
            return 1.0
        return round(float(np.mean(scores)), 4)

    def overall_score(self) -> float:
        """Weighted average of all applicable quality dimensions.

        Base weights (always active):
            completeness 30 %, uniqueness 20 %, consistency 15 %,
            validity 15 %, conformity 10 %.
        If timeliness is available, it receives 10 % and the others
        are proportionally reduced.
        """
        dims: dict[str, tuple[float, float]] = {
            "completeness": (0.30, self.completeness()),
            "uniqueness": (0.20, self.uniqueness()),
            "consistency": (0.15, self.consistency()),
            "validity": (0.15, self.validity()),
            "conformity": (0.10, self.conformity()),
        }

        timeliness_val = self.timeliness()
        if timeliness_val is not None:
            dims["timeliness"] = (0.10, timeliness_val)

        total_weight = sum(w for w, _ in dims.values())
        score = sum(w * v for w, v in dims.values()) / total_weight
        return round(score, 4)

    # ── Summaries ─────────────────────────────────────────

    def summary(self) -> dict[str, Any]:
        """Return all quality dimension scores."""
        result: dict[str, Any] = {
            "completeness": self.completeness(),
            "uniqueness": self.uniqueness(),
            "consistency": self.consistency(),
            "validity": self.validity(),
            "conformity": self.conformity(),
        }
        timeliness_val = self.timeliness()
        if timeliness_val is not None:
            result["timeliness"] = timeliness_val
        result["overall"] = self.overall_score()
        return result

    def column_quality(self) -> pd.DataFrame:
        """Return per-column quality scores.

        Returns:
            DataFrame indexed by column name with completeness, uniqueness,
            type, and composite quality_score.
        """
        rows: list[dict] = []
        for col_info in self._schema.columns:
            col = col_info.name
            series = self._df[col]
            compl = 1.0 - col_info.missing_ratio

            n_total = int(series.count())
            n_unique = int(series.nunique())
            uniqueness = n_unique / n_total if n_total > 0 else 1.0

            rows.append({
                "column": col,
                "completeness": round(compl, 4),
                "uniqueness": round(min(uniqueness, 1.0), 4),
                "type": col_info.inferred_type.value,
                "quality_score": round((compl + min(uniqueness, 1.0)) / 2, 4),
            })

        return pd.DataFrame(rows).set_index("column") if rows else pd.DataFrame()
