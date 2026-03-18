"""Column role classification — auto-detect the semantic role of each column.

Infers whether a column acts as an ID, timestamp, numeric feature, categorical
feature, ordinal feature, binary variable, text field, constant, or potential
target variable.  Each assignment comes with a confidence score and evidence
so downstream consumers (ML readiness, insight engine) can make informed
decisions.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from f2a.core.schema import DataSchema
from f2a.utils.logging import get_logger

logger = get_logger(__name__)

# =====================================================================
#  Data classes
# =====================================================================

@dataclass
class ColumnRole:
    """Role assignment for a single column."""

    column: str
    primary_role: str          # id | timestamp | numeric_feature | categorical_feature | ordinal_feature | binary | text | constant | target_candidate
    confidence: float          # 0-1
    secondary_role: str | None = None
    properties: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "column": self.column,
            "primary_role": self.primary_role,
            "confidence": self.confidence,
            "secondary_role": self.secondary_role,
            "properties": self.properties,
        }


# Regex patterns for name-based heuristics
_ID_PATTERNS = re.compile(
    r"(^id$|_id$|^pk$|^key$|^index$|^uid$|^uuid$|^guid$|^row_?num|^seq)",
    re.IGNORECASE,
)
_TIME_PATTERNS = re.compile(
    r"(date|time|_at$|_ts$|timestamp|created|updated|modified|year|month|day)",
    re.IGNORECASE,
)
_TARGET_PATTERNS = re.compile(
    r"(^target$|^label$|^y$|^class$|^outcome$|^response$|^result$|^is_|^has_)",
    re.IGNORECASE,
)
_ORDINAL_PATTERNS = re.compile(
    r"(level|grade|rating|rank|score|priority|stage|phase|tier|degree)",
    re.IGNORECASE,
)


class ColumnRoleClassifier:
    """Automatically assign a semantic role to every column in the dataset.

    Parameters
    ----------
    df : pd.DataFrame
        The analysis DataFrame.
    schema : DataSchema
        Column type metadata.
    """

    def __init__(self, df: pd.DataFrame, schema: DataSchema) -> None:
        self._df = df
        self._schema = schema

    def classify(self) -> list[ColumnRole]:
        """Return a role assignment for each column."""
        roles: list[ColumnRole] = []
        for col_info in self._schema.columns:
            role = self._classify_single(col_info)
            roles.append(role)
        return roles

    def summary(self) -> pd.DataFrame:
        """Summary table: column × role × confidence."""
        roles = self.classify()
        rows = [r.to_dict() for r in roles]
        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.set_index("column")
        return df

    # ------------------------------------------------------------------

    def _classify_single(self, col_info: Any) -> ColumnRole:
        col_name = col_info.name
        dtype = str(col_info.dtype)
        inferred = col_info.inferred_type   # "numeric", "categorical", "text", "datetime", "boolean"
        n_unique = col_info.n_unique
        n_missing = col_info.n_missing
        n_total = self._schema.n_rows

        unique_ratio = n_unique / max(n_total, 1)

        # 1. Constant
        if n_unique <= 1:
            return ColumnRole(
                column=col_name,
                primary_role="constant",
                confidence=1.0,
                properties={"n_unique": n_unique},
            )

        # 2. Binary
        if n_unique == 2:
            conf = 0.9
            secondary = None
            if _TARGET_PATTERNS.search(col_name):
                secondary = "target_candidate"
                conf = 0.85
            return ColumnRole(
                column=col_name,
                primary_role="binary",
                confidence=conf,
                secondary_role=secondary,
                properties={"n_unique": 2, "values": self._top_values(col_name, 2)},
            )

        # 3. Datetime / timestamp
        if inferred == "datetime":
            return ColumnRole(
                column=col_name,
                primary_role="timestamp",
                confidence=0.95,
                properties={"dtype": dtype},
            )
        if _TIME_PATTERNS.search(col_name) and inferred == "numeric":
            # Possibly an epoch timestamp
            series = pd.to_numeric(self._df[col_name], errors="coerce").dropna()
            if not series.empty and self._is_monotonic(series):
                return ColumnRole(
                    column=col_name,
                    primary_role="timestamp",
                    confidence=0.70,
                    properties={"dtype": dtype, "hint": "monotonic numeric with time-like name"},
                )

        # 4. ID-like
        if self._is_id_like(col_name, unique_ratio, n_unique, inferred):
            conf = 0.6
            if _ID_PATTERNS.search(col_name):
                conf = 0.9
            elif unique_ratio > 0.99:
                conf = 0.85
            return ColumnRole(
                column=col_name,
                primary_role="id",
                confidence=conf,
                properties={"unique_ratio": round(unique_ratio, 4)},
            )

        # 5. Text
        if inferred == "text":
            return ColumnRole(
                column=col_name,
                primary_role="text",
                confidence=0.9,
                properties={"avg_length": self._avg_str_length(col_name)},
            )

        # 6. Ordinal feature
        if self._is_ordinal(col_name, inferred, n_unique, n_total):
            conf = 0.7
            if _ORDINAL_PATTERNS.search(col_name):
                conf = 0.85
            return ColumnRole(
                column=col_name,
                primary_role="ordinal_feature",
                confidence=conf,
                properties={"n_unique": n_unique},
            )

        # 7. Target candidate (categorical with specific naming)
        if _TARGET_PATTERNS.search(col_name) and n_unique <= 20:
            return ColumnRole(
                column=col_name,
                primary_role="target_candidate",
                confidence=0.7,
                properties={"n_unique": n_unique, "inferred_type": inferred},
            )

        # 8. Categorical feature
        if inferred == "categorical" or inferred == "boolean":
            return ColumnRole(
                column=col_name,
                primary_role="categorical_feature",
                confidence=0.85,
                properties={"n_unique": n_unique, "unique_ratio": round(unique_ratio, 4)},
            )

        # 9. Numeric feature (default for numeric)
        if inferred == "numeric":
            secondary = None
            if _TARGET_PATTERNS.search(col_name):
                secondary = "target_candidate"
            return ColumnRole(
                column=col_name,
                primary_role="numeric_feature",
                confidence=0.85,
                secondary_role=secondary,
                properties={"dtype": dtype},
            )

        # Fallback
        return ColumnRole(
            column=col_name,
            primary_role="numeric_feature" if inferred == "numeric" else "categorical_feature",
            confidence=0.5,
            properties={"inferred_type": inferred},
        )

    # ------------------------------------------------------------------
    #  Heuristics
    # ------------------------------------------------------------------

    def _is_id_like(self, col_name: str, unique_ratio: float, n_unique: int, inferred: str) -> bool:
        if _ID_PATTERNS.search(col_name):
            return unique_ratio > 0.8
        if unique_ratio > 0.95 and n_unique > 20:
            if inferred in ("text", "categorical"):
                return True
            if inferred == "numeric":
                series = pd.to_numeric(self._df[col_name], errors="coerce").dropna()
                if not series.empty and self._is_monotonic(series):
                    return True
        return False

    @staticmethod
    def _is_monotonic(series: pd.Series) -> bool:
        """Check if a numeric series is (roughly) monotonic."""
        if len(series) < 5:
            return False
        diffs = series.diff().dropna()
        if diffs.empty:
            return False
        pos = (diffs >= 0).sum()
        neg = (diffs <= 0).sum()
        ratio = max(pos, neg) / len(diffs)
        return ratio > 0.95

    def _is_ordinal(self, col_name: str, inferred: str, n_unique: int, n_total: int) -> bool:
        """Heuristic: integer column with small distinct count and ordinal-like name."""
        if n_unique > 20 or n_unique < 3:
            return False
        if _ORDINAL_PATTERNS.search(col_name):
            return True
        if inferred == "numeric" and col_name in self._df.columns:
            series = pd.to_numeric(self._df[col_name], errors="coerce").dropna()
            if not series.empty and (series == series.astype(int)).all():
                vals = sorted(series.unique())
                if len(vals) <= 15:
                    # Check if values are roughly consecutive
                    span = vals[-1] - vals[0]
                    if span > 0 and len(vals) / (span + 1) > 0.5:
                        return True
        return False

    def _top_values(self, col_name: str, n: int = 5) -> list:
        if col_name not in self._df.columns:
            return []
        return self._df[col_name].dropna().value_counts().head(n).index.tolist()

    def _avg_str_length(self, col_name: str) -> float:
        if col_name not in self._df.columns:
            return 0.0
        s = self._df[col_name].dropna().astype(str)
        return round(float(s.str.len().mean()), 1) if not s.empty else 0.0
