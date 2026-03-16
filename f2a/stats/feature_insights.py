"""Feature engineering insights module.

Provides interaction detection, monotonic relationship analysis,
optimal binning, cardinality analysis, and data leakage detection.

References:
    - Friedman & Popescu (2008) — interaction detection
    - Fayyad & Irani (1993) — entropy-based binning
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from f2a.core.schema import DataSchema
from f2a.utils.logging import get_logger

logger = get_logger(__name__)


class FeatureInsightsStats:
    """Feature engineering recommendations and insights.

    Args:
        df: Target DataFrame.
        schema: Data schema.
        max_sample: Max rows to sample.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        schema: DataSchema,
        max_sample: int = 5000,
    ) -> None:
        self._df = df
        self._schema = schema
        self._max_sample = max_sample

    # ── Interaction detection ─────────────────────────────

    def interaction_detection(self) -> pd.DataFrame:
        """Detect potential feature interactions.

        For each pair of numeric features, computes the correlation of
        their product with each feature individually.  High product-
        correlation suggests a meaningful interaction term.

        Returns:
            DataFrame with col_a, col_b, interaction_strength, and
            correlation of the product with each original feature.
        """
        cols = self._schema.numeric_columns
        if len(cols) < 2:
            return pd.DataFrame()

        cols = cols[:15]
        df_clean = self._df[cols].dropna()
        if len(df_clean) < 30:
            return pd.DataFrame()

        if len(df_clean) > self._max_sample:
            df_clean = df_clean.sample(self._max_sample, random_state=42)

        rows: list[dict] = []
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                a = df_clean[cols[i]]
                b = df_clean[cols[j]]

                # Product interaction
                product = a * b
                if product.std() == 0 or a.std() == 0 or b.std() == 0:
                    continue

                # How much does the product correlate beyond individual features?
                r_prod_a = float(product.corr(a))
                r_prod_b = float(product.corr(b))
                r_ab = float(a.corr(b))

                # Interaction strength: residual correlation after removing linear
                interaction_strength = max(abs(r_prod_a), abs(r_prod_b)) - abs(r_ab)

                if abs(interaction_strength) > 0.1:
                    rows.append({
                        "col_a": cols[i],
                        "col_b": cols[j],
                        "interaction_strength": round(interaction_strength, 4),
                        "corr_product_a": round(r_prod_a, 4),
                        "corr_product_b": round(r_prod_b, 4),
                        "corr_a_b": round(r_ab, 4),
                        "recommendation": (
                            "Strong interaction"
                            if interaction_strength > 0.3
                            else "Moderate interaction"
                        ),
                    })

        if not rows:
            return pd.DataFrame()

        return pd.DataFrame(rows).sort_values(
            "interaction_strength", ascending=False
        ).reset_index(drop=True)

    # ── Monotonic relationship detection ──────────────────

    def monotonic_detection(self) -> pd.DataFrame:
        """Detect monotonic relationships using Spearman correlation.

        A high |Spearman| but low |Pearson| suggests a non-linear
        monotonic relationship.

        Returns:
            DataFrame with col_a, col_b, pearson, spearman, monotonic_gap.
        """
        cols = self._schema.numeric_columns
        if len(cols) < 2:
            return pd.DataFrame()

        cols = cols[:20]
        df_clean = self._df[cols].dropna()
        if len(df_clean) < 20:
            return pd.DataFrame()

        pearson = df_clean.corr(method="pearson")
        spearman = df_clean.corr(method="spearman")

        rows: list[dict] = []
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                r_p = float(pearson.iloc[i, j])
                r_s = float(spearman.iloc[i, j])
                gap = abs(r_s) - abs(r_p)

                if gap > 0.05 and abs(r_s) > 0.3:
                    rows.append({
                        "col_a": cols[i],
                        "col_b": cols[j],
                        "pearson": round(r_p, 4),
                        "spearman": round(r_s, 4),
                        "monotonic_gap": round(gap, 4),
                        "relationship": (
                            "Strong non-linear monotonic"
                            if gap > 0.15
                            else "Moderate non-linear monotonic"
                        ),
                    })

        if not rows:
            return pd.DataFrame()

        return pd.DataFrame(rows).sort_values(
            "monotonic_gap", ascending=False
        ).reset_index(drop=True)

    # ── Binning analysis ──────────────────────────────────

    def binning_analysis(self, n_bins: int = 10) -> pd.DataFrame:
        """Analyze optimal binning for numeric columns.

        Computes equal-width and equal-frequency binning, then evaluates
        the entropy of each binning to recommend the best strategy.

        Args:
            n_bins: Number of bins.

        Returns:
            DataFrame with binning statistics per column.
        """
        cols = self._schema.numeric_columns
        if not cols:
            return pd.DataFrame()

        rows: list[dict] = []
        for col in cols:
            series = self._df[col].dropna()
            if len(series) < n_bins:
                continue

            # Equal-width binning
            try:
                ew_bins = pd.cut(series, bins=n_bins)
                ew_counts = ew_bins.value_counts(normalize=True)
                ew_entropy = float(-np.sum(
                    ew_counts * np.log2(ew_counts + 1e-15)
                ))
            except Exception:
                ew_entropy = None

            # Equal-frequency binning
            try:
                ef_bins = pd.qcut(series, q=n_bins, duplicates="drop")
                ef_counts = ef_bins.value_counts(normalize=True)
                ef_entropy = float(-np.sum(
                    ef_counts * np.log2(ef_counts + 1e-15)
                ))
            except Exception:
                ef_entropy = None

            max_entropy = float(np.log2(n_bins))

            recommendation = "equal_frequency"  # default
            if ew_entropy is not None and ef_entropy is not None:
                if ew_entropy > ef_entropy * 0.95:
                    recommendation = "equal_width"

            rows.append({
                "column": col,
                "n_bins": n_bins,
                "equal_width_entropy": round(ew_entropy, 4) if ew_entropy else None,
                "equal_freq_entropy": round(ef_entropy, 4) if ef_entropy else None,
                "max_entropy": round(max_entropy, 4),
                "recommended_method": recommendation,
                "skewness": round(float(series.skew()), 4),
            })

        return pd.DataFrame(rows).set_index("column") if rows else pd.DataFrame()

    # ── Cardinality analysis ──────────────────────────────

    def cardinality_analysis(self) -> pd.DataFrame:
        """Analyze cardinality of all columns for encoding recommendations.

        Returns:
            DataFrame with cardinality stats and encoding recommendations.
        """
        rows: list[dict] = []
        for col in self._df.columns:
            series = self._df[col]
            n_unique = int(series.nunique())
            n_total = int(series.count())
            ratio = n_unique / n_total if n_total > 0 else 0.0

            # Determine type recommendation
            if ratio > 0.95:
                encoding = "id_column (drop or hash)"
            elif n_unique <= 2:
                encoding = "binary encoding"
            elif n_unique <= 10:
                encoding = "one-hot encoding"
            elif n_unique <= 50:
                encoding = "label encoding or target encoding"
            elif n_unique <= 500:
                encoding = "target encoding or frequency encoding"
            else:
                encoding = "hash encoding or embeddings"

            rows.append({
                "column": col,
                "n_unique": n_unique,
                "n_total": n_total,
                "cardinality_ratio": round(ratio, 4),
                "cardinality_level": (
                    "binary" if n_unique <= 2
                    else "low" if n_unique <= 10
                    else "medium" if n_unique <= 50
                    else "high" if n_unique <= 500
                    else "very_high"
                ),
                "recommended_encoding": encoding,
            })

        return pd.DataFrame(rows).set_index("column") if rows else pd.DataFrame()

    # ── Leakage detection ─────────────────────────────────

    def leakage_detection(self) -> pd.DataFrame:
        """Detect potential data leakage indicators.

        Flags columns with:
        - Perfect or near-perfect correlation with other columns
        - Suspiciously high unique ratio (possible target leak)
        - Constant or near-constant values

        Returns:
            DataFrame with leakage risk assessment per column.
        """
        cols = self._schema.numeric_columns
        all_cols = list(self._df.columns)

        rows: list[dict] = []

        for col in all_cols:
            series = self._df[col]
            n_total = int(series.count())
            n_unique = int(series.nunique())
            ratio = n_unique / n_total if n_total > 0 else 0

            risks: list[str] = []

            # Near-constant
            if n_unique <= 1:
                risks.append("constant_column")
            elif n_unique == 2 and n_total > 100:
                top_freq = series.value_counts().iloc[0] / n_total
                if top_freq > 0.99:
                    risks.append("near_constant")

            # ID-like
            if ratio > 0.95 and n_total > 100:
                risks.append("id_like")

            # Perfect correlation with another column
            if col in cols:
                for other in cols:
                    if other == col:
                        continue
                    try:
                        r = abs(float(self._df[col].corr(self._df[other])))
                        if r > 0.99:
                            risks.append(f"perfect_corr_with_{other}")
                            break
                    except Exception:
                        continue

            risk_level = (
                "high" if len(risks) >= 2
                else "medium" if len(risks) == 1
                else "low"
            )

            if risks:
                rows.append({
                    "column": col,
                    "risk_level": risk_level,
                    "risks": "; ".join(risks),
                    "unique_ratio": round(ratio, 4),
                    "n_unique": n_unique,
                })

        return pd.DataFrame(rows).set_index("column") if rows else pd.DataFrame()

    # ── Summary ───────────────────────────────────────────

    def summary(self) -> dict[str, Any]:
        """Return combined feature insight results."""
        result: dict[str, Any] = {}

        try:
            inter = self.interaction_detection()
            if not inter.empty:
                result["interactions"] = inter
        except Exception as exc:
            logger.debug("Interaction detection skipped: %s", exc)

        try:
            mono = self.monotonic_detection()
            if not mono.empty:
                result["monotonic"] = mono
        except Exception as exc:
            logger.debug("Monotonic detection skipped: %s", exc)

        try:
            bins = self.binning_analysis()
            if not bins.empty:
                result["binning"] = bins
        except Exception as exc:
            logger.debug("Binning analysis skipped: %s", exc)

        try:
            card = self.cardinality_analysis()
            if not card.empty:
                result["cardinality"] = card
        except Exception as exc:
            logger.debug("Cardinality analysis skipped: %s", exc)

        try:
            leak = self.leakage_detection()
            if not leak.empty:
                result["leakage"] = leak
        except Exception as exc:
            logger.debug("Leakage detection skipped: %s", exc)

        return result
