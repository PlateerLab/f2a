"""Cross-dimensional analysis — discovers patterns across analysis boundaries.

Instead of treating each analysis (correlation, cluster, outlier, missing, …)
in isolation, this module crosses two or more dimensions to reveal composite
patterns that single-axis analyses miss:

* **Outlier × Cluster**: Are anomalies concentrated in specific clusters?
* **Missing × Correlation**: Is missingness systematic (MAR) or random (MCAR)?
* **Distribution × Outlier**: Which outlier method is appropriate given tail shape?
* **Cluster × Correlation (Simpson's Paradox)**: Does aggregation mask reversed relationships?
* **Feature Importance × Missing**: Are critical features losing information?
* **Dim-Reduction × Cluster × Anomaly**: Unified 2-D embedding overlay.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from f2a.core.schema import DataSchema
from f2a.utils.logging import get_logger

logger = get_logger(__name__)


class CrossAnalysis:
    """Run all cross-dimensional analyses given pre-computed stats.

    Parameters
    ----------
    df : pd.DataFrame
        The (cleaned) analysis DataFrame.
    schema : DataSchema
        Inferred schema.
    stats : StatsResult
        Previously computed statistical results (basic + advanced).
    max_cols : int
        Maximum numeric columns to consider in expensive pairwise ops.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        schema: DataSchema,
        stats: Any,
        *,
        max_cols: int = 20,
    ) -> None:
        self._df = df
        self._schema = schema
        self._stats = stats
        self._max_cols = max_cols

    # ------------------------------------------------------------------
    #  Public
    # ------------------------------------------------------------------

    def summary(self) -> dict[str, Any]:
        """Execute all cross-analyses and return a combined dict."""
        result: dict[str, Any] = {}

        try:
            r = self.outlier_by_cluster()
            if r is not None:
                result["outlier_by_cluster"] = r
        except Exception as exc:
            logger.debug("outlier_by_cluster failed: %s", exc)

        try:
            r = self.missing_correlation()
            if r is not None:
                result["missing_correlation"] = r
        except Exception as exc:
            logger.debug("missing_correlation failed: %s", exc)

        try:
            r = self.distribution_outlier_fitness()
            if r is not None:
                result["distribution_outlier_fitness"] = r
        except Exception as exc:
            logger.debug("distribution_outlier_fitness failed: %s", exc)

        try:
            r = self.simpson_paradox()
            if r is not None:
                result["simpson_paradox"] = r
        except Exception as exc:
            logger.debug("simpson_paradox failed: %s", exc)

        try:
            r = self.importance_vs_missing()
            if r is not None:
                result["importance_vs_missing"] = r
        except Exception as exc:
            logger.debug("importance_vs_missing failed: %s", exc)

        try:
            r = self.unified_2d_embedding()
            if r is not None:
                result["unified_embedding"] = r
        except Exception as exc:
            logger.debug("unified_2d_embedding failed: %s", exc)

        return result

    # ------------------------------------------------------------------
    #  X1. Outlier × Cluster
    # ------------------------------------------------------------------

    def outlier_by_cluster(self) -> dict[str, Any] | None:
        """Per-cluster anomaly rates from consensus anomaly + K-Means labels."""
        adv = self._stats.advanced_stats
        clustering = adv.get("clustering", {})
        anomaly_full = adv.get("advanced_anomaly_full", {})

        km = clustering.get("kmeans")
        iso = anomaly_full.get("isolation_forest")
        if not km or not iso:
            return None

        labels_cluster = km.get("labels")
        labels_anomaly = iso.get("labels")
        if labels_cluster is None or labels_anomaly is None:
            return None

        # Align lengths (both should be n_samples after sampling)
        n = min(len(labels_cluster), len(labels_anomaly))
        if n == 0:
            return None

        c_labels = np.asarray(labels_cluster[:n])
        a_labels = np.asarray(labels_anomaly[:n])

        anomaly_mask = a_labels == -1
        unique_clusters = np.unique(c_labels)

        rows = []
        for cl in unique_clusters:
            cl_mask = c_labels == cl
            cl_size = int(cl_mask.sum())
            cl_anomalies = int((cl_mask & anomaly_mask).sum())
            rows.append({
                "cluster": f"cluster_{cl}" if cl >= 0 else "noise",
                "size": cl_size,
                "anomaly_count": cl_anomalies,
                "anomaly_rate": round(cl_anomalies / max(cl_size, 1), 4),
            })

        df_result = pd.DataFrame(rows)

        # Chi-square test for uniform anomaly distribution
        expected_rate = anomaly_mask.sum() / max(n, 1)
        chi2_p = None
        if len(unique_clusters) >= 2 and anomaly_mask.sum() > 0:
            observed = df_result["anomaly_count"].values
            expected = df_result["size"].values * expected_rate
            expected = np.where(expected < 1, 1, expected)
            try:
                chi2, p = sp_stats.chisquare(observed, f_exp=expected)
                chi2_p = float(p)
            except Exception:
                pass

        return {
            "table": df_result,
            "overall_anomaly_rate": float(expected_rate),
            "chi2_uniform_p": chi2_p,
            "is_uniform": chi2_p is not None and chi2_p > 0.05,
        }

    # ------------------------------------------------------------------
    #  X2. Missing × Correlation (MAR detection)
    # ------------------------------------------------------------------

    def missing_correlation(self) -> dict[str, Any] | None:
        """Correlate missing-indicators with numeric columns to diagnose MAR."""
        mi = self._stats.missing_info
        if mi.empty or "missing_ratio" not in mi.columns:
            return None

        # Columns with any missing
        miss_cols = mi[mi["missing_ratio"] > 0].index.tolist()
        if not miss_cols:
            return None

        num_cols = self._schema.numeric_columns[:self._max_cols]
        if not num_cols:
            return None

        # Build indicator matrix
        indicators = pd.DataFrame(index=self._df.index)
        for col in miss_cols:
            if col in self._df.columns:
                indicators[f"{col}_missing"] = self._df[col].isna().astype(int)

        if indicators.empty:
            return None

        # Correlate indicators with numeric columns
        num_data = self._df[num_cols].apply(pd.to_numeric, errors="coerce")

        corr_matrix = pd.DataFrame(
            np.nan, index=indicators.columns, columns=num_cols,
        )
        mar_suspects: list[dict[str, Any]] = []

        for ind_col in indicators.columns:
            ind_series = indicators[ind_col]
            if ind_series.sum() < 5 or ind_series.sum() == len(ind_series):
                continue  # too few or all missing
            for num_col in num_cols:
                valid = num_data[num_col].notna() & ind_series.notna()
                if valid.sum() < 10:
                    continue
                try:
                    r, p = sp_stats.pointbiserialr(
                        ind_series[valid].values,
                        num_data[num_col][valid].values,
                    )
                    corr_matrix.loc[ind_col, num_col] = r
                    if abs(r) > 0.2 and p < 0.05:
                        mar_suspects.append({
                            "missing_column": ind_col.replace("_missing", ""),
                            "correlated_with": num_col,
                            "correlation": round(float(r), 4),
                            "p_value": round(float(p), 6),
                        })
                except Exception:
                    continue

        # Diagnose MCAR vs MAR
        diagnosis = "MCAR_likely"
        if mar_suspects:
            max_abs_r = max(abs(s["correlation"]) for s in mar_suspects)
            if max_abs_r > 0.4:
                diagnosis = "MAR_strong"
            elif max_abs_r > 0.2:
                diagnosis = "MAR_moderate"

        # Imputation strategy recommendation
        strategies: dict[str, str] = {}
        for col in miss_cols:
            ratio = float(mi.loc[col, "missing_ratio"]) if col in mi.index else 0
            is_numeric = col in self._schema.numeric_columns
            has_mar = any(s["missing_column"] == col for s in mar_suspects)

            if ratio > 0.5:
                strategies[col] = "drop_column"
            elif has_mar:
                strategies[col] = "knn_or_mice" if is_numeric else "model_based"
            elif is_numeric:
                strategies[col] = "median"
            else:
                strategies[col] = "mode"

        return {
            "indicator_correlation": corr_matrix.dropna(how="all", axis=0).dropna(how="all", axis=1),
            "mar_suspects": pd.DataFrame(mar_suspects) if mar_suspects else pd.DataFrame(),
            "diagnosis": diagnosis,
            "imputation_strategy": strategies,
        }

    # ------------------------------------------------------------------
    #  X3. Distribution × Outlier Method Fitness
    # ------------------------------------------------------------------

    def distribution_outlier_fitness(self) -> pd.DataFrame | None:
        """Recommend the best outlier detection method per column based on distribution shape."""
        dist = self._stats.distribution_info
        summary = self._stats.summary
        if dist.empty or summary.empty:
            return None

        rows = []
        for col in dist.index:
            if col not in summary.index:
                continue

            skew = dist.loc[col].get("skewness", 0) or 0
            kurt = dist.loc[col].get("kurtosis", 0) or 0
            is_normal = dist.loc[col].get("is_normal_0.05", False)

            abs_skew = abs(skew)
            reasons = []

            if is_normal and abs_skew < 1 and abs(kurt) < 3:
                method = "zscore"
                reasons.append("approximately normal distribution")
            elif abs_skew > 2 or kurt > 7:
                method = "isolation_forest"
                reasons.append("heavy-tailed or highly skewed distribution")
                if abs_skew > 2:
                    reasons.append(f"skewness={skew:.2f}")
                if kurt > 7:
                    reasons.append(f"kurtosis={kurt:.1f}")
            elif abs_skew > 1 or kurt > 3:
                method = "iqr"
                reasons.append("moderately skewed/heavy-tailed")
            else:
                method = "iqr"
                reasons.append("moderate distribution shape")

            rows.append({
                "column": col,
                "skewness": round(float(skew), 3),
                "kurtosis": round(float(kurt), 3),
                "is_normal": bool(is_normal),
                "recommended_method": method,
                "reason": "; ".join(reasons),
            })

        if not rows:
            return None
        return pd.DataFrame(rows).set_index("column")

    # ------------------------------------------------------------------
    #  X4. Cluster × Correlation — Simpson's Paradox Detection
    # ------------------------------------------------------------------

    def simpson_paradox(self) -> dict[str, Any] | None:
        """Detect Simpson's paradox: overall correlation direction reverses within clusters."""
        adv = self._stats.advanced_stats
        clustering = adv.get("clustering", {})
        km = clustering.get("kmeans")
        if not km:
            return None

        cluster_labels = km.get("labels")
        if cluster_labels is None:
            return None

        pearson = self._stats.correlation_matrix
        if pearson.empty:
            return None

        num_cols = [c for c in pearson.columns if c in self._df.columns][:self._max_cols]
        if len(num_cols) < 2:
            return None

        n = min(len(cluster_labels), len(self._df))
        labels = np.asarray(cluster_labels[:n])
        df_sub = self._df.iloc[:n]

        unique_clusters = np.unique(labels)
        if len(unique_clusters) < 2:
            return None

        paradoxes: list[dict[str, Any]] = []

        for i, c1 in enumerate(num_cols):
            for c2 in num_cols[i + 1:]:
                overall_r = pearson.loc[c1, c2] if c1 in pearson.index and c2 in pearson.columns else 0
                if abs(overall_r) < 0.1:
                    continue  # skip negligible correlations

                cluster_corrs = {}
                n_reversed = 0
                for cl in unique_clusters:
                    mask = labels == cl
                    if mask.sum() < 10:
                        continue
                    try:
                        x = pd.to_numeric(df_sub.loc[mask.nonzero()[0], c1], errors="coerce").dropna()
                        y = pd.to_numeric(df_sub.loc[mask.nonzero()[0], c2], errors="coerce").dropna()
                        common_idx = x.index.intersection(y.index)
                        if len(common_idx) < 10:
                            continue
                        r, _ = sp_stats.pearsonr(x.loc[common_idx], y.loc[common_idx])
                        cluster_corrs[f"cluster_{cl}"] = round(float(r), 4)
                        if np.sign(r) != np.sign(overall_r) and abs(r) > 0.1:
                            n_reversed += 1
                    except Exception:
                        continue

                if n_reversed > 0 and len(cluster_corrs) >= 2:
                    paradoxes.append({
                        "col_a": c1,
                        "col_b": c2,
                        "overall_corr": round(float(overall_r), 4),
                        "cluster_corrs": cluster_corrs,
                        "n_reversed_clusters": n_reversed,
                        "is_paradox": True,
                        "paradox_strength": round(
                            n_reversed / max(len(cluster_corrs), 1), 3
                        ),
                    })

        if not paradoxes:
            return None

        paradoxes.sort(key=lambda x: x["paradox_strength"], reverse=True)
        return {
            "paradoxes": pd.DataFrame(paradoxes),
            "n_paradoxes": len(paradoxes),
        }

    # ------------------------------------------------------------------
    #  X5. Feature Importance × Missing Rate
    # ------------------------------------------------------------------

    def importance_vs_missing(self) -> pd.DataFrame | None:
        """Cross-tabulate feature importance with missing rate."""
        fi = self._stats.feature_importance
        mi = self._stats.missing_info
        if fi.empty or mi.empty:
            return None

        if "missing_ratio" not in mi.columns:
            return None

        # Detect the importance column name
        imp_col = None
        for candidate in ["variance", "cv", "mean_abs_corr", "mutual_info"]:
            if candidate in fi.columns:
                imp_col = candidate
                break
        if imp_col is None and len(fi.columns) > 0:
            imp_col = fi.columns[0]
        if imp_col is None:
            return None

        common_cols = list(set(fi.index) & set(mi.index))
        if not common_cols:
            return None

        rows = []
        for col in common_cols:
            importance = float(fi.loc[col, imp_col]) if col in fi.index else 0
            missing_ratio = float(mi.loc[col, "missing_ratio"]) if col in mi.index else 0
            risk = "none"
            if missing_ratio > 0.3 and importance > fi[imp_col].median():
                risk = "high"
            elif missing_ratio > 0.1 and importance > fi[imp_col].median():
                risk = "medium"
            elif missing_ratio > 0.05:
                risk = "low"
            rows.append({
                "column": col,
                "importance": round(importance, 4),
                "missing_ratio": round(missing_ratio, 4),
                "information_loss_risk": risk,
            })

        df_result = pd.DataFrame(rows).set_index("column")
        df_result = df_result.sort_values("importance", ascending=False)
        return df_result

    # ------------------------------------------------------------------
    #  X6. Unified 2D Embedding (t-SNE/UMAP + Cluster + Anomaly overlay)
    # ------------------------------------------------------------------

    def unified_2d_embedding(self) -> dict[str, Any] | None:
        """Prepare a unified 2D scatter dataset with cluster + anomaly labels."""
        adv = self._stats.advanced_stats
        dr = adv.get("dimreduction", {})
        clustering = adv.get("clustering", {})
        anomaly_full = adv.get("advanced_anomaly_full", {})

        # Get 2D coordinates (prefer t-SNE then UMAP)
        coords = None
        method = None
        for key in ["tsne", "umap"]:
            emb = dr.get(key)
            if emb is not None and isinstance(emb, dict):
                c = emb.get("coordinates")
                if c is not None and hasattr(c, "shape") and len(c) > 0:
                    coords = np.asarray(c)
                    method = key
                    break

        if coords is None or coords.shape[1] < 2:
            return None

        n = coords.shape[0]
        result: dict[str, Any] = {
            "x": coords[:, 0].tolist(),
            "y": coords[:, 1].tolist(),
            "method": method,
            "n_points": n,
        }

        # Add cluster labels
        km = clustering.get("kmeans")
        if km and km.get("labels") is not None:
            cl = np.asarray(km["labels"])
            result["cluster_labels"] = cl[:n].tolist() if len(cl) >= n else cl.tolist()

        # Add anomaly labels
        iso = anomaly_full.get("isolation_forest")
        if iso and iso.get("labels") is not None:
            al = np.asarray(iso["labels"])
            result["anomaly_labels"] = al[:n].tolist() if len(al) >= n else al.tolist()

        return result
