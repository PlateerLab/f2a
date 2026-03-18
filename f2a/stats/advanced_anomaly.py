"""Advanced anomaly detection module.

Provides Isolation Forest, Local Outlier Factor, Mahalanobis distance,
and consensus anomaly scoring.

References:
    - Liu et al. (2008) — Isolation Forest
    - Breunig et al. (2000) — Local Outlier Factor
    - Mahalanobis (1936) — Mahalanobis distance
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from f2a.core.schema import DataSchema
from f2a.utils.logging import get_logger

logger = get_logger(__name__)


class AdvancedAnomalyStats:
    """Multi-method anomaly detection for numeric columns.

    Args:
        df: Target DataFrame.
        schema: Data schema.
        max_sample: Max rows to sample for expensive operations.
        contamination: Expected proportion of anomalies.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        schema: DataSchema,
        max_sample: int = 5000,
        contamination: float = 0.05,
    ) -> None:
        self._df = df
        self._schema = schema
        self._max_sample = max_sample
        self._contamination = contamination

    def _prepare_data(self) -> tuple[np.ndarray, pd.DataFrame, list[str]] | None:
        """Prepare and scale numeric data."""
        cols = self._schema.numeric_columns
        if len(cols) < 2:
            return None

        try:
            from sklearn.preprocessing import StandardScaler
        except ImportError:
            return None

        df_clean = self._df[cols].dropna()
        if len(df_clean) < 20:
            return None

        if len(df_clean) > self._max_sample:
            df_clean = df_clean.sample(self._max_sample, random_state=42)

        scaler = StandardScaler()
        X = scaler.fit_transform(df_clean)
        return X, df_clean, cols

    # ── Isolation Forest ──────────────────────────────────

    def isolation_forest(self) -> dict[str, Any]:
        """Detect anomalies using Isolation Forest.

        Isolation Forest isolates observations by randomly selecting a
        feature and then randomly selecting a split value.  Anomalies
        require fewer splits (shorter path length).

        Returns:
            Dictionary with anomaly_count, anomaly_ratio, scores_summary.
        """
        prepared = self._prepare_data()
        if prepared is None:
            return {}

        X, df_clean, cols = prepared

        try:
            from sklearn.ensemble import IsolationForest
        except ImportError:
            return {}

        try:
            iso = IsolationForest(
                contamination=self._contamination,
                random_state=42,
                max_samples=min(256, len(X)),
                n_estimators=100,
            )
            labels = iso.fit_predict(X)  # -1 = anomaly, 1 = normal
            scores = iso.decision_function(X)

            n_anomaly = int((labels == -1).sum())

            return {
                "method": "Isolation Forest",
                "anomaly_count": n_anomaly,
                "anomaly_ratio": round(n_anomaly / len(X), 4),
                "n_samples": len(X),
                "score_mean": round(float(scores.mean()), 4),
                "score_std": round(float(scores.std()), 4),
                "score_min": round(float(scores.min()), 4),
                "score_threshold": round(float(np.percentile(scores, self._contamination * 100)), 4),
                "labels": labels,
                "scores": scores,
            }
        except Exception as exc:
            logger.debug("Isolation Forest failed: %s", exc)
            return {}

    # ── Local Outlier Factor ──────────────────────────────

    def local_outlier_factor(self) -> dict[str, Any]:
        """Detect anomalies using Local Outlier Factor (LOF).

        LOF measures the local deviation of density for each sample
        compared to its neighbors.

        Returns:
            Dictionary with anomaly_count, anomaly_ratio, scores summary.
        """
        prepared = self._prepare_data()
        if prepared is None:
            return {}

        X, df_clean, cols = prepared

        try:
            from sklearn.neighbors import LocalOutlierFactor
        except ImportError:
            return {}

        try:
            n_neighbors = min(20, len(X) - 1)
            lof = LocalOutlierFactor(
                n_neighbors=n_neighbors,
                contamination=self._contamination,
            )
            labels = lof.fit_predict(X)  # -1 = anomaly
            scores = lof.negative_outlier_factor_

            n_anomaly = int((labels == -1).sum())

            return {
                "method": "Local Outlier Factor",
                "anomaly_count": n_anomaly,
                "anomaly_ratio": round(n_anomaly / len(X), 4),
                "n_samples": len(X),
                "n_neighbors": n_neighbors,
                "lof_mean": round(float(scores.mean()), 4),
                "lof_std": round(float(scores.std()), 4),
                "lof_min": round(float(scores.min()), 4),
                "labels": labels,
                "scores": scores,
            }
        except Exception as exc:
            logger.debug("LOF failed: %s", exc)
            return {}

    # ── Mahalanobis distance ──────────────────────────────

    def mahalanobis_distance(self) -> dict[str, Any]:
        """Detect anomalies using Mahalanobis distance.

        Points with high Mahalanobis distance from the centroid
        are potential multivariate outliers.

        Returns:
            Dictionary with threshold, anomaly count, distances summary.
        """
        cols = self._schema.numeric_columns
        if len(cols) < 2:
            return {}

        df_clean = self._df[cols].dropna()
        if len(df_clean) < len(cols) + 5:
            return {}

        if len(df_clean) > self._max_sample:
            df_clean = df_clean.sample(self._max_sample, random_state=42)

        cols = cols[:30]  # limit columns to avoid ill-conditioned matrices
        df_clean = df_clean[cols]
        data = df_clean.values
        try:
            mean = np.mean(data, axis=0)
            cov = np.cov(data.T)
            # Regularise to handle singular/near-singular covariance
            cov += np.eye(cov.shape[0]) * 1e-6
            if np.linalg.cond(cov) > 1e10:
                logger.debug("Covariance matrix ill-conditioned; skipping Mahalanobis.")
                return {}
            cov_inv = np.linalg.inv(cov)

            diff = data - mean
            left = diff @ cov_inv
            maha_sq = np.sum(left * diff, axis=1)
            maha = np.sqrt(np.maximum(maha_sq, 0))

            # Chi-squared threshold at 97.5% with p degrees of freedom
            from scipy.stats import chi2

            p = len(cols)
            threshold = float(np.sqrt(chi2.ppf(0.975, p)))

            anomaly_mask = maha > threshold
            n_anomaly = int(anomaly_mask.sum())

            return {
                "method": "Mahalanobis Distance",
                "anomaly_count": n_anomaly,
                "anomaly_ratio": round(n_anomaly / len(data), 4),
                "threshold": round(threshold, 4),
                "n_features": p,
                "n_samples": len(data),
                "distance_mean": round(float(maha.mean()), 4),
                "distance_std": round(float(maha.std()), 4),
                "distance_max": round(float(maha.max()), 4),
                "distances": maha,
                "labels": np.where(anomaly_mask, -1, 1),
            }
        except (np.linalg.LinAlgError, Exception) as exc:
            logger.debug("Mahalanobis distance failed: %s", exc)
            return {}

    # ── Consensus anomaly ─────────────────────────────────

    def consensus_anomaly(self) -> dict[str, Any]:
        """Consensus anomaly detection combining multiple methods.

        An observation is flagged as anomalous if flagged by at least
        2 out of 3 methods (IF, LOF, Mahalanobis).

        Returns:
            Dictionary with per-method counts, consensus count, and
            agreement statistics.
        """
        iso_result = self.isolation_forest()
        lof_result = self.local_outlier_factor()
        maha_result = self.mahalanobis_distance()

        methods = []
        if "labels" in iso_result:
            methods.append(("isolation_forest", iso_result["labels"]))
        if "labels" in lof_result:
            methods.append(("local_outlier_factor", lof_result["labels"]))
        if "labels" in maha_result:
            methods.append(("mahalanobis", maha_result["labels"]))

        if len(methods) < 2:
            return {}

        # Align lengths (should be same, but just in case)
        min_len = min(len(labels) for _, labels in methods)
        vote_matrix = np.zeros((min_len, len(methods)))

        for i, (_, labels) in enumerate(methods):
            vote_matrix[:, i] = (labels[:min_len] == -1).astype(int)

        votes = vote_matrix.sum(axis=1)
        # Consensus: flagged by >= 2 methods
        consensus_mask = votes >= 2

        per_method = {}
        for name, labels in methods:
            per_method[name] = int((labels[:min_len] == -1).sum())

        return {
            "methods_used": [name for name, _ in methods],
            "per_method_counts": per_method,
            "consensus_count": int(consensus_mask.sum()),
            "consensus_ratio": round(float(consensus_mask.sum()) / min_len, 4),
            "n_samples": min_len,
            "consensus_threshold": 2,
            "agreement_matrix": {
                "all_agree_anomaly": int((votes == len(methods)).sum()),
                "majority_anomaly": int(consensus_mask.sum()),
                "any_anomaly": int((votes >= 1).sum()),
                "no_anomaly": int((votes == 0).sum()),
            },
        }

    # ── Summary ───────────────────────────────────────────

    def summary_full(self) -> tuple[dict[str, Any], dict[str, Any]]:
        """Return combined advanced anomaly results (stripped + full).

        Returns a tuple of (stripped_summary, full_results) so that
        each method is only called once.
        """
        result: dict[str, Any] = {}
        full: dict[str, Any] = {}

        try:
            iso = self.isolation_forest()
            if iso:
                result["isolation_forest"] = {
                    k: v for k, v in iso.items() if k not in ("labels", "scores")
                }
                full["isolation_forest"] = iso
        except Exception as exc:
            logger.debug("Isolation Forest skipped: %s", exc)

        try:
            lof = self.local_outlier_factor()
            if lof:
                result["local_outlier_factor"] = {
                    k: v for k, v in lof.items() if k not in ("labels", "scores")
                }
                full["local_outlier_factor"] = lof
        except Exception as exc:
            logger.debug("LOF skipped: %s", exc)

        try:
            maha = self.mahalanobis_distance()
            if maha:
                result["mahalanobis"] = {
                    k: v for k, v in maha.items() if k not in ("distances", "labels")
                }
                full["mahalanobis"] = maha
        except Exception as exc:
            logger.debug("Mahalanobis skipped: %s", exc)

        # Build consensus from already-computed results instead of re-running
        try:
            methods = []
            if "isolation_forest" in full and "labels" in full["isolation_forest"]:
                methods.append(("isolation_forest", full["isolation_forest"]["labels"]))
            if "local_outlier_factor" in full and "labels" in full["local_outlier_factor"]:
                methods.append(("local_outlier_factor", full["local_outlier_factor"]["labels"]))
            if "mahalanobis" in full and "labels" in full["mahalanobis"]:
                methods.append(("mahalanobis", full["mahalanobis"]["labels"]))

            if len(methods) >= 2:
                min_len = min(len(labels) for _, labels in methods)
                vote_matrix = np.zeros((min_len, len(methods)))
                for i, (_, labels) in enumerate(methods):
                    vote_matrix[:, i] = (labels[:min_len] == -1).astype(int)
                votes = vote_matrix.sum(axis=1)
                consensus_mask = votes >= 2

                per_method = {}
                for name, labels in methods:
                    per_method[name] = int((labels[:min_len] == -1).sum())

                cons = {
                    "methods_used": [name for name, _ in methods],
                    "per_method_counts": per_method,
                    "consensus_count": int(consensus_mask.sum()),
                    "consensus_ratio": round(float(consensus_mask.sum()) / min_len, 4),
                    "n_samples": min_len,
                    "consensus_threshold": 2,
                    "agreement_matrix": {
                        "all_agree_anomaly": int((votes == len(methods)).sum()),
                        "majority_anomaly": int(consensus_mask.sum()),
                        "any_anomaly": int((votes >= 1).sum()),
                        "no_anomaly": int((votes == 0).sum()),
                    },
                }
                result["consensus"] = cons
                full["consensus"] = cons
        except Exception as exc:
            logger.debug("Consensus anomaly skipped: %s", exc)

        return result, full

    def summary(self) -> dict[str, Any]:
        """Return combined advanced anomaly detection results (stripped)."""
        result, _ = self.summary_full()
        return result
