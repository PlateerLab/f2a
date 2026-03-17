use ndarray::Array2;
use polars::prelude::*;
use serde::{Deserialize, Serialize};

use crate::core::schema::DataSchema;
use crate::stats::descriptive::DescriptiveStats;

// ─── Result types ───────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ElbowPoint {
    pub k: usize,
    pub inertia: f64,
    pub silhouette: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KMeansResult {
    pub elbow_data: Vec<ElbowPoint>,
    pub optimal_k: usize,
    pub silhouette_score: f64,
    pub cluster_sizes: Vec<usize>,
    pub labels: Vec<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DbscanResult {
    pub eps: f64,
    pub min_samples: usize,
    pub n_clusters: usize,
    pub n_noise: usize,
    pub cluster_sizes: Vec<usize>,
    pub labels: Vec<i32>, // -1 = noise
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusteringResult {
    pub kmeans: Option<KMeansResult>,
    pub dbscan: Option<DbscanResult>,
}

// ─── Implementation ─────────────────────────────────────────────────

pub struct ClusteringStats<'a> {
    df: &'a DataFrame,
    schema: &'a DataSchema,
    max_k: usize,
    max_sample: usize,
}

impl<'a> ClusteringStats<'a> {
    pub fn new(df: &'a DataFrame, schema: &'a DataSchema, max_k: usize, max_sample: usize) -> Self {
        Self {
            df,
            schema,
            max_k,
            max_sample,
        }
    }

    pub fn compute(&self) -> ClusteringResult {
        let matrix = match self.prepare_data() {
            Some(m) => m,
            None => {
                return ClusteringResult {
                    kmeans: None,
                    dbscan: None,
                }
            }
        };

        let kmeans = self.kmeans_analysis(&matrix);
        let dbscan = self.dbscan_analysis(&matrix);

        ClusteringResult { kmeans, dbscan }
    }

    // ── K-Means ─────────────────────────────────────────────────

    fn kmeans_analysis(&self, data: &Array2<f64>) -> Option<KMeansResult> {
        let n = data.nrows();
        if n < 4 {
            return None;
        }

        let max_k = self.max_k.min(n / 2).max(2);
        let mut elbow_data = Vec::new();
        let mut best_silhouette = -1.0f64;
        let mut best_k = 2;
        let mut best_labels = vec![0usize; n];

        for k in 2..=max_k {
            let (labels, inertia) = Self::kmeans_fit(data, k, 100);
            let silhouette = Self::silhouette_score(data, &labels, k);

            elbow_data.push(ElbowPoint {
                k,
                inertia,
                silhouette,
            });

            if silhouette > best_silhouette {
                best_silhouette = silhouette;
                best_k = k;
                best_labels = labels;
            }
        }

        let mut cluster_sizes = vec![0usize; best_k];
        for &l in &best_labels {
            if l < best_k {
                cluster_sizes[l] += 1;
            }
        }

        Some(KMeansResult {
            elbow_data,
            optimal_k: best_k,
            silhouette_score: best_silhouette,
            cluster_sizes,
            labels: best_labels,
        })
    }

    /// Simple K-Means implementation (Lloyd's algorithm).
    fn kmeans_fit(data: &Array2<f64>, k: usize, max_iter: usize) -> (Vec<usize>, f64) {
        let n = data.nrows();
        let d = data.ncols();

        // Initialize centroids using K-Means++ style
        let mut centroids = Array2::<f64>::zeros((k, d));

        // First centroid: random (just pick first row for determinism)
        centroids.row_mut(0).assign(&data.row(0));

        for c in 1..k {
            // Pick the point furthest from existing centroids
            let mut max_dist = 0.0f64;
            let mut max_idx = 0;
            for i in 0..n {
                let min_dist: f64 = (0..c)
                    .map(|j| Self::euclidean_sq(&data.row(i), &centroids.row(j)))
                    .fold(f64::INFINITY, f64::min);
                if min_dist > max_dist {
                    max_dist = min_dist;
                    max_idx = i;
                }
            }
            centroids.row_mut(c).assign(&data.row(max_idx));
        }

        let mut labels = vec![0usize; n];

        for _iter in 0..max_iter {
            // Assign step
            let mut changed = false;
            for i in 0..n {
                let mut best_c = 0;
                let mut best_d = f64::INFINITY;
                for c in 0..k {
                    let dist = Self::euclidean_sq(&data.row(i), &centroids.row(c));
                    if dist < best_d {
                        best_d = dist;
                        best_c = c;
                    }
                }
                if labels[i] != best_c {
                    labels[i] = best_c;
                    changed = true;
                }
            }

            if !changed {
                break;
            }

            // Update step
            let mut sums = Array2::<f64>::zeros((k, d));
            let mut counts = vec![0usize; k];
            for i in 0..n {
                let c = labels[i];
                for j in 0..d {
                    sums[[c, j]] += data[[i, j]];
                }
                counts[c] += 1;
            }
            for c in 0..k {
                if counts[c] > 0 {
                    for j in 0..d {
                        centroids[[c, j]] = sums[[c, j]] / counts[c] as f64;
                    }
                }
            }
        }

        // Inertia (within-cluster sum of squares)
        let inertia: f64 = (0..n)
            .map(|i| Self::euclidean_sq(&data.row(i), &centroids.row(labels[i])))
            .sum();

        (labels, inertia)
    }

    /// Simplified silhouette score.
    fn silhouette_score(data: &Array2<f64>, labels: &[usize], k: usize) -> f64 {
        let n = data.nrows();
        if n < 2 || k < 2 {
            return 0.0;
        }

        // Sample for large datasets
        let sample_size = n.min(1000);
        let step = n / sample_size;

        let mut total = 0.0f64;
        let mut count = 0;

        for i in (0..n).step_by(step.max(1)) {
            let mut intra_sum = 0.0;
            let mut intra_count = 0;

            // Mean distance to same-cluster points (a)
            for j in 0..n {
                if i != j && labels[i] == labels[j] {
                    intra_sum += Self::euclidean(&data.row(i), &data.row(j));
                    intra_count += 1;
                }
            }
            let a = if intra_count > 0 {
                intra_sum / intra_count as f64
            } else {
                0.0
            };

            // Mean distance to nearest other cluster (b)
            let mut b = f64::INFINITY;
            for c in 0..k {
                if c == labels[i] {
                    continue;
                }
                let mut inter_sum = 0.0;
                let mut inter_count = 0;
                for j in 0..n {
                    if labels[j] == c {
                        inter_sum += Self::euclidean(&data.row(i), &data.row(j));
                        inter_count += 1;
                    }
                }
                if inter_count > 0 {
                    b = b.min(inter_sum / inter_count as f64);
                }
            }

            let s = if a.max(b) > f64::EPSILON {
                (b - a) / a.max(b)
            } else {
                0.0
            };
            total += s;
            count += 1;
        }

        if count > 0 {
            total / count as f64
        } else {
            0.0
        }
    }

    // ── DBSCAN ──────────────────────────────────────────────────

    fn dbscan_analysis(&self, data: &Array2<f64>) -> Option<DbscanResult> {
        let n = data.nrows();
        if n < 5 {
            return None;
        }

        let min_samples = (n as f64).ln().ceil() as usize;
        let eps = self.estimate_eps(data, min_samples);

        // DBSCAN implementation
        let mut labels = vec![-2i32; n]; // -2 = unvisited
        let mut cluster_id = 0i32;

        for i in 0..n {
            if labels[i] != -2 {
                continue;
            }

            let neighbors = self.range_query(data, i, eps);
            if neighbors.len() < min_samples {
                labels[i] = -1; // noise
                continue;
            }

            labels[i] = cluster_id;
            let mut seed_set: Vec<usize> = neighbors;
            let mut idx = 0;

            while idx < seed_set.len() {
                let q = seed_set[idx];
                if labels[q] == -1 {
                    labels[q] = cluster_id;
                }
                if labels[q] != -2 {
                    idx += 1;
                    continue;
                }

                labels[q] = cluster_id;
                let q_neighbors = self.range_query(data, q, eps);
                if q_neighbors.len() >= min_samples {
                    for nn in q_neighbors {
                        if !seed_set.contains(&nn) {
                            seed_set.push(nn);
                        }
                    }
                }
                idx += 1;
            }

            cluster_id += 1;
        }

        let n_clusters = cluster_id as usize;
        let n_noise = labels.iter().filter(|&&l| l == -1).count();
        let mut cluster_sizes = vec![0usize; n_clusters];
        for &l in &labels {
            if l >= 0 {
                cluster_sizes[l as usize] += 1;
            }
        }

        Some(DbscanResult {
            eps,
            min_samples,
            n_clusters,
            n_noise,
            cluster_sizes,
            labels,
        })
    }

    /// Estimate eps using k-nearest neighbor distance heuristic.
    fn estimate_eps(&self, data: &Array2<f64>, k: usize) -> f64 {
        let n = data.nrows();
        let sample_size = n.min(500);
        let step = (n / sample_size).max(1);

        let mut k_distances: Vec<f64> = Vec::new();

        for i in (0..n).step_by(step) {
            let mut dists: Vec<f64> = (0..n)
                .filter(|&j| j != i)
                .map(|j| Self::euclidean(&data.row(i), &data.row(j)))
                .collect();
            dists.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            if dists.len() >= k {
                k_distances.push(dists[k - 1]);
            }
        }

        k_distances.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        // Use the "knee" of the sorted k-distance plot
        // Approximate: use the value at 90th percentile
        let idx = (k_distances.len() as f64 * 0.9) as usize;
        k_distances
            .get(idx.min(k_distances.len().saturating_sub(1)))
            .copied()
            .unwrap_or(1.0)
    }

    fn range_query(&self, data: &Array2<f64>, idx: usize, eps: f64) -> Vec<usize> {
        let n = data.nrows();
        (0..n)
            .filter(|&j| Self::euclidean(&data.row(idx), &data.row(j)) <= eps)
            .collect()
    }

    // ── Data preparation ────────────────────────────────────────

    fn prepare_data(&self) -> Option<Array2<f64>> {
        let num_cols = self.schema.numeric_columns();
        if num_cols.len() < 2 {
            return None;
        }

        let mut col_data: Vec<Vec<f64>> = Vec::new();
        for &col_name in &num_cols {
            if let Some(vals) = self
                .df
                .column(col_name)
                .ok()
                .and_then(|c| DescriptiveStats::column_to_f64_vec(c))
            {
                col_data.push(vals);
            }
        }

        if col_data.is_empty() {
            return None;
        }

        let min_len = col_data.iter().map(|v| v.len()).min().unwrap_or(0);
        let sample_len = min_len.min(self.max_sample);
        let step = (min_len / sample_len).max(1);
        let n_cols = col_data.len();

        let mut matrix = Array2::<f64>::zeros((sample_len, n_cols));
        for (j, data) in col_data.iter().enumerate() {
            for (i_out, i_in) in (0..min_len).step_by(step).take(sample_len).enumerate() {
                matrix[[i_out, j]] = data[i_in];
            }
        }

        // Standardize
        for j in 0..n_cols {
            let col = matrix.column(j);
            let mean = col.mean().unwrap_or(0.0);
            let std = {
                let var: f64 = col.iter().map(|x| (x - mean).powi(2)).sum::<f64>()
                    / (sample_len as f64 - 1.0).max(1.0);
                var.sqrt()
            };
            if std > f64::EPSILON {
                for i in 0..sample_len {
                    matrix[[i, j]] = (matrix[[i, j]] - mean) / std;
                }
            }
        }

        Some(matrix)
    }

    // ── Distance helpers ────────────────────────────────────────

    fn euclidean_sq(a: &ndarray::ArrayView1<f64>, b: &ndarray::ArrayView1<f64>) -> f64 {
        a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum()
    }

    fn euclidean(a: &ndarray::ArrayView1<f64>, b: &ndarray::ArrayView1<f64>) -> f64 {
        Self::euclidean_sq(a, b).sqrt()
    }
}
