#!/usr/bin/env python3
"""Compute structural metrics from PHATE embeddings for ML classification."""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy.spatial.distance import pdist, cdist
from scipy.stats import entropy
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

# Paths
PHATE_RESULTS_DIR = Path("/nfs/roberts/project/pi_sk2433/shared/Geomancer_2025_Data/phate_results")
CLASSIFICATION_CSV = Path("/home/btd8/llm-paper-analyze/data/manylatents_benchmark/datasets_for_classification.csv")
OUTPUT_PATH = Path("/home/btd8/llm-paper-analyze/data/manylatents_benchmark/embedding_metrics.csv")


def compute_metrics(embedding: np.ndarray, dataset_id: str) -> dict:
    """Compute structural metrics from 2D PHATE embedding."""

    n_points = len(embedding)
    x, y = embedding[:, 0], embedding[:, 1]

    metrics = {
        'dataset_id': dataset_id,
        'n_points': n_points,
    }

    # === BASIC STATISTICS ===
    metrics['x_mean'] = np.mean(x)
    metrics['y_mean'] = np.mean(y)
    metrics['x_std'] = np.std(x)
    metrics['y_std'] = np.std(y)
    metrics['x_range'] = np.ptp(x)
    metrics['y_range'] = np.ptp(y)

    # Aspect ratio (shape indicator)
    metrics['aspect_ratio'] = metrics['x_range'] / (metrics['y_range'] + 1e-10)

    # Total spread
    metrics['total_variance'] = np.var(x) + np.var(y)

    # === PAIRWISE DISTANCES ===
    # Sample if too large (> 5000 points)
    if n_points > 5000:
        sample_idx = np.random.choice(n_points, 5000, replace=False)
        emb_sample = embedding[sample_idx]
    else:
        emb_sample = embedding

    pairwise_dists = pdist(emb_sample)
    metrics['pairwise_mean'] = np.mean(pairwise_dists)
    metrics['pairwise_std'] = np.std(pairwise_dists)
    metrics['pairwise_median'] = np.median(pairwise_dists)
    metrics['pairwise_q25'] = np.percentile(pairwise_dists, 25)
    metrics['pairwise_q75'] = np.percentile(pairwise_dists, 75)
    metrics['pairwise_iqr'] = metrics['pairwise_q75'] - metrics['pairwise_q25']

    # === KNN DISTANCES (for density estimation) ===
    for k in [5, 10, 20, 50]:
        if k < n_points:
            nbrs = NearestNeighbors(n_neighbors=k+1).fit(embedding)
            distances, _ = nbrs.kneighbors(embedding)
            knn_dists = distances[:, -1]  # Distance to k-th neighbor
            metrics[f'knn_{k}_mean'] = np.mean(knn_dists)
            metrics[f'knn_{k}_std'] = np.std(knn_dists)
            metrics[f'knn_{k}_max'] = np.max(knn_dists)
        else:
            metrics[f'knn_{k}_mean'] = np.nan
            metrics[f'knn_{k}_std'] = np.nan
            metrics[f'knn_{k}_max'] = np.nan

    # === LOCAL DENSITY VARIANCE ===
    # Using k=10 neighbors as local density proxy
    if n_points > 10:
        nbrs = NearestNeighbors(n_neighbors=11).fit(embedding)
        distances, _ = nbrs.kneighbors(embedding)
        local_density = 1.0 / (np.mean(distances[:, 1:], axis=1) + 1e-10)
        metrics['density_mean'] = np.mean(local_density)
        metrics['density_std'] = np.std(local_density)
        metrics['density_cv'] = metrics['density_std'] / (metrics['density_mean'] + 1e-10)  # Coefficient of variation
        metrics['density_skew'] = pd.Series(local_density).skew()

    # === CLUSTERING METRICS (DBSCAN) ===
    # Try multiple eps values
    for eps_percentile in [5, 10, 25]:
        eps = np.percentile(pairwise_dists, eps_percentile)
        try:
            db = DBSCAN(eps=eps, min_samples=5).fit(emb_sample)
            labels = db.labels_
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            noise_ratio = np.sum(labels == -1) / len(labels)
            metrics[f'dbscan_eps{eps_percentile}_n_clusters'] = n_clusters
            metrics[f'dbscan_eps{eps_percentile}_noise_ratio'] = noise_ratio

            # Silhouette score (if >1 cluster and not all noise)
            if n_clusters > 1 and noise_ratio < 0.9:
                mask = labels != -1
                if np.sum(mask) > 10:
                    sil = silhouette_score(emb_sample[mask], labels[mask])
                    metrics[f'dbscan_eps{eps_percentile}_silhouette'] = sil
                else:
                    metrics[f'dbscan_eps{eps_percentile}_silhouette'] = np.nan
            else:
                metrics[f'dbscan_eps{eps_percentile}_silhouette'] = np.nan
        except Exception:
            metrics[f'dbscan_eps{eps_percentile}_n_clusters'] = np.nan
            metrics[f'dbscan_eps{eps_percentile}_noise_ratio'] = np.nan
            metrics[f'dbscan_eps{eps_percentile}_silhouette'] = np.nan

    # === CONNECTIVITY / GRAPH METRICS ===
    # How connected is the point cloud at various thresholds?
    for percentile in [10, 25, 50]:
        threshold = np.percentile(pairwise_dists, percentile)
        nbrs = NearestNeighbors(radius=threshold).fit(emb_sample)
        n_neighbors = nbrs.radius_neighbors(emb_sample, return_distance=False)
        avg_neighbors = np.mean([len(nn) - 1 for nn in n_neighbors])  # -1 to exclude self
        metrics[f'connectivity_p{percentile}'] = avg_neighbors / len(emb_sample)

    # === SHAPE METRICS ===
    # PCA of embedding to measure elongation
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    pca.fit(embedding)
    metrics['pca_var_ratio'] = pca.explained_variance_ratio_[0]  # Higher = more elongated
    metrics['pca_elongation'] = pca.explained_variance_[0] / (pca.explained_variance_[1] + 1e-10)

    # === ENTROPY / UNIFORMITY ===
    # Grid-based entropy (how uniformly distributed are points?)
    try:
        n_bins = min(20, int(np.sqrt(n_points)))
        hist, _, _ = np.histogram2d(x, y, bins=n_bins)
        hist_flat = hist.flatten()
        hist_norm = hist_flat / hist_flat.sum()
        hist_norm = hist_norm[hist_norm > 0]  # Remove zeros
        metrics['spatial_entropy'] = entropy(hist_norm)
        metrics['spatial_entropy_normalized'] = metrics['spatial_entropy'] / np.log(n_bins * n_bins)
    except Exception:
        metrics['spatial_entropy'] = np.nan
        metrics['spatial_entropy_normalized'] = np.nan

    # === CONVEX HULL ===
    try:
        from scipy.spatial import ConvexHull
        hull = ConvexHull(embedding)
        metrics['hull_area'] = hull.volume  # In 2D, volume is area
        metrics['hull_perimeter'] = hull.area  # In 2D, area is perimeter
        metrics['hull_compactness'] = 4 * np.pi * metrics['hull_area'] / (metrics['hull_perimeter']**2 + 1e-10)
        metrics['point_density_in_hull'] = n_points / (metrics['hull_area'] + 1e-10)
    except Exception:
        metrics['hull_area'] = np.nan
        metrics['hull_perimeter'] = np.nan
        metrics['hull_compactness'] = np.nan
        metrics['point_density_in_hull'] = np.nan

    # === CENTER OF MASS DISTANCE ===
    center = np.array([np.mean(x), np.mean(y)])
    dist_to_center = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    metrics['center_dist_mean'] = np.mean(dist_to_center)
    metrics['center_dist_std'] = np.std(dist_to_center)
    metrics['center_dist_skew'] = pd.Series(dist_to_center).skew()

    return metrics


def main():
    import gc

    print("Loading classification dataset info...")
    df_datasets = pd.read_csv(CLASSIFICATION_CSV)
    print(f"Found {len(df_datasets)} datasets")

    # Check for existing output and resume
    if OUTPUT_PATH.exists():
        existing = pd.read_csv(OUTPUT_PATH)
        processed_ids = set(existing['dataset_id'].tolist())
        all_metrics = existing.to_dict('records')
        print(f"Resuming: {len(processed_ids)} already processed")
    else:
        processed_ids = set()
        all_metrics = []

    for idx, row in df_datasets.iterrows():
        dataset_id = row['dataset_id']

        # Skip if already processed
        if dataset_id in processed_ids:
            continue

        # Find the PHATE results directory
        # The phate_plot_image path contains the version ID
        phate_path = row.get('phate_plot_image', '')
        if pd.isna(phate_path) or not phate_path:
            print(f"[{idx+1}/{len(df_datasets)}] {dataset_id}: No PHATE path, skipping")
            continue

        # Extract version ID from path
        # Path format: /nfs/.../phate_results/{version_id}/embedding_plot_...
        path_parts = Path(phate_path).parts
        try:
            phate_idx = path_parts.index('phate_results')
            version_id = path_parts[phate_idx + 1]
        except (ValueError, IndexError):
            print(f"[{idx+1}/{len(df_datasets)}] {dataset_id}: Cannot parse version ID, skipping")
            continue

        # Find embedding CSV
        results_dir = PHATE_RESULTS_DIR / version_id
        if not results_dir.exists():
            print(f"[{idx+1}/{len(df_datasets)}] {dataset_id}: Results dir not found, skipping")
            continue

        # Get the most recent embedding CSV
        embedding_files = list(results_dir.glob("embeddings_phate_*.csv"))
        if not embedding_files:
            print(f"[{idx+1}/{len(df_datasets)}] {dataset_id}: No embedding CSV, skipping")
            continue

        embedding_file = sorted(embedding_files)[-1]  # Most recent

        try:
            # Load embedding
            emb_df = pd.read_csv(embedding_file)
            embedding = emb_df[['dim_1', 'dim_2']].values

            # Compute metrics
            metrics = compute_metrics(embedding, dataset_id)

            # Add dataset metadata
            metrics['dataset_name'] = row.get('name', '')
            metrics['size_mb'] = row.get('size_mb', np.nan)
            metrics['num_features'] = row.get('num_features', np.nan)

            all_metrics.append(metrics)

            print(f"[{idx+1}/{len(df_datasets)}] Processed {dataset_id} ({metrics['n_points']} pts)")

            # Periodic save every 10 datasets
            if len(all_metrics) % 10 == 0:
                df_metrics = pd.DataFrame(all_metrics)
                df_metrics.to_csv(OUTPUT_PATH, index=False)
                print(f"  -> Saved checkpoint ({len(all_metrics)} datasets)")

            # Memory cleanup
            del emb_df, embedding, metrics
            gc.collect()

        except Exception as e:
            print(f"[{idx+1}/{len(df_datasets)}] {dataset_id}: Error - {e}")
            gc.collect()
            continue

    # Final save
    df_metrics = pd.DataFrame(all_metrics)
    df_metrics.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved {len(df_metrics)} datasets with metrics to {OUTPUT_PATH}")

    # Summary
    print(f"\nMetrics computed: {len(df_metrics.columns)} features")
    print(f"Datasets processed: {len(df_metrics)}")


if __name__ == "__main__":
    main()
