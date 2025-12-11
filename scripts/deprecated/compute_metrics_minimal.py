#!/usr/bin/env python3
"""
Memory-efficient script to compute metrics for PHATE embeddings one at a time.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import warnings
warnings.filterwarnings('ignore')

def compute_basic_metrics(embedding_data, dataset_id):
    """Compute essential structural metrics for a 2D embedding."""
    coords = embedding_data[['dim_1', 'dim_2']].values
    n_points = len(coords)

    if n_points < 10:
        return None

    metrics = {}

    # Basic statistics
    x_coords, y_coords = coords[:, 0], coords[:, 1]
    metrics['x_std'] = np.std(x_coords)
    metrics['y_std'] = np.std(y_coords)
    metrics['x_range'] = np.ptp(x_coords)
    metrics['y_range'] = np.ptp(y_coords)
    metrics['aspect_ratio'] = metrics['x_range'] / (metrics['y_range'] + 1e-10)
    metrics['total_variance'] = np.var(x_coords) + np.var(y_coords)

    # Pairwise distances (sample for large datasets)
    if n_points > 1000:
        # Sample 1000 points for large datasets
        sample_idx = np.random.choice(n_points, 1000, replace=False)
        sample_coords = coords[sample_idx]
    else:
        sample_coords = coords

    from scipy.spatial.distance import pdist
    distances = pdist(sample_coords)
    metrics['pairwise_mean'] = np.mean(distances)
    metrics['pairwise_std'] = np.std(distances)
    metrics['pairwise_median'] = np.median(distances)

    # KNN distances (small k only)
    from sklearn.neighbors import NearestNeighbors
    for k in [5, 10]:
        k_actual = min(k, n_points - 1)
        if k_actual <= 0:
            continue
        nbrs = NearestNeighbors(n_neighbors=k_actual + 1).fit(coords)
        distances_k, _ = nbrs.kneighbors(coords)
        knn_dists = distances_k[:, -1]
        metrics[f'knn_{k}_mean'] = np.mean(knn_dists)
        metrics[f'knn_{k}_std'] = np.std(knn_dists)

    # PCA shape
    from sklearn.decomposition import PCA
    pca = PCA()
    pca.fit(coords)
    var_ratios = pca.explained_variance_ratio_
    metrics['pca_var_ratio'] = var_ratios[0]
    if len(var_ratios) > 1:
        metrics['pca_elongation'] = var_ratios[0] / (var_ratios[1] + 1e-10)
    else:
        metrics['pca_elongation'] = 100

    # Convex hull
    try:
        if n_points >= 3:
            from scipy.spatial import ConvexHull
            hull = ConvexHull(coords)
            metrics['hull_area'] = hull.volume
            metrics['hull_compactness'] = (4 * np.pi * hull.volume) / (hull.area ** 2)
        else:
            metrics['hull_area'] = 0
            metrics['hull_compactness'] = 0
    except:
        metrics['hull_area'] = 0
        metrics['hull_compactness'] = 0

    return metrics

def main():
    print("Computing basic metrics for PHATE embeddings...")

    output_dir = Path("/home/btd8/manylatents/outputs/phate_k100_benchmark")
    dataset_dirs = list(output_dir.glob("*"))
    print(f"Found {len(dataset_dirs)} directories")

    results = []

    for i, dataset_dir in enumerate(dataset_dirs):
        if i % 10 == 0:
            print(f"Processing {i+1}/{len(dataset_dirs)}...")

        dataset_id = dataset_dir.name
        csv_files = list(dataset_dir.glob("embeddings_*.csv"))

        if not csv_files:
            continue

        try:
            embeddings = pd.read_csv(csv_files[0])
            if len(embeddings) < 10:
                continue

            metrics = compute_basic_metrics(embeddings, dataset_id)
            if metrics:
                metrics['dataset_id'] = dataset_id
                metrics['n_points'] = len(embeddings)
                results.append(metrics)

        except Exception as e:
            print(f"Error with {dataset_id}: {e}")
            continue

    # Save metrics
    metrics_df = pd.DataFrame(results)
    metrics_path = "/home/btd8/llm-paper-analyze/data/phate_basic_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f"\nSaved {len(results)} metrics to {metrics_path}")

    # Load labels and merge
    labels_df = pd.read_csv("/home/btd8/Documents/phate_labels_rich (1).csv")
    labeled = labels_df[labels_df['primary_structure'].notna() & (labels_df['primary_structure'] != '')]

    merged = pd.merge(metrics_df, labeled[['dataset_id', 'primary_structure']],
                     on='dataset_id', how='inner')

    print(f"Merged {len(merged)} labeled samples")
    if len(merged) > 0:
        print("Label distribution:")
        for label, count in merged['primary_structure'].value_counts().items():
            print(f"  {label}: {count}")

    return metrics_df, merged

if __name__ == "__main__":
    metrics_df, merged = main()