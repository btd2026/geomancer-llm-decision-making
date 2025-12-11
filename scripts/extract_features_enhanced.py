#!/usr/bin/env python3
"""
Enhanced feature extraction for PHATE embeddings with more sophisticated structural metrics.
This will create a more robust training dataset for improved classification.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def compute_advanced_metrics(embedding_data, dataset_id):
    """Compute comprehensive structural metrics for a 2D embedding."""
    from scipy.spatial.distance import pdist, squareform
    from scipy.spatial import ConvexHull, distance_matrix
    from scipy.stats import entropy, zscore
    from sklearn.decomposition import PCA
    from sklearn.cluster import DBSCAN, KMeans
    from sklearn.neighbors import NearestNeighbors
    from sklearn.metrics import silhouette_score

    coords = embedding_data[['dim_1', 'dim_2']].values
    n_points = len(coords)

    if n_points < 10:
        return None

    metrics = {}

    # === BASIC GEOMETRIC PROPERTIES ===
    x_coords, y_coords = coords[:, 0], coords[:, 1]

    # Central moments and shape
    metrics['x_mean'] = np.mean(x_coords)
    metrics['y_mean'] = np.mean(y_coords)
    metrics['x_std'] = np.std(x_coords)
    metrics['y_std'] = np.std(y_coords)
    metrics['x_skew'] = pd.Series(x_coords).skew()
    metrics['y_skew'] = pd.Series(y_coords).skew()
    metrics['x_kurt'] = pd.Series(x_coords).kurtosis()
    metrics['y_kurt'] = pd.Series(y_coords).kurtosis()

    # Range and aspect ratio
    x_range = np.ptp(x_coords)
    y_range = np.ptp(y_coords)
    metrics['x_range'] = x_range
    metrics['y_range'] = y_range
    metrics['aspect_ratio'] = x_range / (y_range + 1e-10)
    metrics['range_ratio'] = min(x_range, y_range) / (max(x_range, y_range) + 1e-10)

    # === DISTANCE-BASED METRICS ===
    # Sample for large datasets to manage memory
    if n_points > 2000:
        sample_idx = np.random.choice(n_points, 2000, replace=False)
        sample_coords = coords[sample_idx]
    else:
        sample_coords = coords

    distances = pdist(sample_coords)

    # Distance statistics
    metrics['dist_mean'] = np.mean(distances)
    metrics['dist_std'] = np.std(distances)
    metrics['dist_median'] = np.median(distances)
    metrics['dist_skew'] = pd.Series(distances).skew()
    metrics['dist_q25'] = np.percentile(distances, 25)
    metrics['dist_q75'] = np.percentile(distances, 75)
    metrics['dist_iqr'] = metrics['dist_q75'] - metrics['dist_q25']
    metrics['dist_cv'] = metrics['dist_std'] / (metrics['dist_mean'] + 1e-10)

    # === NEIGHBORHOOD ANALYSIS ===
    for k in [5, 10, 20, min(50, n_points-1)]:
        if k >= n_points:
            continue

        nbrs = NearestNeighbors(n_neighbors=k + 1).fit(coords)
        distances_k, indices_k = nbrs.kneighbors(coords)
        knn_dists = distances_k[:, -1]  # k-th neighbor distance

        metrics[f'knn_{k}_mean'] = np.mean(knn_dists)
        metrics[f'knn_{k}_std'] = np.std(knn_dists)
        metrics[f'knn_{k}_cv'] = np.std(knn_dists) / (np.mean(knn_dists) + 1e-10)
        metrics[f'knn_{k}_skew'] = pd.Series(knn_dists).skew()

    # Local density estimation
    k_density = min(10, n_points - 1)
    if k_density > 0:
        nbrs = NearestNeighbors(n_neighbors=k_density + 1).fit(coords)
        distances_density, _ = nbrs.kneighbors(coords)
        densities = 1 / (distances_density[:, -1] + 1e-10)

        metrics['density_mean'] = np.mean(densities)
        metrics['density_std'] = np.std(densities)
        metrics['density_cv'] = metrics['density_std'] / (metrics['density_mean'] + 1e-10)
        metrics['density_skew'] = pd.Series(densities).skew()
        metrics['density_gini'] = np.sum(np.abs(np.subtract.outer(densities, densities))) / (2 * len(densities) * np.mean(densities))

    # === CLUSTERING ANALYSIS ===
    # DBSCAN at multiple scales
    for eps_pct in [5, 10, 20, 30]:
        eps = np.percentile(distances, eps_pct)
        min_samples = max(3, min(10, n_points // 20))

        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(coords)

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        noise_ratio = (labels == -1).sum() / len(labels)

        metrics[f'dbscan_eps{eps_pct}_clusters'] = n_clusters
        metrics[f'dbscan_eps{eps_pct}_noise'] = noise_ratio

        if n_clusters > 1:
            try:
                sil_score = silhouette_score(coords, labels)
                metrics[f'dbscan_eps{eps_pct}_silhouette'] = sil_score
            except:
                metrics[f'dbscan_eps{eps_pct}_silhouette'] = -1
        else:
            metrics[f'dbscan_eps{eps_pct}_silhouette'] = -1

    # K-means clustering for different k values
    for n_clusters_k in [2, 3, 5, 8]:
        if n_clusters_k >= n_points:
            continue

        try:
            kmeans = KMeans(n_clusters=n_clusters_k, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(coords)

            # Inertia (within-cluster sum of squares)
            metrics[f'kmeans_{n_clusters_k}_inertia'] = kmeans.inertia_

            # Silhouette score
            sil_score = silhouette_score(coords, cluster_labels)
            metrics[f'kmeans_{n_clusters_k}_silhouette'] = sil_score

            # Cluster balance (entropy of cluster sizes)
            cluster_sizes = np.bincount(cluster_labels)
            cluster_probs = cluster_sizes / cluster_sizes.sum()
            metrics[f'kmeans_{n_clusters_k}_balance'] = entropy(cluster_probs)

        except:
            metrics[f'kmeans_{n_clusters_k}_inertia'] = np.inf
            metrics[f'kmeans_{n_clusters_k}_silhouette'] = -1
            metrics[f'kmeans_{n_clusters_k}_balance'] = 0

    # === SHAPE ANALYSIS ===
    # PCA-based shape analysis
    pca = PCA()
    pca_coords = pca.fit_transform(coords)
    var_ratios = pca.explained_variance_ratio_

    metrics['pca_var_1'] = var_ratios[0]
    metrics['pca_var_2'] = var_ratios[1] if len(var_ratios) > 1 else 0
    metrics['pca_elongation'] = var_ratios[0] / (var_ratios[1] + 1e-10)
    metrics['pca_compactness'] = var_ratios[1] / (var_ratios[0] + 1e-10)

    # Convex hull analysis
    try:
        if n_points >= 3:
            hull = ConvexHull(coords)
            hull_area = hull.volume
            hull_perimeter = hull.area

            metrics['hull_area'] = hull_area
            metrics['hull_perimeter'] = hull_perimeter
            metrics['hull_compactness'] = (4 * np.pi * hull_area) / (hull_perimeter ** 2)
            metrics['hull_point_density'] = n_points / hull_area

            # Hull vertex ratio
            metrics['hull_vertex_ratio'] = len(hull.vertices) / n_points

        else:
            for key in ['hull_area', 'hull_perimeter', 'hull_compactness', 'hull_point_density', 'hull_vertex_ratio']:
                metrics[key] = 0
    except:
        for key in ['hull_area', 'hull_perimeter', 'hull_compactness', 'hull_point_density', 'hull_vertex_ratio']:
            metrics[key] = 0

    # === TRAJECTORY-LIKE FEATURES ===
    # Minimum spanning tree analysis (for trajectory detection)
    try:
        from scipy.sparse.csgraph import minimum_spanning_tree
        dist_matrix = squareform(distances)
        mst = minimum_spanning_tree(dist_matrix[:len(sample_coords), :len(sample_coords)])
        mst_edges = mst.toarray()
        mst_weights = mst_edges[mst_edges > 0]

        metrics['mst_total_weight'] = np.sum(mst_weights)
        metrics['mst_mean_edge'] = np.mean(mst_weights)
        metrics['mst_std_edge'] = np.std(mst_weights)
        metrics['mst_max_edge'] = np.max(mst_weights)

    except:
        for key in ['mst_total_weight', 'mst_mean_edge', 'mst_std_edge', 'mst_max_edge']:
            metrics[key] = 0

    # === SPATIAL DISTRIBUTION ===
    # Grid-based spatial entropy
    try:
        n_bins = min(10, int(np.sqrt(n_points // 4)))
        if n_bins >= 2:
            hist, _, _ = np.histogram2d(x_coords, y_coords, bins=n_bins)
            hist_flat = hist.flatten()
            hist_probs = hist_flat / hist_flat.sum()
            hist_probs = hist_probs[hist_probs > 0]

            spatial_entropy = entropy(hist_probs)
            max_entropy = np.log(len(hist_probs))

            metrics['spatial_entropy'] = spatial_entropy
            metrics['spatial_entropy_norm'] = spatial_entropy / (max_entropy + 1e-10)
            metrics['occupied_bins'] = len(hist_probs)
            metrics['bin_occupation_rate'] = len(hist_probs) / (n_bins * n_bins)
        else:
            for key in ['spatial_entropy', 'spatial_entropy_norm', 'occupied_bins', 'bin_occupation_rate']:
                metrics[key] = 0
    except:
        for key in ['spatial_entropy', 'spatial_entropy_norm', 'occupied_bins', 'bin_occupation_rate']:
            metrics[key] = 0

    # === CENTER-BASED ANALYSIS ===
    # Distance from various centers
    centroid = np.mean(coords, axis=0)
    median_point = np.median(coords, axis=0)

    # Centroid distances
    centroid_dists = np.linalg.norm(coords - centroid, axis=1)
    metrics['centroid_dist_mean'] = np.mean(centroid_dists)
    metrics['centroid_dist_std'] = np.std(centroid_dists)
    metrics['centroid_dist_skew'] = pd.Series(centroid_dists).skew()
    metrics['centroid_dist_max'] = np.max(centroid_dists)

    # Median point distances
    median_dists = np.linalg.norm(coords - median_point, axis=1)
    metrics['median_dist_mean'] = np.mean(median_dists)
    metrics['median_dist_std'] = np.std(median_dists)

    # === CONNECTIVITY ANALYSIS ===
    # Graph connectivity at different thresholds
    for pct in [5, 10, 20, 30]:
        threshold = np.percentile(distances, pct)

        # For sampled coordinates to manage memory
        if len(sample_coords) < 1000:
            dist_matrix_sample = squareform(distances)
            connected_matrix = (dist_matrix_sample < threshold).astype(int)
            connected_counts = connected_matrix.sum(axis=1) - 1  # -1 for self

            metrics[f'connectivity_p{pct}_mean'] = np.mean(connected_counts) / len(sample_coords)
            metrics[f'connectivity_p{pct}_std'] = np.std(connected_counts)
            metrics[f'connectivity_p{pct}_max'] = np.max(connected_counts)
        else:
            # Simplified for large datasets
            metrics[f'connectivity_p{pct}_mean'] = 0
            metrics[f'connectivity_p{pct}_std'] = 0
            metrics[f'connectivity_p{pct}_max'] = 0

    return metrics

def compute_enhanced_metrics_for_all():
    """Compute enhanced metrics for all datasets."""
    print("="*70)
    print("EXTRACTING ENHANCED FEATURES FOR ALL DATASETS")
    print("="*70)

    # Process the 100 PHATE datasets
    print("Processing 100 PHATE datasets with enhanced features...")
    output_dir = Path("/home/btd8/manylatents/outputs/phate_k100_benchmark")
    dataset_dirs = list(output_dir.glob("*"))

    all_results = []

    for i, dataset_dir in enumerate(dataset_dirs):
        if i % 20 == 0:
            print(f"  Processing {i+1}/{len(dataset_dirs)}...")

        dataset_id = dataset_dir.name
        csv_files = list(dataset_dir.glob("embeddings_*.csv"))

        if not csv_files:
            continue

        try:
            embeddings = pd.read_csv(csv_files[0])
            if len(embeddings) < 10:
                continue

            # Set random seed for reproducible sampling in large datasets
            np.random.seed(42)

            metrics = compute_advanced_metrics(embeddings, dataset_id)
            if metrics:
                metrics['dataset_id'] = dataset_id
                metrics['n_points'] = len(embeddings)
                metrics['dataset_source'] = 'phate_100'
                all_results.append(metrics)

        except Exception as e:
            print(f"    Error processing {dataset_id}: {e}")
            continue

    print(f"Extracted enhanced features for {len(all_results)} datasets")

    # Save enhanced metrics
    enhanced_df = pd.DataFrame(all_results)
    enhanced_path = "/home/btd8/llm-paper-analyze/data/enhanced_phate_metrics.csv"
    enhanced_df.to_csv(enhanced_path, index=False)

    print(f"✓ Enhanced metrics saved to: {enhanced_path}")
    print(f"✓ Feature matrix shape: {enhanced_df.shape}")

    # Show feature summary
    feature_cols = [c for c in enhanced_df.columns if c not in ['dataset_id', 'n_points', 'dataset_source']]
    print(f"✓ Total features extracted: {len(feature_cols)}")

    return enhanced_df, feature_cols

def main():
    # Compute enhanced metrics
    enhanced_df, feature_cols = compute_enhanced_metrics_for_all()

    print(f"\n📊 Enhanced Feature Summary:")
    print(f"   • Basic geometric: shape, moments, ranges")
    print(f"   • Distance-based: pairwise distances, neighborhoods")
    print(f"   • Clustering: DBSCAN and K-means at multiple scales")
    print(f"   • Shape analysis: PCA, convex hull properties")
    print(f"   • Trajectory features: minimum spanning tree")
    print(f"   • Spatial distribution: entropy, grid occupation")
    print(f"   • Connectivity: graph connectivity at thresholds")

    return enhanced_df, feature_cols

if __name__ == "__main__":
    enhanced_df, features = main()