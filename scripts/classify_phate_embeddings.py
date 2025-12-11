#!/usr/bin/env python3
"""
Classify PHATE embeddings from the 100 small datasets using trained classifier.

Loads embeddings from /home/btd8/manylatents/outputs/phate_k100_benchmark/
Computes structural metrics and predicts structure classes.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import glob
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

# Import metrics computation functions
import sys
sys.path.append('/home/btd8/llm-paper-analyze/scripts')

def compute_embedding_metrics(embedding_data, dataset_id):
    """Compute structural metrics for a 2D embedding."""
    from scipy.spatial.distance import pdist, squareform
    from scipy.spatial import ConvexHull
    from sklearn.decomposition import PCA
    from sklearn.cluster import DBSCAN
    from sklearn.neighbors import NearestNeighbors

    # Extract coordinates
    coords = embedding_data[['dim_1', 'dim_2']].values
    n_points = len(coords)

    if n_points < 10:
        return None  # Skip if too few points

    metrics = {}

    # Basic statistics
    x_coords, y_coords = coords[:, 0], coords[:, 1]
    metrics['x_mean'] = np.mean(x_coords)
    metrics['y_mean'] = np.mean(y_coords)
    metrics['x_std'] = np.std(x_coords)
    metrics['y_std'] = np.std(y_coords)
    metrics['x_range'] = np.ptp(x_coords)
    metrics['y_range'] = np.ptp(y_coords)
    metrics['aspect_ratio'] = metrics['x_range'] / (metrics['y_range'] + 1e-10)
    metrics['total_variance'] = np.var(x_coords) + np.var(y_coords)

    # Pairwise distances
    distances = pdist(coords)
    metrics['pairwise_mean'] = np.mean(distances)
    metrics['pairwise_std'] = np.std(distances)
    metrics['pairwise_median'] = np.median(distances)
    metrics['pairwise_q25'] = np.percentile(distances, 25)
    metrics['pairwise_q75'] = np.percentile(distances, 75)
    metrics['pairwise_iqr'] = metrics['pairwise_q75'] - metrics['pairwise_q25']

    # KNN distances
    for k in [5, 10, 20, 50]:
        k_actual = min(k, n_points - 1)
        nbrs = NearestNeighbors(n_neighbors=k_actual + 1).fit(coords)
        distances_k, _ = nbrs.kneighbors(coords)
        knn_dists = distances_k[:, -1]  # k-th neighbor distance
        metrics[f'knn_{k}_mean'] = np.mean(knn_dists)
        metrics[f'knn_{k}_std'] = np.std(knn_dists)
        metrics[f'knn_{k}_max'] = np.max(knn_dists)

    # Local density (using 10th nearest neighbor)
    k_density = min(10, n_points - 1)
    nbrs = NearestNeighbors(n_neighbors=k_density + 1).fit(coords)
    distances_density, _ = nbrs.kneighbors(coords)
    densities = 1 / (distances_density[:, -1] + 1e-10)
    metrics['density_mean'] = np.mean(densities)
    metrics['density_std'] = np.std(densities)
    metrics['density_cv'] = metrics['density_std'] / (metrics['density_mean'] + 1e-10)
    metrics['density_skew'] = pd.Series(densities).skew()

    # DBSCAN clustering at different scales
    for eps_pct in [5, 10, 25]:
        eps = np.percentile(distances, eps_pct)
        dbscan = DBSCAN(eps=eps, min_samples=3)
        labels = dbscan.fit_predict(coords)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        noise_ratio = (labels == -1).sum() / len(labels)

        metrics[f'dbscan_eps{eps_pct}_n_clusters'] = n_clusters
        metrics[f'dbscan_eps{eps_pct}_noise_ratio'] = noise_ratio

        # Silhouette score (if we have clusters)
        if n_clusters > 1:
            from sklearn.metrics import silhouette_score
            try:
                sil_score = silhouette_score(coords, labels)
                metrics[f'dbscan_eps{eps_pct}_silhouette'] = sil_score
            except:
                metrics[f'dbscan_eps{eps_pct}_silhouette'] = -1
        else:
            metrics[f'dbscan_eps{eps_pct}_silhouette'] = -1

    # Connectivity metrics
    for pct in [10, 25, 50]:
        threshold = np.percentile(distances, pct)
        dist_matrix = squareform(distances)
        connected = (dist_matrix < threshold).sum(axis=1) - 1  # -1 to exclude self
        metrics[f'connectivity_p{pct}'] = np.mean(connected) / n_points

    # PCA shape metrics
    pca = PCA()
    pca.fit(coords)
    var_ratios = pca.explained_variance_ratio_
    metrics['pca_var_ratio'] = var_ratios[0]
    if len(var_ratios) > 1:
        metrics['pca_elongation'] = var_ratios[0] / (var_ratios[1] + 1e-10)
    else:
        metrics['pca_elongation'] = 100

    # Convex hull properties
    try:
        if n_points >= 3:
            hull = ConvexHull(coords)
            metrics['hull_area'] = hull.volume
            metrics['hull_perimeter'] = hull.area
            # Compactness (1.0 = circle)
            metrics['hull_compactness'] = (4 * np.pi * hull.volume) / (hull.area ** 2)
            metrics['point_density_in_hull'] = n_points / hull.volume
        else:
            metrics['hull_area'] = 0
            metrics['hull_perimeter'] = 0
            metrics['hull_compactness'] = 0
            metrics['point_density_in_hull'] = 0
    except:
        metrics['hull_area'] = 0
        metrics['hull_perimeter'] = 0
        metrics['hull_compactness'] = 0
        metrics['point_density_in_hull'] = 0

    # Spatial entropy (uniformity of distribution)
    # Divide space into grid and compute entropy
    try:
        n_bins = min(10, int(np.sqrt(n_points)))
        hist, _, _ = np.histogram2d(x_coords, y_coords, bins=n_bins)
        hist = hist.flatten()
        hist = hist / hist.sum()
        hist = hist[hist > 0]
        entropy = -np.sum(hist * np.log(hist))
        max_entropy = np.log(len(hist))
        metrics['spatial_entropy'] = entropy
        metrics['spatial_entropy_normalized'] = entropy / (max_entropy + 1e-10)
    except:
        metrics['spatial_entropy'] = 0
        metrics['spatial_entropy_normalized'] = 0

    # Distance from center metrics
    center = np.mean(coords, axis=0)
    center_distances = np.linalg.norm(coords - center, axis=1)
    metrics['center_dist_mean'] = np.mean(center_distances)
    metrics['center_dist_std'] = np.std(center_distances)
    metrics['center_dist_skew'] = pd.Series(center_distances).skew()

    return metrics

def load_training_data():
    """Load the labeled training data and train classifier."""
    print("Loading training data...")

    # Load metrics and labels
    metrics_df = pd.read_csv("/home/btd8/llm-paper-analyze/data/manylatents_benchmark/embedding_metrics.csv")
    labels_df = pd.read_csv("/home/btd8/Documents/phate_labels_rich.csv")

    # Filter labeled
    labeled = labels_df[labels_df['primary_structure'].notna() & (labels_df['primary_structure'] != '')]

    # Merge
    merge_cols = ['dataset_id', 'primary_structure', 'flagged']
    merge_cols = [c for c in merge_cols if c in labeled.columns]
    merged = pd.merge(metrics_df, labeled[merge_cols], on='dataset_id', how='inner')

    # Remove flagged
    if 'flagged' in merged.columns:
        merged = merged[~merged['flagged'].fillna(False).astype(bool)]

    print(f"Training data: {len(merged)} labeled samples")

    # Prepare features
    EXCLUDE_COLS = ['dataset_id', 'dataset_name', 'size_mb', 'num_features', 'n_points', 'original_n_points']
    label_cols = ['primary_structure', 'flagged']
    feature_cols = [c for c in merged.columns if c not in EXCLUDE_COLS and c not in label_cols]

    X_train = merged[feature_cols].copy()
    y_train = merged['primary_structure'].copy()

    # Handle NaN/inf
    X_train = X_train.replace([np.inf, -np.inf], np.nan).fillna(X_train.median())

    return X_train, y_train, feature_cols

def process_phate_embeddings():
    """Process all PHATE embeddings and compute metrics."""
    output_dir = Path("/home/btd8/manylatents/outputs/phate_k100_benchmark")
    dataset_dirs = list(output_dir.glob("*"))

    print(f"Processing {len(dataset_dirs)} PHATE embedding directories...")

    results = []

    for i, dataset_dir in enumerate(dataset_dirs):
        if i % 20 == 0:
            print(f"  Processing {i+1}/{len(dataset_dirs)}...")

        dataset_id = dataset_dir.name

        # Find CSV file
        csv_files = list(dataset_dir.glob("embeddings_*.csv"))
        if not csv_files:
            print(f"  Warning: No CSV found for {dataset_id}")
            continue

        csv_path = csv_files[0]

        try:
            # Load embeddings
            embeddings = pd.read_csv(csv_path)

            if len(embeddings) < 10:
                print(f"  Warning: Too few points ({len(embeddings)}) for {dataset_id}")
                continue

            # Compute metrics
            metrics = compute_embedding_metrics(embeddings, dataset_id)

            if metrics is None:
                continue

            metrics['dataset_id'] = dataset_id
            metrics['n_points'] = len(embeddings)
            results.append(metrics)

        except Exception as e:
            print(f"  Error processing {dataset_id}: {e}")
            continue

    print(f"Successfully processed {len(results)} datasets")
    return pd.DataFrame(results)

def main():
    print("="*70)
    print("CLASSIFYING 100 PHATE EMBEDDINGS")
    print("="*70)

    # Load training data
    X_train, y_train, feature_cols = load_training_data()

    # Process new PHATE embeddings
    new_metrics_df = process_phate_embeddings()

    if len(new_metrics_df) == 0:
        print("No data to process!")
        return

    print(f"\nComputed metrics for {len(new_metrics_df)} new datasets")

    # Prepare new data features (align with training features)
    common_features = [f for f in feature_cols if f in new_metrics_df.columns]
    print(f"Using {len(common_features)} common features (out of {len(feature_cols)} training features)")

    # Missing features - fill with median from training data
    X_new = pd.DataFrame(index=new_metrics_df.index)
    for feat in feature_cols:
        if feat in new_metrics_df.columns:
            X_new[feat] = new_metrics_df[feat]
        else:
            # Fill missing feature with training median
            X_new[feat] = X_train[feat].median()

    # Handle NaN/inf in new data
    X_new = X_new.replace([np.inf, -np.inf], np.nan).fillna(X_train.median())

    # Train classifier on full training data
    print(f"\nTraining classifier on {len(y_train)} samples...")

    # Use SVM (best performer from previous results)
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)

    clf = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', SVC(kernel='rbf', class_weight='balanced', random_state=42, probability=True))
    ])

    clf.fit(X_train, y_train_encoded)

    # Predict on new data
    print(f"Predicting on {len(X_new)} new datasets...")
    y_pred_encoded = clf.predict(X_new)
    y_pred_proba = clf.predict_proba(X_new)

    # Convert back to class names
    y_pred = le.inverse_transform(y_pred_encoded)

    # Create results dataframe
    results_df = new_metrics_df[['dataset_id', 'n_points']].copy()
    results_df['predicted_structure'] = y_pred

    # Add prediction probabilities
    for i, class_name in enumerate(le.classes_):
        results_df[f'prob_{class_name}'] = y_pred_proba[:, i]

    # Add confidence (max probability)
    results_df['confidence'] = y_pred_proba.max(axis=1)

    # Sort by confidence
    results_df = results_df.sort_values('confidence', ascending=False)

    # Save results
    output_path = "/home/btd8/llm-paper-analyze/data/phate_classification_results.csv"
    results_df.to_csv(output_path, index=False)

    print(f"\nResults saved to: {output_path}")

    # Summary statistics
    print("\n" + "="*70)
    print("CLASSIFICATION SUMMARY")
    print("="*70)

    print(f"\nPredicted structure distribution:")
    structure_counts = results_df['predicted_structure'].value_counts()
    for structure, count in structure_counts.items():
        pct = count / len(results_df) * 100
        print(f"  {structure}: {count} ({pct:.1f}%)")

    print(f"\nConfidence statistics:")
    print(f"  Mean confidence: {results_df['confidence'].mean():.3f}")
    print(f"  Median confidence: {results_df['confidence'].median():.3f}")
    print(f"  High confidence (>0.7): {(results_df['confidence'] > 0.7).sum()}")
    print(f"  Low confidence (<0.4): {(results_df['confidence'] < 0.4).sum()}")

    print(f"\nTop 10 most confident predictions:")
    for _, row in results_df.head(10).iterrows():
        print(f"  {row['dataset_id'][:36]}: {row['predicted_structure']} ({row['confidence']:.3f})")

    print(f"\n10 least confident predictions:")
    for _, row in results_df.tail(10).iterrows():
        print(f"  {row['dataset_id'][:36]}: {row['predicted_structure']} ({row['confidence']:.3f})")

    return results_df

if __name__ == "__main__":
    results = main()