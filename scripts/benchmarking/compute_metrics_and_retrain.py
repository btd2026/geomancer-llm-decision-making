#!/usr/bin/env python3
"""
Compute structural metrics for the 100 PHATE embeddings and retrain model with new labels.

This script:
1. Computes embedding metrics for all 100 PHATE embeddings
2. Uses the updated manual classifications from phate_labels_rich (1).csv
3. Trains a new classifier model
4. Compares performance with predictions from the old model
"""

import pandas as pd
import numpy as np
from pathlib import Path
import pickle
from sklearn.model_selection import LeaveOneOut, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

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
        if k_actual <= 0:
            continue
        nbrs = NearestNeighbors(n_neighbors=k_actual + 1).fit(coords)
        distances_k, _ = nbrs.kneighbors(coords)
        knn_dists = distances_k[:, -1]  # k-th neighbor distance
        metrics[f'knn_{k}_mean'] = np.mean(knn_dists)
        metrics[f'knn_{k}_std'] = np.std(knn_dists)
        metrics[f'knn_{k}_max'] = np.max(knn_dists)

    # Local density
    k_density = min(10, n_points - 1)
    if k_density > 0:
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
        dbscan = DBSCAN(eps=eps, min_samples=max(3, min(10, n_points // 10)))
        labels = dbscan.fit_predict(coords)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        noise_ratio = (labels == -1).sum() / len(labels)

        metrics[f'dbscan_eps{eps_pct}_n_clusters'] = n_clusters
        metrics[f'dbscan_eps{eps_pct}_noise_ratio'] = noise_ratio

        # Silhouette score
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
        connected = (dist_matrix < threshold).sum(axis=1) - 1
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

    # Spatial entropy
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

def compute_all_phate_metrics():
    """Compute metrics for all 100 PHATE embeddings."""
    print("Computing metrics for all PHATE embeddings...")

    output_dir = Path("/home/btd8/manylatents/outputs/phate_k100_benchmark")
    dataset_dirs = list(output_dir.glob("*"))

    print(f"Found {len(dataset_dirs)} PHATE embedding directories")

    results = []

    for i, dataset_dir in enumerate(dataset_dirs):
        if i % 20 == 0:
            print(f"  Processing {i+1}/{len(dataset_dirs)}...")

        dataset_id = dataset_dir.name

        # Find CSV file
        csv_files = list(dataset_dir.glob("embeddings_*.csv"))
        if not csv_files:
            print(f"    Warning: No CSV found for {dataset_id}")
            continue

        csv_path = csv_files[0]

        try:
            # Load embeddings
            embeddings = pd.read_csv(csv_path)

            if len(embeddings) < 10:
                print(f"    Warning: Too few points ({len(embeddings)}) for {dataset_id}")
                continue

            # Compute metrics
            metrics = compute_embedding_metrics(embeddings, dataset_id)

            if metrics is None:
                continue

            metrics['dataset_id'] = dataset_id
            metrics['n_points'] = len(embeddings)
            results.append(metrics)

        except Exception as e:
            print(f"    Error processing {dataset_id}: {e}")
            continue

    print(f"Successfully computed metrics for {len(results)} datasets")
    return pd.DataFrame(results)

def main():
    print("="*70)
    print("COMPUTING METRICS AND RETRAINING WITH NEW LABELS")
    print("="*70)

    # Step 1: Compute metrics for all PHATE embeddings
    metrics_df = compute_all_phate_metrics()

    # Step 2: Load updated manual classifications
    print("\nLoading updated manual classifications...")
    labels_df = pd.read_csv("/home/btd8/Documents/phate_labels_rich (1).csv")

    # Filter to labeled samples
    labeled = labels_df[labels_df['primary_structure'].notna() & (labels_df['primary_structure'] != '')]
    print(f"Found {len(labeled)} labeled samples in the new data")

    print("Label distribution:")
    for label, count in labeled['primary_structure'].value_counts().items():
        print(f"  {label}: {count}")

    # Step 3: Merge metrics with labels
    print("\nMerging metrics with labels...")
    merged = pd.merge(metrics_df, labeled[['dataset_id', 'primary_structure']],
                     on='dataset_id', how='inner')

    print(f"Successfully merged {len(merged)} samples with labels and metrics")

    if len(merged) == 0:
        print("ERROR: No overlap between computed metrics and labeled data!")
        return

    # Step 4: Prepare training data
    print("\nPreparing training data...")

    # Define feature columns (exclude metadata)
    exclude_cols = ['dataset_id', 'n_points', 'primary_structure']
    feature_cols = [c for c in merged.columns if c not in exclude_cols]

    X = merged[feature_cols].copy()
    y = merged['primary_structure'].copy()

    # Handle missing values
    X = X.replace([np.inf, -np.inf], np.nan)

    # Fill NaN with median
    for col in X.columns:
        if X[col].isnull().any():
            median_val = X[col].median()
            if pd.isna(median_val):
                X[col] = X[col].fillna(0)
            else:
                X[col] = X[col].fillna(median_val)

    print(f"Training data shape: {X.shape}")
    print(f"Number of classes: {y.nunique()}")

    # Step 5: Train classifier
    print("\nTraining classifier...")

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # Use SVM with RBF kernel
    clf = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', SVC(kernel='rbf', class_weight='balanced', random_state=42, probability=True))
    ])

    # Cross-validation
    loo = LeaveOneOut()
    scores = cross_val_score(clf, X, y_encoded, cv=loo, scoring='accuracy')

    print(f"Leave-One-Out CV Accuracy: {scores.mean():.3f} ± {scores.std():.3f}")

    # Train on full data
    clf.fit(X, y_encoded)

    # Step 6: Save new model
    print("\nSaving updated model...")

    model_data = {
        'classifier': clf,
        'label_encoder': le,
        'feature_cols': feature_cols,
        'training_samples': len(X),
        'training_labels': y.tolist(),
        'cv_accuracy': scores.mean(),
        'cv_std': scores.std()
    }

    model_path = "/home/btd8/llm-paper-analyze/models/phate_structure_classifier_updated.pkl"
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)

    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)

    # Step 7: Predict on all computed metrics
    print("\nPredicting structures for all datasets...")

    # Prepare all data for prediction
    X_all = metrics_df[feature_cols].copy()

    # Handle missing values
    X_all = X_all.replace([np.inf, -np.inf], np.nan)
    for col in X_all.columns:
        if X_all[col].isnull().any():
            median_val = X[col].median()  # Use training median
            if pd.isna(median_val):
                X_all[col] = X_all[col].fillna(0)
            else:
                X_all[col] = X_all[col].fillna(median_val)

    # Predict
    y_pred_encoded = clf.predict(X_all)
    y_pred_proba = clf.predict_proba(X_all)
    y_pred = le.inverse_transform(y_pred_encoded)

    # Create results
    results_df = metrics_df[['dataset_id', 'n_points']].copy()
    results_df['predicted_structure'] = y_pred
    results_df['confidence'] = y_pred_proba.max(axis=1)

    # Add individual class probabilities
    for i, class_name in enumerate(le.classes_):
        results_df[f'prob_{class_name}'] = y_pred_proba[:, i]

    # Save results
    results_path = "/home/btd8/llm-paper-analyze/data/phate_updated_predictions.csv"
    results_df.to_csv(results_path, index=False)

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"✓ Computed metrics for {len(metrics_df)} PHATE embeddings")
    print(f"✓ Trained on {len(merged)} labeled samples")
    print(f"✓ Cross-validation accuracy: {scores.mean():.3f} ± {scores.std():.3f}")
    print(f"✓ Model saved to: {model_path}")
    print(f"✓ Predictions saved to: {results_path}")

    print(f"\nPredicted structure distribution:")
    for structure, count in results_df['predicted_structure'].value_counts().items():
        pct = count / len(results_df) * 100
        print(f"  {structure}: {count} ({pct:.1f}%)")

    return results_df, clf, le

if __name__ == "__main__":
    results_df, clf, le = main()