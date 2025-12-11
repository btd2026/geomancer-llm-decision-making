#!/usr/bin/env python3
"""
Classify the 100 PHATE embeddings using the existing training data.

Since the new manual labels are for different datasets than our computed embeddings,
we'll use the original training data to classify our 100 datasets.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import pickle
from sklearn.model_selection import LeaveOneOut, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

def load_original_training_data():
    """Load the original training data."""
    print("Loading original training data...")

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
        flagged_count = merged['flagged'].fillna(False).astype(bool).sum()
        if flagged_count > 0:
            print(f"Removing {flagged_count} flagged samples")
            merged = merged[~merged['flagged'].fillna(False).astype(bool)]

    print(f"Training data: {len(merged)} labeled samples")
    print("\nLabel distribution:")
    for label, count in merged['primary_structure'].value_counts().items():
        print(f"  {label}: {count}")

    # Prepare features
    EXCLUDE_COLS = ['dataset_id', 'dataset_name', 'size_mb', 'num_features', 'n_points', 'original_n_points']
    label_cols = ['primary_structure', 'flagged']
    feature_cols = [c for c in merged.columns if c not in EXCLUDE_COLS and c not in label_cols]

    X = merged[feature_cols].copy()
    y = merged['primary_structure'].copy()

    # Handle NaN/inf
    X = X.replace([np.inf, -np.inf], np.nan).fillna(X.median())

    return X, y, feature_cols

def align_features(new_data, training_feature_cols, training_medians):
    """Align new data features with training features."""
    X_aligned = pd.DataFrame(index=new_data.index)

    for col in training_feature_cols:
        if col in new_data.columns:
            X_aligned[col] = new_data[col]
        else:
            # Missing feature - use training median
            X_aligned[col] = training_medians[col]

    # Handle NaN/inf with training medians
    X_aligned = X_aligned.replace([np.inf, -np.inf], np.nan)
    for col in X_aligned.columns:
        X_aligned[col] = X_aligned[col].fillna(training_medians[col])

    return X_aligned

def main():
    print("="*70)
    print("CLASSIFYING 100 PHATE EMBEDDINGS WITH ORIGINAL TRAINING DATA")
    print("="*70)

    # Step 1: Load original training data
    X_train, y_train, feature_cols = load_original_training_data()

    # Step 2: Train classifier
    print(f"\nTraining classifier with {len(feature_cols)} features...")

    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)

    clf = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', SVC(kernel='rbf', class_weight='balanced', random_state=42, probability=True))
    ])

    # Cross-validation
    loo = LeaveOneOut()
    scores = cross_val_score(clf, X_train, y_train_encoded, cv=loo, scoring='accuracy')
    print(f"Cross-validation accuracy: {scores.mean():.3f} ± {scores.std():.3f}")

    # Train on full data
    clf.fit(X_train, y_train_encoded)

    # Step 3: Load new computed metrics
    print(f"\nLoading computed metrics for 100 datasets...")
    new_metrics = pd.read_csv("/home/btd8/llm-paper-analyze/data/phate_basic_metrics.csv")
    print(f"Loaded metrics for {len(new_metrics)} datasets")

    # Step 4: Align features
    print(f"Aligning features...")
    training_medians = X_train.median()
    X_new = align_features(new_metrics, feature_cols, training_medians)

    print(f"Aligned feature matrix: {X_new.shape}")

    # Check feature overlap
    common_features = [f for f in feature_cols if f in new_metrics.columns]
    print(f"Common features: {len(common_features)}/{len(feature_cols)}")

    # Step 5: Predict
    print(f"\nPredicting structures...")
    y_pred_encoded = clf.predict(X_new)
    y_pred_proba = clf.predict_proba(X_new)
    y_pred = le.inverse_transform(y_pred_encoded)

    # Step 6: Create results
    results_df = new_metrics[['dataset_id', 'n_points']].copy()
    results_df['predicted_structure'] = y_pred
    results_df['confidence'] = y_pred_proba.max(axis=1)

    # Add individual class probabilities
    for i, class_name in enumerate(le.classes_):
        results_df[f'prob_{class_name}'] = y_pred_proba[:, i]

    # Sort by confidence
    results_df = results_df.sort_values('confidence', ascending=False)

    # Save results
    output_path = "/home/btd8/llm-paper-analyze/data/phate_structure_predictions.csv"
    results_df.to_csv(output_path, index=False)

    # Step 7: Summary
    print("\n" + "="*70)
    print("CLASSIFICATION RESULTS")
    print("="*70)

    print(f"✓ Predicted structures for {len(results_df)} datasets")
    print(f"✓ Results saved to: {output_path}")

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

    # Save model for future use
    model_data = {
        'classifier': clf,
        'label_encoder': le,
        'feature_cols': feature_cols,
        'training_medians': training_medians,
        'cv_accuracy': scores.mean(),
        'cv_std': scores.std()
    }

    model_path = "/home/btd8/llm-paper-analyze/models/phate_structure_classifier.pkl"
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)

    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)

    print(f"\n✓ Model saved to: {model_path}")

    return results_df

if __name__ == "__main__":
    results = main()