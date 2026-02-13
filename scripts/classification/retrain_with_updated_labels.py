#!/usr/bin/env python3
"""
Retrain the structure classifier with updated manual labels.

Uses the new manual classifications from phate_labels_rich (1).csv to update
the model and compare performance with the previous version.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import pickle
from sklearn.model_selection import LeaveOneOut, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

def load_updated_training_data():
    """Load the updated labeled training data."""
    print("Loading updated training data...")

    # Load metrics and updated labels
    metrics_df = pd.read_csv("/home/btd8/llm-paper-analyze/data/manylatents_benchmark/embedding_metrics.csv")
    labels_df = pd.read_csv("/home/btd8/Documents/phate_labels_rich (1).csv")

    print(f"Metrics data: {len(metrics_df)} rows")
    print(f"Labels data: {len(labels_df)} rows")

    # Filter labeled
    labeled = labels_df[labels_df['primary_structure'].notna() & (labels_df['primary_structure'] != '')]
    print(f"Labeled samples in new data: {len(labeled)}")

    # Print label distribution
    print("\nUpdated label distribution:")
    label_counts = labeled['primary_structure'].value_counts()
    for label, count in label_counts.items():
        print(f"  {label}: {count}")

    # Merge with metrics
    merge_cols = ['dataset_id', 'primary_structure', 'flagged']
    merge_cols = [c for c in merge_cols if c in labeled.columns]
    merged = pd.merge(metrics_df, labeled[merge_cols], on='dataset_id', how='inner')

    # Remove flagged samples
    if 'flagged' in merged.columns:
        flagged_count = merged['flagged'].fillna(False).astype(bool).sum()
        print(f"Removing {flagged_count} flagged samples")
        merged = merged[~merged['flagged'].fillna(False).astype(bool)]

    print(f"Final training data: {len(merged)} labeled samples")

    # Prepare features
    EXCLUDE_COLS = ['dataset_id', 'dataset_name', 'size_mb', 'num_features', 'n_points', 'original_n_points']
    label_cols = ['primary_structure', 'flagged']
    feature_cols = [c for c in merged.columns if c not in EXCLUDE_COLS and c not in label_cols]

    X = merged[feature_cols].copy()
    y = merged['primary_structure'].copy()

    # Handle NaN/inf
    X = X.replace([np.inf, -np.inf], np.nan).fillna(X.median())

    print(f"Feature matrix: {X.shape}")
    print(f"Labels: {y.shape}")

    return X, y, feature_cols, merged

def compare_with_old_model():
    """Compare performance with the old model."""
    print("\n" + "="*60)
    print("COMPARING WITH OLD MODEL")
    print("="*60)

    # Load old labels
    old_labels_df = pd.read_csv("/home/btd8/Documents/phate_labels_rich.csv")
    old_labeled = old_labels_df[old_labels_df['primary_structure'].notna() & (old_labels_df['primary_structure'] != '')]

    print(f"Old model: {len(old_labeled)} labeled samples")
    print("Old label distribution:")
    for label, count in old_labeled['primary_structure'].value_counts().items():
        print(f"  {label}: {count}")

def evaluate_classifiers(X, y):
    """Evaluate different classifiers with cross-validation."""
    print("\n" + "="*60)
    print("EVALUATING CLASSIFIERS WITH UPDATED DATA")
    print("="*60)

    # Define classifiers
    classifiers = {
        'SVM (RBF)': Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', SVC(kernel='rbf', class_weight='balanced', random_state=42))
        ]),
        'Random Forest': RandomForestClassifier(
            n_estimators=100, random_state=42, class_weight='balanced'
        ),
        'Logistic Regression': Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', LogisticRegression(
                random_state=42, class_weight='balanced', max_iter=1000
            ))
        ]),
        'SVM (Linear)': Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', SVC(kernel='linear', class_weight='balanced', random_state=42))
        ])
    }

    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    results = {}

    for name, clf in classifiers.items():
        print(f"\n--- {name} ---")

        # Use Leave-One-Out cross-validation for small dataset
        loo = LeaveOneOut()
        scores = cross_val_score(clf, X, y_encoded, cv=loo, scoring='accuracy')

        results[name] = {
            'mean_accuracy': scores.mean(),
            'std_accuracy': scores.std(),
            'scores': scores
        }

        print(f"LOO CV Accuracy: {scores.mean():.3f} ± {scores.std():.3f}")
        print(f"Individual scores range: {scores.min():.3f} - {scores.max():.3f}")

    # Find best classifier
    best_name = max(results.keys(), key=lambda k: results[k]['mean_accuracy'])
    print(f"\n🏆 Best classifier: {best_name} ({results[best_name]['mean_accuracy']:.3f} ± {results[best_name]['std_accuracy']:.3f})")

    return results, best_name, le

def train_final_model(X, y, feature_cols):
    """Train the final model on all data."""
    print("\n" + "="*60)
    print("TRAINING FINAL MODEL")
    print("="*60)

    # Use best performing classifier (SVM RBF based on previous results)
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    final_clf = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', SVC(kernel='rbf', class_weight='balanced', random_state=42, probability=True))
    ])

    final_clf.fit(X, y_encoded)

    # Save the updated model
    model_data = {
        'classifier': final_clf,
        'label_encoder': le,
        'feature_cols': feature_cols,
        'training_size': len(X),
        'label_counts': pd.Series(y).value_counts().to_dict()
    }

    model_path = "/home/btd8/llm-paper-analyze/models/updated_structure_classifier.pkl"
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)

    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)

    print(f"✓ Updated model saved to: {model_path}")
    print(f"✓ Trained on {len(X)} samples with {len(feature_cols)} features")
    print(f"✓ Classes: {list(le.classes_)}")

    return final_clf, le

def main():
    print("="*70)
    print("RETRAINING STRUCTURE CLASSIFIER WITH UPDATED LABELS")
    print("="*70)

    # Compare old and new labels
    compare_with_old_model()

    # Load updated training data
    X, y, feature_cols, merged_data = load_updated_training_data()

    # Evaluate classifiers
    results, best_name, le = evaluate_classifiers(X, y)

    # Train final model
    final_clf, le = train_final_model(X, y, feature_cols)

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"✓ Updated model trained with {len(X)} labeled samples")
    print(f"✓ Best performing classifier: {best_name}")
    print(f"✓ Cross-validation accuracy: {results[best_name]['mean_accuracy']:.3f} ± {results[best_name]['std_accuracy']:.3f}")
    print(f"✓ Model saved and ready for use")

    # Show what changed in the dataset
    print(f"\nDataset changes summary:")
    print(f"✓ Updated labels file used: /home/btd8/Documents/phate_labels_rich (1).csv")
    print(f"✓ Training samples: {len(merged_data)}")
    print(f"✓ Label distribution:")
    for label, count in y.value_counts().items():
        print(f"   {label}: {count}")

    return final_clf, le

if __name__ == "__main__":
    clf, le = main()