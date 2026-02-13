#!/usr/bin/env python3
"""Train ML classifier to predict PHATE structure type from embedding metrics."""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import cross_val_score, StratifiedKFold, LeaveOneOut
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Paths
METRICS_PATH = Path("/home/btd8/llm-paper-analyze/data/manylatents_benchmark/embedding_metrics.csv")
LABELS_PATH = Path("/home/btd8/Documents/phate_labels_rich.csv")
OUTPUT_DIR = Path("/home/btd8/llm-paper-analyze/data/manylatents_benchmark/ml_results")

# Feature columns to use (exclude metadata)
EXCLUDE_COLS = ['dataset_id', 'dataset_name', 'size_mb', 'num_features', 'n_points']


def load_data():
    """Load and merge metrics with labels."""
    print("Loading data...")

    # Load metrics
    metrics_df = pd.read_csv(METRICS_PATH)
    print(f"  Metrics: {len(metrics_df)} datasets, {len(metrics_df.columns)} columns")

    # Load labels
    labels_df = pd.read_csv(LABELS_PATH)
    labeled = labels_df[labels_df['primary_structure'].notna() & (labels_df['primary_structure'] != '')]
    print(f"  Labels: {len(labeled)} datasets")

    # Merge
    merged = pd.merge(
        metrics_df,
        labeled[['dataset_id', 'primary_structure', 'density_pattern', 'branch_quality', 'overall_quality', 'n_components', 'n_clusters', 'flagged']],
        on='dataset_id',
        how='inner'
    )
    print(f"  Merged: {len(merged)} datasets")

    # Remove flagged datasets
    if 'flagged' in merged.columns:
        flagged_count = merged['flagged'].fillna(False).astype(bool).sum()
        merged = merged[~merged['flagged'].fillna(False).astype(bool)]
        print(f"  After removing flagged: {len(merged)} datasets ({flagged_count} removed)")

    return merged


def prepare_features(df):
    """Prepare feature matrix and labels."""
    # Get feature columns
    feature_cols = [c for c in df.columns if c not in EXCLUDE_COLS and c not in
                   ['primary_structure', 'density_pattern', 'branch_quality', 'overall_quality',
                    'n_components', 'n_clusters', 'flagged']]

    print(f"\nUsing {len(feature_cols)} features:")
    for i, col in enumerate(feature_cols):
        if i < 10:
            print(f"  - {col}")
    if len(feature_cols) > 10:
        print(f"  ... and {len(feature_cols) - 10} more")

    X = df[feature_cols].copy()
    y = df['primary_structure'].copy()

    # Handle NaN values
    nan_counts = X.isna().sum()
    cols_with_nan = nan_counts[nan_counts > 0]
    if len(cols_with_nan) > 0:
        print(f"\nFeatures with NaN values:")
        for col, count in cols_with_nan.items():
            print(f"  - {col}: {count} NaN")
        # Fill NaN with column median
        X = X.fillna(X.median())

    # Handle infinite values
    X = X.replace([np.inf, -np.inf], np.nan).fillna(X.median())

    return X, y, feature_cols


def train_and_evaluate(X, y, feature_cols):
    """Train multiple classifiers and evaluate with cross-validation."""

    print("\n" + "="*60)
    print("TRAINING AND EVALUATION")
    print("="*60)

    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    class_names = le.classes_

    print(f"\nClasses: {list(class_names)}")
    print(f"Class distribution:")
    for cls, count in zip(*np.unique(y_encoded, return_counts=True)):
        print(f"  {class_names[cls]}: {count}")

    # Define classifiers
    classifiers = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
        'SVM (RBF)': SVC(kernel='rbf', class_weight='balanced', random_state=42),
        'SVM (Linear)': SVC(kernel='linear', class_weight='balanced', random_state=42),
        'Logistic Regression': LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42),
        'KNN (k=3)': KNeighborsClassifier(n_neighbors=3),
        'KNN (k=5)': KNeighborsClassifier(n_neighbors=5),
    }

    results = {}

    # Use Leave-One-Out CV for small dataset
    print(f"\nUsing Leave-One-Out Cross-Validation (n={len(y)})")
    cv = LeaveOneOut()

    for name, clf in classifiers.items():
        # Create pipeline with scaling
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', clf)
        ])

        # Cross-validation
        scores = cross_val_score(pipeline, X, y_encoded, cv=cv, scoring='accuracy')

        results[name] = {
            'mean_accuracy': scores.mean(),
            'std_accuracy': scores.std(),
            'scores': scores
        }

        print(f"\n{name}:")
        print(f"  Accuracy: {scores.mean():.3f} (+/- {scores.std():.3f})")

    # Find best classifier
    best_clf_name = max(results, key=lambda x: results[x]['mean_accuracy'])
    print(f"\n{'='*60}")
    print(f"BEST CLASSIFIER: {best_clf_name}")
    print(f"Accuracy: {results[best_clf_name]['mean_accuracy']:.3f}")
    print(f"{'='*60}")

    # Train best classifier on full data and show feature importance
    if best_clf_name in ['Random Forest', 'Gradient Boosting']:
        print(f"\nFeature Importance ({best_clf_name}):")
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', classifiers[best_clf_name])
        ])
        pipeline.fit(X, y_encoded)

        importances = pipeline.named_steps['classifier'].feature_importances_
        importance_df = pd.DataFrame({
            'feature': feature_cols,
            'importance': importances
        }).sort_values('importance', ascending=False)

        print("\nTop 15 features:")
        for _, row in importance_df.head(15).iterrows():
            print(f"  {row['feature']}: {row['importance']:.4f}")

    return results, class_names


def train_simplified_classes(X, y):
    """Train with merged classes for better performance."""

    print("\n" + "="*60)
    print("SIMPLIFIED CLASSIFICATION (Merged Classes)")
    print("="*60)

    # Merge similar classes
    class_mapping = {
        'clusters': 'discrete',
        'diffuse': 'diffuse',
        'simple_trajectory': 'continuous',
        'horseshoe': 'continuous',
        'bifurcation': 'branching',
        'multi_branch': 'branching',
        'complex_tree': 'branching',
        'cyclic': 'continuous',
    }

    y_simplified = y.map(class_mapping)

    print(f"\nSimplified class distribution:")
    print(y_simplified.value_counts())

    # Encode
    le = LabelEncoder()
    y_encoded = le.fit_transform(y_simplified)

    # Train Random Forest
    clf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', clf)
    ])

    cv = LeaveOneOut()
    scores = cross_val_score(pipeline, X, y_encoded, cv=cv, scoring='accuracy')

    print(f"\nRandom Forest (Simplified):")
    print(f"  Accuracy: {scores.mean():.3f} (+/- {scores.std():.3f})")

    return scores.mean()


def main():
    # Load data
    df = load_data()

    # Prepare features
    X, y, feature_cols = prepare_features(df)

    print(f"\nFinal dataset shape: {X.shape}")
    print(f"Classes: {y.nunique()}")

    # Train and evaluate (original 7 classes)
    results, class_names = train_and_evaluate(X, y, feature_cols)

    # Train with simplified classes
    simplified_acc = train_simplified_classes(X, y)

    # Save results
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Save feature importance from Random Forest
    clf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    pipeline = Pipeline([('scaler', StandardScaler()), ('classifier', clf)])
    le = LabelEncoder()
    pipeline.fit(X, le.fit_transform(y))

    importance_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': pipeline.named_steps['classifier'].feature_importances_
    }).sort_values('importance', ascending=False)
    importance_df.to_csv(OUTPUT_DIR / 'feature_importance.csv', index=False)

    # Save summary
    summary = {
        'n_samples': len(y),
        'n_features': len(feature_cols),
        'n_classes': y.nunique(),
        'classes': list(y.unique()),
        'results': {name: {'accuracy': r['mean_accuracy']} for name, r in results.items()},
        'simplified_accuracy': simplified_acc
    }

    import json
    with open(OUTPUT_DIR / 'training_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
