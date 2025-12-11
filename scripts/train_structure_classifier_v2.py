#!/usr/bin/env python3
"""
Enhanced structure classifier training with improved features and methodology.
Uses the enhanced features extracted from PHATE embeddings.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import pickle
from sklearn.model_selection import LeaveOneOut, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.feature_selection import SelectKBest, f_classif, RFE
import warnings
warnings.filterwarnings('ignore')

# Paths (relative to script location)
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
METRICS_PATH = PROJECT_ROOT / "data" / "manylatents_benchmark" / "manylatents_metrics.csv"
OLD_METRICS_PATH = PROJECT_ROOT / "data" / "manylatents_benchmark" / "embedding_metrics.csv"
LABELS_PATH = PROJECT_ROOT / "data" / "phate_labels_rich.csv"  # Note: moved from Documents to project data
OUTPUT_DIR = PROJECT_ROOT / "data" / "manylatents_benchmark" / "ml_results_v2"

# Metadata columns to exclude from features
EXCLUDE_COLS = ['dataset_id', 'dataset_name', 'size_mb', 'num_features', 'n_points', 'original_n_points']


def load_data():
    """Load and merge metrics with labels."""
    print("Loading data...")

    # Try to load manylatents metrics first, fall back to old metrics
    if METRICS_PATH.exists():
        metrics_df = pd.read_csv(METRICS_PATH)
        print(f"  Manylatents metrics: {len(metrics_df)} datasets, {len(metrics_df.columns)} columns")
    elif OLD_METRICS_PATH.exists():
        metrics_df = pd.read_csv(OLD_METRICS_PATH)
        print(f"  Old metrics: {len(metrics_df)} datasets, {len(metrics_df.columns)} columns")
    else:
        raise FileNotFoundError("No metrics file found")

    # Load labels
    labels_df = pd.read_csv(LABELS_PATH)
    labeled = labels_df[labels_df['primary_structure'].notna() & (labels_df['primary_structure'] != '')]
    print(f"  Labels: {len(labeled)} datasets")

    # Merge
    merge_cols = ['dataset_id', 'primary_structure', 'density_pattern', 'branch_quality',
                  'overall_quality', 'n_components', 'n_clusters', 'flagged']
    merge_cols = [c for c in merge_cols if c in labeled.columns]

    merged = pd.merge(
        metrics_df,
        labeled[merge_cols],
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
    # Get feature columns (exclude metadata and label columns)
    label_cols = ['primary_structure', 'density_pattern', 'branch_quality', 'overall_quality',
                  'n_components', 'n_clusters', 'flagged']
    feature_cols = [c for c in df.columns if c not in EXCLUDE_COLS and c not in label_cols]

    print(f"\nUsing {len(feature_cols)} features:")

    # Group features by type
    manylatents_features = [c for c in feature_cols if c.startswith(('fractal', 'participation', 'persistent', 'reeb', 'aniso'))]
    geometric_features = [c for c in feature_cols if c not in manylatents_features]

    print(f"\n  ManyLatents metrics ({len(manylatents_features)}):")
    for col in manylatents_features[:10]:
        print(f"    - {col}")
    if len(manylatents_features) > 10:
        print(f"    ... and {len(manylatents_features) - 10} more")

    print(f"\n  Geometric metrics ({len(geometric_features)}):")
    for col in geometric_features[:10]:
        print(f"    - {col}")
    if len(geometric_features) > 10:
        print(f"    ... and {len(geometric_features) - 10} more")

    X = df[feature_cols].copy()
    y = df['primary_structure'].copy()

    # Handle NaN values
    nan_counts = X.isna().sum()
    cols_with_nan = nan_counts[nan_counts > 0]
    if len(cols_with_nan) > 0:
        print(f"\nFeatures with NaN values ({len(cols_with_nan)}):")
        for col, count in list(cols_with_nan.items())[:5]:
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
    n_samples = len(y)
    if n_samples < 50:
        print(f"\nUsing Leave-One-Out Cross-Validation (n={n_samples})")
        cv = LeaveOneOut()
    else:
        print(f"\nUsing 5-Fold Stratified Cross-Validation (n={n_samples})")
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

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

        print("\nTop 20 features:")
        for _, row in importance_df.head(20).iterrows():
            # Mark manylatents features
            marker = "[ML]" if row['feature'].startswith(('fractal', 'participation', 'persistent', 'reeb', 'aniso')) else ""
            print(f"  {row['feature']}: {row['importance']:.4f} {marker}")

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

    cv = LeaveOneOut() if len(y) < 50 else StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(pipeline, X, y_encoded, cv=cv, scoring='accuracy')

    print(f"\nRandom Forest (Simplified 4-class):")
    print(f"  Accuracy: {scores.mean():.3f} (+/- {scores.std():.3f})")

    return scores.mean()


def analyze_manylatents_features(X, y, feature_cols):
    """Analyze how well manylatents-specific features perform."""

    print("\n" + "="*60)
    print("MANYLATENTS FEATURES ANALYSIS")
    print("="*60)

    # Separate manylatents features from geometric features
    ml_features = [c for c in feature_cols if c.startswith(('fractal', 'participation', 'persistent', 'reeb', 'aniso'))]
    geo_features = [c for c in feature_cols if c not in ml_features]

    if len(ml_features) == 0:
        print("\nNo manylatents-specific features found.")
        return

    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    cv = LeaveOneOut() if len(y) < 50 else StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Test with only manylatents features
    if len(ml_features) > 0:
        X_ml = X[ml_features].copy()
        clf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
        pipeline = Pipeline([('scaler', StandardScaler()), ('classifier', clf)])
        scores_ml = cross_val_score(pipeline, X_ml, y_encoded, cv=cv, scoring='accuracy')
        print(f"\nManyLatents features only ({len(ml_features)} features):")
        print(f"  Accuracy: {scores_ml.mean():.3f} (+/- {scores_ml.std():.3f})")

    # Test with only geometric features
    if len(geo_features) > 0:
        X_geo = X[geo_features].copy()
        clf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
        pipeline = Pipeline([('scaler', StandardScaler()), ('classifier', clf)])
        scores_geo = cross_val_score(pipeline, X_geo, y_encoded, cv=cv, scoring='accuracy')
        print(f"\nGeometric features only ({len(geo_features)} features):")
        print(f"  Accuracy: {scores_geo.mean():.3f} (+/- {scores_geo.std():.3f})")

    # Test with all features
    clf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    pipeline = Pipeline([('scaler', StandardScaler()), ('classifier', clf)])
    scores_all = cross_val_score(pipeline, X, y_encoded, cv=cv, scoring='accuracy')
    print(f"\nAll features combined ({len(feature_cols)} features):")
    print(f"  Accuracy: {scores_all.mean():.3f} (+/- {scores_all.std():.3f})")


def main():
    # Load data
    df = load_data()

    # Prepare features
    X, y, feature_cols = prepare_features(df)

    print(f"\nFinal dataset shape: {X.shape}")
    print(f"Classes: {y.nunique()}")

    # Train and evaluate (original 7 classes)
    results, class_names = train_and_evaluate(X, y, feature_cols)

    # Analyze manylatents features specifically
    analyze_manylatents_features(X, y, feature_cols)

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
    importance_df.to_csv(OUTPUT_DIR / 'feature_importance_v2.csv', index=False)

    # Save summary
    summary = {
        'n_samples': len(y),
        'n_features': len(feature_cols),
        'n_classes': y.nunique(),
        'classes': list(y.unique()),
        'manylatents_features': [c for c in feature_cols if c.startswith(('fractal', 'participation', 'persistent', 'reeb', 'aniso'))],
        'results': {name: {'accuracy': r['mean_accuracy']} for name, r in results.items()},
        'simplified_accuracy': simplified_acc
    }

    with open(OUTPUT_DIR / 'training_summary_v2.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
