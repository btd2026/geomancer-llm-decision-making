#!/usr/bin/env python3
"""
Train ML classifier using ALL 91 manually labeled PHATE embeddings.
Combines both label files and trains on the complete dataset.
Memory-optimized for SLURM execution.
"""
# Setup paths relative to script location
from pathlib import Path
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent


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
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings
import joblib
import json
from datetime import datetime
import gc
warnings.filterwarnings('ignore')

# Paths
METRICS_PATH = Path(PROJECT_ROOT / "data" / "manylatents_benchmark" / "embedding_metrics.csv")
LABELS_MAIN = Path("/home/btd8/Documents/phate_labels_rich.csv")
LABELS_REMAINDER = Path("/home/btd8/Documents/phate_labels_rich_remainders.csv")
OUTPUT_DIR = Path(PROJECT_ROOT / "data" / "manylatents_benchmark" / "ml_results_91_labels")

# Feature columns to exclude (metadata)
EXCLUDE_COLS = ['dataset_id', 'dataset_name', 'size_mb', 'num_features', 'n_points']


def load_all_labels():
    """Load and combine both label files."""
    print("Loading all manual labels...")

    # Load both label files
    labels_main = pd.read_csv(LABELS_MAIN)
    labels_remainder = pd.read_csv(LABELS_REMAINDER)

    print(f"  Main labels: {len(labels_main)} datasets")
    print(f"  Remainder labels: {len(labels_remainder)} datasets")

    # Filter to only labeled entries
    labeled_main = labels_main[labels_main['primary_structure'].notna() & (labels_main['primary_structure'] != '')]
    labeled_remainder = labels_remainder[labels_remainder['primary_structure'].notna() & (labels_remainder['primary_structure'] != '')]

    print(f"  Labeled main: {len(labeled_main)} datasets")
    print(f"  Labeled remainder: {len(labeled_remainder)} datasets")

    # Combine both datasets
    all_labels = pd.concat([labeled_main, labeled_remainder], ignore_index=True)

    # Remove any duplicates (just in case)
    initial_count = len(all_labels)
    all_labels = all_labels.drop_duplicates(subset=['dataset_id'])
    final_count = len(all_labels)

    if initial_count != final_count:
        print(f"  Removed {initial_count - final_count} duplicate dataset IDs")

    print(f"  Total combined labels: {len(all_labels)} datasets")

    # Show label distribution
    label_counts = all_labels['primary_structure'].value_counts()
    print(f"\n  Label distribution:")
    for label, count in label_counts.items():
        print(f"    {label}: {count}")

    return all_labels


def load_data():
    """Load and merge metrics with all labels."""
    print("Loading data...")

    # Load metrics
    metrics_df = pd.read_csv(METRICS_PATH)
    print(f"  Metrics: {len(metrics_df)} datasets, {len(metrics_df.columns)} columns")

    # Load all labels
    all_labels = load_all_labels()

    # Merge
    merged = pd.merge(
        metrics_df,
        all_labels[['dataset_id', 'primary_structure', 'density_pattern', 'branch_quality',
                   'overall_quality', 'n_components', 'n_clusters', 'flagged', 'notes']],
        on='dataset_id',
        how='inner'
    )
    print(f"  After merge: {len(merged)} datasets")

    # Remove flagged datasets (spatial transcriptomics, single cell types, etc.)
    if 'flagged' in merged.columns:
        initial_count = len(merged)
        merged = merged[~merged['flagged'].fillna(False).astype(bool)]
        flagged_count = initial_count - len(merged)
        print(f"  After removing flagged: {len(merged)} datasets ({flagged_count} removed)")

    # Show final distribution
    final_counts = merged['primary_structure'].value_counts()
    print(f"\n  Final training distribution:")
    for label, count in final_counts.items():
        print(f"    {label}: {count}")

    return merged


def prepare_features(df):
    """Prepare feature matrix and labels."""
    # Get feature columns
    feature_cols = [c for c in df.columns if c not in EXCLUDE_COLS and c not in
                   ['primary_structure', 'density_pattern', 'branch_quality', 'overall_quality',
                    'n_components', 'n_clusters', 'flagged', 'notes']]

    print(f"\nUsing {len(feature_cols)} features:")
    for i, col in enumerate(feature_cols[:10]):
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
            print(f"  {col}: {count}")
        X = X.fillna(X.median())
        print("  Filled NaN with median values")

    print(f"\nFinal dataset shape: {X.shape}")
    print(f"Target classes: {sorted(y.unique())}")

    return X, y, feature_cols


def train_classifiers(X, y):
    """Train multiple classifiers and return results."""
    print(f"\nTraining classifiers on {len(X)} samples...")

    # Define classifiers
    classifiers = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=2),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
        'SVM (RBF)': SVC(kernel='rbf', random_state=42),
        'SVM (Linear)': SVC(kernel='linear', random_state=42),
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'KNN (k=3)': KNeighborsClassifier(n_neighbors=3),
        'KNN (k=5)': KNeighborsClassifier(n_neighbors=5)
    }

    results = {}
    best_model = None
    best_score = 0

    # Use Leave-One-Out CV for small dataset
    cv = LeaveOneOut()

    for name, clf in classifiers.items():
        print(f"\n  Training {name}...")

        # Create pipeline with scaling
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', clf)
        ])

        # Perform cross-validation
        scores = cross_val_score(pipeline, X, y, cv=cv, scoring='accuracy', n_jobs=1)
        accuracy = np.mean(scores)

        results[name] = {
            'accuracy': accuracy,
            'std': np.std(scores),
            'scores': scores.tolist()
        }

        print(f"    Accuracy: {accuracy:.4f} ± {np.std(scores):.4f}")

        if accuracy > best_score:
            best_score = accuracy
            best_model = (name, pipeline)

        # Clear memory
        gc.collect()

    return results, best_model


def save_results(results, best_model, X, y, feature_cols, merged_df):
    """Save training results and best model."""
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

    # Save training summary
    summary = {
        'timestamp': datetime.now().isoformat(),
        'n_samples': len(X),
        'n_features': len(feature_cols),
        'n_classes': len(y.unique()),
        'classes': sorted(y.unique()),
        'label_distribution': y.value_counts().to_dict(),
        'best_model': best_model[0],
        'best_accuracy': results[best_model[0]]['accuracy'],
        'results': results
    }

    with open(OUTPUT_DIR / 'training_summary_91_labels.json', 'w') as f:
        json.dump(summary, f, indent=2)

    # Save feature importance if available
    best_pipeline = best_model[1]
    best_pipeline.fit(X, y)

    if hasattr(best_pipeline.named_steps['classifier'], 'feature_importances_'):
        importance_df = pd.DataFrame({
            'feature': feature_cols,
            'importance': best_pipeline.named_steps['classifier'].feature_importances_
        }).sort_values('importance', ascending=False)

        importance_df.to_csv(OUTPUT_DIR / 'feature_importance_91_labels.csv', index=False)
        print(f"\nTop 10 most important features:")
        for _, row in importance_df.head(10).iterrows():
            print(f"  {row['feature']}: {row['importance']:.4f}")

    # Save the best model
    joblib.dump(best_pipeline, OUTPUT_DIR / f'best_classifier_91_labels.pkl')

    # Save detailed classification report
    y_pred = best_pipeline.predict(X)
    report = classification_report(y, y_pred, output_dict=True)

    with open(OUTPUT_DIR / 'classification_report_91_labels.json', 'w') as f:
        json.dump(report, f, indent=2)

    # Save dataset info
    dataset_info = merged_df[['dataset_id', 'dataset_name', 'primary_structure', 'flagged', 'notes']].copy()
    dataset_info.to_csv(OUTPUT_DIR / 'training_datasets_91_labels.csv', index=False)

    print(f"\nResults saved to {OUTPUT_DIR}")
    print(f"Best model: {best_model[0]} (accuracy: {results[best_model[0]]['accuracy']:.4f})")


def main():
    """Main training pipeline."""
    print("=== TRAINING CLASSIFIER WITH ALL 91 LABELS ===")
    print(f"Started at: {datetime.now()}")

    # Load and prepare data
    merged_df = load_data()
    X, y, feature_cols = prepare_features(merged_df)

    # Train classifiers
    results, best_model = train_classifiers(X, y)

    # Save results
    save_results(results, best_model, X, y, feature_cols, merged_df)

    print(f"\nCompleted at: {datetime.now()}")
    print("=== TRAINING COMPLETE ===")


if __name__ == "__main__":
    main()