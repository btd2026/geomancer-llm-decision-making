#!/usr/bin/env python3
"""
Analyze the trained classifier and generate confusion matrix visualization.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import joblib
import json
from pathlib import Path

# Paths
MODEL_PATH = "/home/btd8/llm-paper-analyze/data/manylatents_benchmark/ml_results_91_labels/best_classifier_91_labels.pkl"
METRICS_PATH = "/home/btd8/llm-paper-analyze/data/manylatents_benchmark/embedding_metrics.csv"
LABELS_MAIN = "/home/btd8/Documents/phate_labels_rich.csv"
LABELS_REMAINDER = "/home/btd8/Documents/phate_labels_rich_remainders.csv"
OUTPUT_DIR = Path("/home/btd8/llm-paper-analyze/data/manylatents_benchmark/ml_results_91_labels")

def load_data_and_model():
    """Load the trained model and prepare the same dataset used for training."""
    print("Loading trained model and data...")

    # Load the trained pipeline
    model_pipeline = joblib.load(MODEL_PATH)
    print(f"Loaded model: {type(model_pipeline)}")

    # Load metrics
    metrics_df = pd.read_csv(METRICS_PATH)
    print(f"Loaded metrics: {len(metrics_df)} datasets, {len(metrics_df.columns)} features")

    # Load and combine labels (same process as training)
    labels_main = pd.read_csv(LABELS_MAIN)
    labels_remainder = pd.read_csv(LABELS_REMAINDER)

    # Filter to labeled entries
    labeled_main = labels_main[labels_main['primary_structure'].notna() & (labels_main['primary_structure'] != '')]
    labeled_remainder = labels_remainder[labels_remainder['primary_structure'].notna() & (labels_remainder['primary_structure'] != '')]

    # Combine
    all_labels = pd.concat([labeled_main, labeled_remainder], ignore_index=True)
    all_labels = all_labels.drop_duplicates(subset=['dataset_id'])

    print(f"Combined labels: {len(all_labels)} datasets")

    # Merge with metrics (same as training)
    merged = pd.merge(
        metrics_df,
        all_labels[['dataset_id', 'primary_structure', 'flagged']],
        on='dataset_id',
        how='inner'
    )

    # Remove flagged datasets (same as training)
    if 'flagged' in merged.columns:
        merged = merged[~merged['flagged'].fillna(False).astype(bool)]

    print(f"Final dataset: {len(merged)} samples")
    print(f"Class distribution: {merged['primary_structure'].value_counts().to_dict()}")

    return model_pipeline, merged

def prepare_features(df):
    """Prepare features exactly as done during training."""
    # Exclude metadata columns (same as training script)
    exclude_cols = ['dataset_id', 'dataset_name', 'size_mb', 'num_features', 'n_points',
                   'primary_structure', 'flagged']

    feature_cols = [c for c in df.columns if c not in exclude_cols]
    print(f"Using {len(feature_cols)} features")

    X = df[feature_cols].copy()
    y = df['primary_structure'].copy()

    # Handle NaN values (same as training)
    nan_counts = X.isna().sum()
    cols_with_nan = nan_counts[nan_counts > 0]
    if len(cols_with_nan) > 0:
        print(f"Features with NaN: {len(cols_with_nan)}")
        X = X.fillna(X.median())

    return X, y, feature_cols

def generate_confusion_matrix(model, X, y_true):
    """Generate confusion matrix and classification report."""
    print("Generating predictions and confusion matrix...")

    # Get predictions
    y_pred = model.predict(X)

    # Get unique classes in order
    classes = sorted(y_true.unique())
    print(f"Classes: {classes}")

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=classes)

    # Classification report
    report = classification_report(y_true, y_pred, labels=classes, output_dict=True)

    return cm, report, classes, y_pred

def plot_confusion_matrix(cm, classes, save_path):
    """Create a detailed confusion matrix visualization."""
    # Calculate percentages
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: Raw counts
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes, ax=ax1)
    ax1.set_title('Confusion Matrix (Raw Counts)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Predicted Class', fontweight='bold')
    ax1.set_ylabel('True Class', fontweight='bold')

    # Plot 2: Percentages
    sns.heatmap(cm_percent, annot=True, fmt='.1f', cmap='Blues',
                xticklabels=classes, yticklabels=classes, ax=ax2)
    ax2.set_title('Confusion Matrix (Percentages)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Predicted Class', fontweight='bold')
    ax2.set_ylabel('True Class', fontweight='bold')

    # Rotate labels for better readability
    for ax in [ax1, ax2]:
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Confusion matrix saved to: {save_path}")

def analyze_predictions(y_true, y_pred, classes):
    """Analyze prediction patterns and errors."""
    print("\n" + "="*60)
    print("PREDICTION ANALYSIS")
    print("="*60)

    # Overall accuracy
    accuracy = (y_true == y_pred).mean()
    print(f"\nOverall Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")

    # Per-class analysis
    print(f"\nPER-CLASS PERFORMANCE:")
    print("-" * 40)

    for cls in classes:
        true_mask = y_true == cls
        pred_mask = y_pred == cls

        true_count = true_mask.sum()
        pred_count = pred_mask.sum()
        correct_count = (true_mask & pred_mask).sum()

        if true_count > 0:
            recall = correct_count / true_count
            precision = correct_count / pred_count if pred_count > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

            print(f"{cls:15} | N={true_count:2d} | Precision={precision:.3f} | Recall={recall:.3f} | F1={f1:.3f}")

    # Most confused pairs
    print(f"\nMOST FREQUENT MISCLASSIFICATIONS:")
    print("-" * 40)

    errors = pd.DataFrame({'true': y_true, 'pred': y_pred})
    errors = errors[errors['true'] != errors['pred']]

    if len(errors) > 0:
        confusion_pairs = errors.groupby(['true', 'pred']).size().sort_values(ascending=False)
        for (true_cls, pred_cls), count in confusion_pairs.head(10).items():
            print(f"{true_cls} → {pred_cls}: {count} errors")
    else:
        print("No misclassifications found!")

    return accuracy

def save_detailed_results(cm, report, classes, y_true, y_pred, output_dir):
    """Save detailed analysis results to files."""
    print(f"\nSaving detailed results to {output_dir}")

    # Save confusion matrix as CSV
    cm_df = pd.DataFrame(cm, index=classes, columns=classes)
    cm_df.to_csv(output_dir / 'confusion_matrix.csv')

    # Save classification report as JSON
    with open(output_dir / 'classification_report_detailed.json', 'w') as f:
        json.dump(report, f, indent=2)

    # Save predictions for analysis
    results_df = pd.DataFrame({
        'true_class': y_true,
        'predicted_class': y_pred,
        'correct': y_true == y_pred
    })
    results_df.to_csv(output_dir / 'prediction_results.csv', index=False)

    print("Results saved:")
    print(f"  - confusion_matrix.csv")
    print(f"  - classification_report_detailed.json")
    print(f"  - prediction_results.csv")

def main():
    """Main analysis pipeline."""
    print("="*60)
    print("CONFUSION MATRIX ANALYSIS")
    print("="*60)

    # Load model and data
    model_pipeline, merged_df = load_data_and_model()

    # Prepare features (same as training)
    X, y_true, feature_cols = prepare_features(merged_df)

    # Generate predictions and confusion matrix
    cm, report, classes, y_pred = generate_confusion_matrix(model_pipeline, X, y_true)

    # Create visualizations
    plot_path = OUTPUT_DIR / 'confusion_matrix_analysis.png'
    plot_confusion_matrix(cm, classes, plot_path)

    # Analyze predictions
    accuracy = analyze_predictions(y_true, y_pred, classes)

    # Save detailed results
    save_detailed_results(cm, report, classes, y_true, y_pred, OUTPUT_DIR)

    # Feature importance (if Random Forest or similar)
    try:
        if hasattr(model_pipeline.named_steps['classifier'], 'feature_importances_'):
            importances = model_pipeline.named_steps['classifier'].feature_importances_
            importance_df = pd.DataFrame({
                'feature': feature_cols,
                'importance': importances
            }).sort_values('importance', ascending=False)

            print(f"\nTOP 10 MOST IMPORTANT FEATURES:")
            print("-" * 40)
            for _, row in importance_df.head(10).iterrows():
                print(f"{row['feature']:25} | {row['importance']:.4f}")

            importance_df.to_csv(OUTPUT_DIR / 'feature_importance_detailed.csv', index=False)
        else:
            print(f"\nFeature importance not available for {type(model_pipeline.named_steps['classifier'])}")
    except Exception as e:
        print(f"Could not extract feature importance: {e}")

    print(f"\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)

if __name__ == "__main__":
    main()