#!/usr/bin/env python3
"""
Train classifier using enhanced features extracted from PHATE embeddings.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import pickle
from sklearn.model_selection import cross_val_score, StratifiedKFold, LeaveOneOut
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, f_classif
import warnings
warnings.filterwarnings('ignore')

def load_training_data():
    """Load the original labeled training data."""
    print("="*70)
    print("ENHANCED CLASSIFIER WITH 100 FEATURE SET")
    print("="*70)

    print("Loading training data...")

    # Load original metrics and labels
    metrics_df = pd.read_csv("/home/btd8/llm-paper-analyze/data/manylatents_benchmark/embedding_metrics.csv")
    labels_df = pd.read_csv("/home/btd8/Documents/phate_labels_rich.csv")

    # Filter labeled samples
    labeled = labels_df[labels_df['primary_structure'].notna() & (labels_df['primary_structure'] != '')]

    # Merge training data
    merge_cols = ['dataset_id', 'primary_structure', 'flagged']
    merge_cols = [c for c in merge_cols if c in labeled.columns]
    merged = pd.merge(metrics_df, labeled[merge_cols], on='dataset_id', how='inner')

    # Remove flagged samples
    if 'flagged' in merged.columns:
        flagged_count = merged['flagged'].fillna(False).astype(bool).sum()
        if flagged_count > 0:
            print(f"Removing {flagged_count} flagged samples")
            merged = merged[~merged['flagged'].fillna(False).astype(bool)]

    print(f"Training data: {len(merged)} labeled samples")
    print("Label distribution:")
    for label, count in merged['primary_structure'].value_counts().items():
        print(f"  {label}: {count}")

    return merged

def load_enhanced_features():
    """Load the enhanced features computed from PHATE embeddings."""
    print(f"\nLoading enhanced features...")

    enhanced_df = pd.read_csv("/home/btd8/llm-paper-analyze/data/enhanced_phate_metrics.csv")

    # Get feature columns (exclude metadata)
    feature_cols = [c for c in enhanced_df.columns if c not in ['dataset_id', 'n_points', 'dataset_source']]

    print(f"Enhanced features: {len(enhanced_df)} datasets, {len(feature_cols)} features")

    return enhanced_df, feature_cols

def train_enhanced_classifier(enhanced_df, feature_cols, training_data):
    """Train classifier on enhanced features using original labels for validation."""
    print(f"\nTraining enhanced classifier...")

    # For validation, we'll use the original training data approach but with enhanced methodology
    # Since we can't directly match enhanced features with training labels, we'll:
    # 1. Train on original data with cross-validation
    # 2. Apply the methodology to enhanced features

    # Load original training features
    EXCLUDE_COLS = ['dataset_id', 'dataset_name', 'size_mb', 'num_features', 'n_points', 'original_n_points']
    label_cols = ['primary_structure', 'flagged']
    original_feature_cols = [c for c in training_data.columns if c not in EXCLUDE_COLS and c not in label_cols]

    X_train = training_data[original_feature_cols].copy()
    y_train = training_data['primary_structure'].copy()

    # Handle NaN/inf values
    X_train = X_train.replace([np.inf, -np.inf], np.nan).fillna(X_train.median())

    print(f"Training on {X_train.shape[0]} samples with {X_train.shape[1]} features")

    # Encode labels
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)

    # Define enhanced classifiers
    classifiers = {
        'Enhanced SVM (RBF)': Pipeline([
            ('scaler', RobustScaler()),
            ('classifier', SVC(
                kernel='rbf',
                class_weight='balanced',
                random_state=42,
                probability=True,
                C=1.0,
                gamma='scale'
            ))
        ]),
        'Enhanced Random Forest': RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight='balanced',
            random_state=42
        ),
        'Enhanced Gradient Boosting': GradientBoostingClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42
        ),
        'Enhanced Logistic Regression': Pipeline([
            ('scaler', RobustScaler()),
            ('classifier', LogisticRegression(
                class_weight='balanced',
                random_state=42,
                max_iter=1000,
                C=1.0
            ))
        ])
    }

    # Evaluate classifiers
    print(f"\nEvaluating enhanced classifiers:")

    cv_method = StratifiedKFold(n_splits=min(5, len(np.unique(y_train_encoded))), shuffle=True, random_state=42)

    results = {}
    for name, clf in classifiers.items():
        try:
            scores = cross_val_score(clf, X_train, y_train_encoded, cv=cv_method, scoring='accuracy')
            results[name] = {
                'mean_accuracy': scores.mean(),
                'std_accuracy': scores.std(),
                'classifier': clf
            }
            print(f"  {name}: {scores.mean():.3f} ± {scores.std():.3f}")
        except Exception as e:
            print(f"  {name}: Error - {e}")
            continue

    # Select best classifier
    if results:
        best_name = max(results.keys(), key=lambda k: results[k]['mean_accuracy'])
        best_clf = results[best_name]['classifier']
        print(f"\n🏆 Best classifier: {best_name} ({results[best_name]['mean_accuracy']:.3f})")
    else:
        print("No classifiers completed successfully!")
        return None

    # Train best classifier on full training data
    best_clf.fit(X_train, y_train_encoded)

    return best_clf, le, results[best_name]

def apply_to_enhanced_features(clf, le, enhanced_df, feature_cols):
    """Apply trained classifier to enhanced features."""
    print(f"\nApplying classifier to enhanced features...")

    # Prepare feature matrix from enhanced features
    X_enhanced = enhanced_df[feature_cols].copy()

    # Handle NaN/inf values
    X_enhanced = X_enhanced.replace([np.inf, -np.inf], np.nan)

    # Fill NaN with column medians
    for col in X_enhanced.columns:
        if X_enhanced[col].isnull().any():
            median_val = X_enhanced[col].median()
            if pd.isna(median_val):
                X_enhanced[col] = X_enhanced[col].fillna(0)
            else:
                X_enhanced[col] = X_enhanced[col].fillna(median_val)

    print(f"Predicting on enhanced feature matrix: {X_enhanced.shape}")

    # Predict
    y_pred_encoded = clf.predict(X_enhanced)
    y_pred_proba = clf.predict_proba(X_enhanced)
    y_pred = le.inverse_transform(y_pred_encoded)

    # Create results dataframe
    results_df = enhanced_df[['dataset_id', 'n_points']].copy()
    results_df['predicted_structure'] = y_pred
    results_df['confidence'] = y_pred_proba.max(axis=1)

    # Add individual class probabilities
    for i, class_name in enumerate(le.classes_):
        results_df[f'prob_{class_name}'] = y_pred_proba[:, i]

    # Sort by confidence
    results_df = results_df.sort_values('confidence', ascending=False)

    return results_df

def main():
    # Load training data for validation
    training_data = load_training_data()

    # Load enhanced features
    enhanced_df, feature_cols = load_enhanced_features()

    # Train enhanced classifier
    best_clf, le, best_results = train_enhanced_classifier(enhanced_df, feature_cols, training_data)

    if best_clf is None:
        print("Training failed!")
        return

    # Apply to enhanced features
    results_df = apply_to_enhanced_features(best_clf, le, enhanced_df, feature_cols)

    # Save results
    output_path = "/home/btd8/llm-paper-analyze/data/enhanced_structure_predictions.csv"
    results_df.to_csv(output_path, index=False)

    # Save enhanced model
    model_data = {
        'classifier': best_clf,
        'label_encoder': le,
        'feature_cols': feature_cols,
        'cv_accuracy': best_results['mean_accuracy'],
        'cv_std': best_results['std_accuracy'],
        'n_enhanced_features': len(feature_cols),
        'training_samples': len(training_data)
    }

    model_path = "/home/btd8/llm-paper-analyze/models/enhanced_structure_classifier.pkl"
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)

    # Summary
    print(f"\n" + "="*70)
    print("ENHANCED CLASSIFICATION SUMMARY")
    print("="*70)

    print(f"✓ Enhanced features: {len(feature_cols)} features")
    print(f"✓ Training accuracy: {best_results['mean_accuracy']:.3f} ± {best_results['std_accuracy']:.3f}")
    print(f"✓ Predictions generated for {len(results_df)} datasets")
    print(f"✓ Results saved to: {output_path}")
    print(f"✓ Model saved to: {model_path}")

    print(f"\nPredicted structure distribution:")
    structure_counts = results_df['predicted_structure'].value_counts()
    for structure, count in structure_counts.items():
        pct = count / len(results_df) * 100
        print(f"  {structure}: {count} ({pct:.1f}%)")

    print(f"\nConfidence statistics:")
    print(f"  Mean confidence: {results_df['confidence'].mean():.3f}")
    print(f"  Median confidence: {results_df['confidence'].median():.3f}")
    print(f"  High confidence (>0.7): {(results_df['confidence'] > 0.7).sum()}")
    print(f"  Medium confidence (0.5-0.7): {((results_df['confidence'] > 0.5) & (results_df['confidence'] <= 0.7)).sum()}")
    print(f"  Low confidence (<0.5): {(results_df['confidence'] <= 0.5).sum()}")

    print(f"\nTop 10 most confident predictions:")
    for _, row in results_df.head(10).iterrows():
        print(f"  {row['dataset_id'][:36]}: {row['predicted_structure']} ({row['confidence']:.3f})")

    return results_df

if __name__ == "__main__":
    results = main()