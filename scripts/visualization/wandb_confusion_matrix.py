#!/usr/bin/env python3
"""
Generate confusion matrix using trained classifier and W&B run data.
Downloads embeddings from the geomancer-phate-deconcat-runs project and applies
the trained SVM model to generate confusion matrix visualizations.
"""
# Setup paths relative to script location
from pathlib import Path
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import joblib
import json
import wandb
from pathlib import Path
import os
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Configuration
ENTITY = "cesar-valdez-mcgill-university"
PROJECT = "geomancer-phate-deconcat-runs"  # The specific project mentioned
OUTPUT_DIR = Path(PROJECT_ROOT / "data" / "wandb_confusion_analysis")
WANDB_DATA_DIR = OUTPUT_DIR / "wandb_data"

# Model and label paths from your existing setup
MODEL_PATH = PROJECT_ROOT / "data" / "manylatents_benchmark" / "ml_results_91_labels" / "best_classifier_91_labels.pkl"
LABELS_MAIN = "/home/btd8/Documents/phate_labels_rich.csv"
LABELS_REMAINDER = "/home/btd8/Documents/phate_labels_rich_remainders.csv"

def setup_directories():
    """Create necessary output directories."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    WANDB_DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR}")

def load_trained_model():
    """Load the trained SVM classifier."""
    print(f"Loading trained model from: {MODEL_PATH}")
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")

    model_pipeline = joblib.load(MODEL_PATH)
    print(f"Loaded model: {type(model_pipeline)}")
    return model_pipeline

def load_labels():
    """Load and combine manual labels."""
    print("Loading manual labels...")

    labels_main = pd.read_csv(LABELS_MAIN)
    labels_remainder = pd.read_csv(LABELS_REMAINDER)

    # Filter to labeled entries
    labeled_main = labels_main[labels_main['primary_structure'].notna() & (labels_main['primary_structure'] != '')]
    labeled_remainder = labels_remainder[labels_remainder['primary_structure'].notna() & (labels_remainder['primary_structure'] != '')]

    # Combine
    all_labels = pd.concat([labeled_main, labeled_remainder], ignore_index=True)
    all_labels = all_labels.drop_duplicates(subset=['dataset_id'])

    # Remove flagged datasets
    if 'flagged' in all_labels.columns:
        all_labels = all_labels[~all_labels['flagged'].fillna(False).astype(bool)]

    print(f"Total labeled datasets: {len(all_labels)}")
    print(f"Class distribution: {all_labels['primary_structure'].value_counts().to_dict()}")

    return all_labels

def download_wandb_data():
    """Download run data and embeddings from the specified W&B project."""
    print(f"Connecting to W&B project: {ENTITY}/{PROJECT}")

    try:
        api = wandb.Api()
        runs = api.runs(f"{ENTITY}/{PROJECT}")
        print(f"Found {len(runs)} runs in the project")
    except Exception as e:
        print(f"Error connecting to W&B: {e}")
        print("Make sure you have wandb installed and are logged in: wandb login")
        return None, None

    run_data = []
    embeddings_data = []

    for i, run in enumerate(tqdm(runs, desc="Processing W&B runs")):
        try:
            # Extract run metadata
            config = dict(run.config) if run.config else {}
            summary = dict(run.summary) if run.summary else {}

            run_info = {
                'run_id': run.id,
                'name': run.name,
                'state': run.state,
                'config': config,
                'summary': summary
            }

            # Extract dataset information from run name
            # Expected format: phate_subdataset_<dataset>__<subset>__...__<label_key>
            name_parts = run.name.split('__')
            if len(name_parts) >= 1:
                first_part = name_parts[0]
                if first_part.startswith('phate_subdataset_'):
                    run_info['dataset_name'] = first_part.replace('phate_subdataset_', '')
                    # Try to extract just the dataset ID part
                    dataset_parts = run_info['dataset_name'].split('_')
                    if len(dataset_parts) > 1:
                        run_info['dataset_id'] = dataset_parts[-1]  # Last part should be dataset ID
                else:
                    run_info['dataset_name'] = first_part

            if len(name_parts) >= 2:
                run_info['label_key'] = name_parts[-1] if len(name_parts) > 1 else ''

            # Download artifacts/files that contain embeddings
            try:
                files = run.files()
                for f in files:
                    # Look for PHATE embedding files or metric files
                    if any(keyword in f.name.lower() for keyword in ['phate', 'embedding', 'metric', '.csv', '.json']):
                        print(f"  Found file: {f.name}")

                        # Download the file
                        local_path = WANDB_DATA_DIR / run.id / f.name
                        local_path.parent.mkdir(parents=True, exist_ok=True)

                        if not local_path.exists():
                            try:
                                f.download(root=str(local_path.parent), replace=True)
                                run_info['files'] = run_info.get('files', []) + [str(local_path)]
                            except Exception as e:
                                print(f"    Error downloading {f.name}: {e}")

                # Try to extract embedding coordinates from artifacts
                for artifact in run.logged_artifacts():
                    try:
                        artifact_dir = WANDB_DATA_DIR / run.id / "artifacts" / artifact.name
                        artifact_dir.mkdir(parents=True, exist_ok=True)
                        artifact.download(root=str(artifact_dir))
                        run_info['artifacts'] = run_info.get('artifacts', []) + [str(artifact_dir)]
                    except Exception as e:
                        print(f"    Error downloading artifact {artifact.name}: {e}")

            except Exception as e:
                print(f"  Error accessing files for run {run.id}: {e}")

            run_data.append(run_info)

        except Exception as e:
            print(f"Error processing run {i}: {e}")
            continue

        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{len(runs)} runs")

    # Save run metadata
    metadata_file = OUTPUT_DIR / "wandb_runs_metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(run_data, f, indent=2)

    print(f"Downloaded data for {len(run_data)} runs")
    print(f"Metadata saved to: {metadata_file}")

    return run_data, metadata_file

def extract_embeddings_from_downloads(run_data):
    """Extract PHATE embeddings and compute geometric features from downloaded files."""
    print("Extracting embeddings and computing geometric features...")

    # This is where we would compute geometric features from the downloaded PHATE embeddings
    # For now, let's see what data we actually have available

    embedding_features = []

    for run_info in tqdm(run_data, desc="Processing downloaded data"):
        try:
            # Look for embedding data in downloaded files
            files = run_info.get('files', [])
            artifacts = run_info.get('artifacts', [])

            dataset_id = run_info.get('dataset_id', '')
            if not dataset_id:
                # Try to extract from dataset_name
                dataset_name = run_info.get('dataset_name', '')
                if dataset_name:
                    # Extract the ID part (usually the last underscore-separated part)
                    parts = dataset_name.split('_')
                    dataset_id = parts[-1] if parts else dataset_name

            # Create a record for this run with basic info
            feature_record = {
                'dataset_id': dataset_id,
                'run_id': run_info['run_id'],
                'dataset_name': run_info.get('dataset_name', ''),
                'label_key': run_info.get('label_key', ''),
                'state': run_info.get('state', ''),
            }

            # TODO: Extract actual geometric features from PHATE embeddings
            # This would involve loading the embedding coordinates and computing:
            # - persistent_entropy_dim_1, persistent_entropy_dim_2
            # - mst_*, alpha_*, beta_*, gamma_*
            # - All 53 geometric features used by your classifier

            # For now, add placeholder to identify what we have
            feature_record['has_files'] = len(files) > 0
            feature_record['has_artifacts'] = len(artifacts) > 0
            feature_record['files_count'] = len(files)
            feature_record['artifacts_count'] = len(artifacts)

            embedding_features.append(feature_record)

        except Exception as e:
            print(f"Error processing run {run_info.get('run_id', 'unknown')}: {e}")
            continue

    embeddings_df = pd.DataFrame(embedding_features)
    print(f"Processed {len(embeddings_df)} run records")

    return embeddings_df

def apply_model_to_wandb_data(model, wandb_embeddings, labels_df):
    """Apply the trained model to W&B embedding data to generate predictions."""
    print("Applying trained model to W&B data...")

    # Merge with manual labels to get ground truth
    merged = pd.merge(
        wandb_embeddings,
        labels_df[['dataset_id', 'primary_structure']],
        on='dataset_id',
        how='inner'
    )

    print(f"Found {len(merged)} W&B runs with manual labels")

    if len(merged) == 0:
        print("Warning: No W&B runs match the manually labeled datasets!")
        print("\nW&B dataset IDs sample:")
        print(wandb_embeddings['dataset_id'].head().tolist())
        print("\nManual label dataset IDs sample:")
        print(labels_df['dataset_id'].head().tolist())
        return None, None, None

    # TODO: For now, we cannot actually apply the model because we need to:
    # 1. Extract PHATE embedding coordinates from the downloaded W&B files
    # 2. Compute the same 53 geometric features that the model was trained on
    # 3. Apply the model to these features

    print("Note: Full model application requires extracting embedding coordinates")
    print("and computing geometric features from the W&B PHATE embeddings.")
    print("This is a complex process that needs to match the exact feature computation")
    print("pipeline used during training.")

    return merged, None, None

def generate_confusion_matrix_from_wandb(wandb_data, labels_df, model):
    """Generate confusion matrix using W&B data and trained model."""
    print("\n" + "="*60)
    print("WANDB CONFUSION MATRIX ANALYSIS")
    print("="*60)

    # Try to apply the model
    merged, y_true, y_pred = apply_model_to_wandb_data(model, wandb_data, labels_df)

    if y_true is None or y_pred is None:
        print("Cannot generate confusion matrix - need to implement feature extraction")
        print("from PHATE embeddings first.")

        # Show what data we have
        print(f"\nAvailable W&B runs: {len(wandb_data)}")
        print(f"Runs with manual labels: {len(merged) if merged is not None else 0}")

        if merged is not None and len(merged) > 0:
            print(f"\nLabel distribution in W&B data:")
            print(merged['primary_structure'].value_counts().to_dict())

            # Save the matched data for reference
            matched_file = OUTPUT_DIR / "wandb_runs_with_labels.csv"
            merged.to_csv(matched_file, index=False)
            print(f"Saved matched runs to: {matched_file}")

        return

    # Generate confusion matrix (this part would work once we have predictions)
    classes = sorted(y_true.unique())
    cm = confusion_matrix(y_true, y_pred, labels=classes)

    # Plot confusion matrix
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Raw counts
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes, ax=ax1)
    ax1.set_title('W&B Runs - Confusion Matrix (Raw Counts)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Predicted Class', fontweight='bold')
    ax1.set_ylabel('True Class', fontweight='bold')

    # Percentages
    sns.heatmap(cm_percent, annot=True, fmt='.1f', cmap='Blues',
                xticklabels=classes, yticklabels=classes, ax=ax2)
    ax2.set_title('W&B Runs - Confusion Matrix (Percentages)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Predicted Class', fontweight='bold')
    ax2.set_ylabel('True Class', fontweight='bold')

    # Rotate labels
    for ax in [ax1, ax2]:
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    plt.tight_layout()

    # Save plot
    plot_path = OUTPUT_DIR / 'wandb_confusion_matrix.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.show()

    print(f"Confusion matrix saved to: {plot_path}")

    # Print accuracy
    accuracy = (y_true == y_pred).mean()
    print(f"W&B Runs Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")

def main():
    """Main pipeline for W&B confusion matrix analysis."""
    print("W&B Confusion Matrix Generator")
    print("="*50)

    # Setup
    setup_directories()

    # Load trained model and labels
    model = load_trained_model()
    labels_df = load_labels()

    # Check if we already have downloaded W&B data
    metadata_file = OUTPUT_DIR / "wandb_runs_metadata.json"

    if metadata_file.exists():
        print(f"Loading existing W&B data from: {metadata_file}")
        with open(metadata_file, 'r') as f:
            wandb_run_data = json.load(f)
    else:
        # Download W&B data
        wandb_run_data, _ = download_wandb_data()
        if wandb_run_data is None:
            return

    # Process the downloaded data
    wandb_embeddings = extract_embeddings_from_downloads(wandb_run_data)

    # Generate confusion matrix
    generate_confusion_matrix_from_wandb(wandb_embeddings, labels_df, model)

    print("\n" + "="*60)
    print("WANDB ANALYSIS COMPLETE")
    print("="*60)
    print(f"Results saved in: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()