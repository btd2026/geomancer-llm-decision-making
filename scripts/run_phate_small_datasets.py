#!/usr/bin/env python3
"""
Run PHATE on 100 small datasets and generate classification CSV.
Creates embeddings, plots, and a master CSV for manual classification.
"""

import scanpy as sc
import phate
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sqlite3
from pathlib import Path
import logging
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Paths
DB_PATH = Path("/home/btd8/llm-paper-analyze/data/manylatents_benchmark/manylatents_datasets.db")
DATA_DIR = Path("/nfs/roberts/project/pi_sk2433/shared/Geomancer_2025_Data/manylatents_small_datasets")
OUTPUT_DIR = Path("/nfs/roberts/project/pi_sk2433/shared/Geomancer_2025_Data/manylatents_small_datasets_phate")
PLOTS_DIR = OUTPUT_DIR / "plots"
EMBEDDINGS_DIR = OUTPUT_DIR / "embeddings"
CSV_OUTPUT = Path("/home/btd8/llm-paper-analyze/data/manylatents_benchmark/datasets_for_classification.csv")

# PHATE parameters
PHATE_PARAMS = {
    'n_components': 2,
    'knn': 100,
    't': 12,
    'decay': 60,
    'n_jobs': -1
}


def setup_directories():
    """Create output directories."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directories created at {OUTPUT_DIR}")


def get_datasets_from_db():
    """Get list of downloaded datasets from database."""
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("""
        SELECT dataset_id, dataset_name, collection_name, file_size_mb, n_cells
        FROM datasets
        WHERE downloaded = 1
        ORDER BY file_size_mb ASC
    """, conn)
    conn.close()
    return df


def run_phate_on_dataset(h5ad_path, dataset_id):
    """
    Run PHATE on a single dataset.
    Returns: (phate_coords, n_features, n_cells) or (None, None, None) on error
    """
    try:
        logger.info(f"Loading {h5ad_path.name}...")
        adata = sc.read_h5ad(h5ad_path)

        n_cells = adata.n_obs
        n_features = adata.n_vars

        logger.info(f"  Shape: {adata.shape} ({n_cells} cells, {n_features} genes)")

        # Prepare data - use .X (raw counts or normalized)
        if adata.X is None:
            logger.error(f"  No expression matrix found!")
            return None, None, None

        # Convert to dense if sparse
        if hasattr(adata.X, 'toarray'):
            X = adata.X.toarray()
        else:
            X = adata.X

        # Run PHATE
        logger.info(f"  Running PHATE (k={PHATE_PARAMS['knn']}, t={PHATE_PARAMS['t']})...")
        phate_op = phate.PHATE(
            n_components=PHATE_PARAMS['n_components'],
            knn=PHATE_PARAMS['knn'],
            t=PHATE_PARAMS['t'],
            decay=PHATE_PARAMS['decay'],
            n_jobs=PHATE_PARAMS['n_jobs'],
            verbose=0
        )

        phate_coords = phate_op.fit_transform(X)
        logger.info(f"  PHATE complete! Embedding shape: {phate_coords.shape}")

        return phate_coords, n_features, n_cells

    except Exception as e:
        logger.error(f"  Error processing {h5ad_path.name}: {e}")
        return None, None, None


def create_phate_plot(phate_coords, dataset_name, dataset_id, n_cells, n_features):
    """
    Create a simple, clean PHATE plot for manual classification.
    Returns: plot file path
    """
    plot_path = PLOTS_DIR / f"{dataset_id}.png"

    fig, ax = plt.subplots(figsize=(8, 8))

    # Simple scatter plot
    scatter = ax.scatter(
        phate_coords[:, 0],
        phate_coords[:, 1],
        s=3,
        alpha=0.5,
        c=phate_coords[:, 0],  # Color by PHATE1 coordinate
        cmap='viridis',
        rasterized=True
    )

    ax.set_xlabel('PHATE 1', fontsize=12)
    ax.set_ylabel('PHATE 2', fontsize=12)
    ax.set_title(f'{dataset_name}\n{n_cells:,} cells | {n_features:,} features',
                 fontsize=10, pad=10)
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()

    return plot_path


def save_embedding(phate_coords, dataset_id):
    """Save PHATE embedding to CSV."""
    embedding_path = EMBEDDINGS_DIR / f"{dataset_id}_phate.csv"
    df = pd.DataFrame(phate_coords, columns=['PHATE1', 'PHATE2'])
    df.to_csv(embedding_path, index=False)
    return embedding_path


def update_database(dataset_id, n_features, phate_plot_path):
    """Update database with n_features and phate_plot_path."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        UPDATE datasets
        SET n_features = ?,
            phate_plot_path = ?,
            benchmarked = 1
        WHERE dataset_id = ?
    """, (n_features, str(phate_plot_path), dataset_id))
    conn.commit()
    conn.close()


def process_all_datasets():
    """Process all datasets: run PHATE, create plots, update database."""
    logger.info("="*80)
    logger.info("PHATE Analysis on 100 Small Datasets")
    logger.info("="*80)

    # Get datasets
    datasets_df = get_datasets_from_db()
    total = len(datasets_df)
    logger.info(f"Found {total} datasets to process")

    results = []
    successful = 0
    failed = 0

    for idx, row in datasets_df.iterrows():
        dataset_id = row['dataset_id']
        dataset_name = row['dataset_name']

        logger.info(f"\n[{idx+1}/{total}] Processing: {dataset_name}")

        # Find H5AD file (it's named by version_id, not dataset_id)
        # We need to find it - they're named by version_id
        h5ad_files = list(DATA_DIR.glob(f"*.h5ad"))

        # We need to map dataset_id to the actual filename
        # The database has h5ad_url which contains the version_id
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT h5ad_url FROM datasets WHERE dataset_id = ?", (dataset_id,))
        url = cursor.fetchone()[0]
        conn.close()

        # Extract version_id from URL
        version_id = url.split('/')[-1].replace('.h5ad', '')
        h5ad_path = DATA_DIR / f"{version_id}.h5ad"

        if not h5ad_path.exists():
            logger.error(f"  File not found: {h5ad_path}")
            failed += 1
            continue

        # Run PHATE
        phate_coords, n_features, n_cells = run_phate_on_dataset(h5ad_path, dataset_id)

        if phate_coords is None:
            failed += 1
            continue

        # Create plot
        plot_path = create_phate_plot(phate_coords, dataset_name, dataset_id,
                                      n_cells, n_features)
        logger.info(f"  Plot saved: {plot_path}")

        # Save embedding
        embedding_path = save_embedding(phate_coords, dataset_id)
        logger.info(f"  Embedding saved: {embedding_path}")

        # Update database
        update_database(dataset_id, n_features, plot_path)

        results.append({
            'dataset_id': dataset_id,
            'n_features': n_features,
            'n_cells': n_cells,
            'plot_path': str(plot_path),
            'embedding_path': str(embedding_path)
        })

        successful += 1

        # Progress update
        if (idx + 1) % 10 == 0:
            logger.info(f"\nProgress: {idx+1}/{total} ({successful} successful, {failed} failed)")

    logger.info("\n" + "="*80)
    logger.info("PHATE ANALYSIS COMPLETE")
    logger.info("="*80)
    logger.info(f"Successful: {successful}/{total}")
    logger.info(f"Failed: {failed}/{total}")

    return results


def generate_classification_csv():
    """Generate final CSV for manual classification."""
    logger.info("\nGenerating classification CSV...")

    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("""
        SELECT
            dataset_id,
            dataset_name AS name,
            description,
            ROUND(file_size_mb, 2) AS size_mb,
            n_cells AS num_points,
            n_features AS num_features,
            phate_plot_path AS phate_plot_image,
            collection_name,
            collection_doi
        FROM datasets
        WHERE benchmarked = 1
        ORDER BY size_mb ASC
    """, conn)
    conn.close()

    # Add empty classification column
    df['manual_classification'] = ''
    df['notes'] = ''

    # Save CSV
    df.to_csv(CSV_OUTPUT, index=False)

    logger.info(f"Classification CSV saved: {CSV_OUTPUT}")
    logger.info(f"Rows: {len(df)}")
    logger.info(f"\nColumns: {', '.join(df.columns)}")

    return df


def main():
    """Main execution function."""
    start_time = datetime.now()

    # Setup
    setup_directories()

    # Process all datasets
    results = process_all_datasets()

    # Generate classification CSV
    classification_df = generate_classification_csv()

    # Summary
    elapsed = (datetime.now() - start_time).total_seconds() / 60

    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Total time: {elapsed:.1f} minutes")
    print(f"Datasets processed: {len(results)}")
    print(f"\nOutput locations:")
    print(f"  Plots: {PLOTS_DIR}")
    print(f"  Embeddings: {EMBEDDINGS_DIR}")
    print(f"  Classification CSV: {CSV_OUTPUT}")
    print(f"\nClassification CSV preview:")
    print(classification_df.head())
    print("="*80)


if __name__ == "__main__":
    main()
