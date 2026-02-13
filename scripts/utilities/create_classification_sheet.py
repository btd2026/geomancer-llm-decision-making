#!/usr/bin/env python3
"""
Create classification CSV with dataset metadata.
Extracts n_features from H5AD files and prepares sheet for manual classification.
"""

import scanpy as sc
import pandas as pd
import sqlite3
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# Paths
DB_PATH = Path("/home/btd8/llm-paper-analyze/data/manylatents_benchmark/manylatents_datasets.db")
DATA_DIR = Path("/nfs/roberts/project/pi_sk2433/shared/Geomancer_2025_Data/manylatents_small_datasets")
CSV_OUTPUT = Path("/home/btd8/llm-paper-analyze/data/manylatents_benchmark/datasets_for_classification.csv")

def extract_features_from_h5ad(h5ad_path):
    """Extract number of features and cells from H5AD file."""
    try:
        adata = sc.read_h5ad(h5ad_path)
        n_vars = adata.n_vars
        n_obs = adata.n_obs
        # Explicitly free memory
        del adata
        import gc
        gc.collect()
        return n_vars, n_obs
    except Exception as e:
        logger.error(f"Error reading {h5ad_path.name}: {e}")
        return None, None

def update_database_features():
    """Extract features from all H5AD files and update database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Get all datasets that need feature extraction
    cursor.execute("""
        SELECT dataset_id, h5ad_url, n_features
        FROM datasets
        WHERE downloaded = 1
    """)
    datasets = cursor.fetchall()

    logger.info(f"Processing {len(datasets)} datasets to extract feature counts...")

    for idx, (dataset_id, h5ad_url, existing_features) in enumerate(datasets, 1):
        # Skip if already extracted
        if existing_features is not None:
            logger.info(f"[{idx}/{len(datasets)}] Already extracted: {dataset_id}")
            continue
        # Extract version_id from URL
        version_id = h5ad_url.split('/')[-1].replace('.h5ad', '')
        h5ad_path = DATA_DIR / f"{version_id}.h5ad"

        if not h5ad_path.exists():
            logger.warning(f"[{idx}/{len(datasets)}] File not found: {version_id}")
            continue

        logger.info(f"[{idx}/{len(datasets)}] Processing {version_id}...")
        n_features, n_cells = extract_features_from_h5ad(h5ad_path)

        if n_features is not None:
            cursor.execute("""
                UPDATE datasets
                SET n_features = ?, n_cells = ?
                WHERE dataset_id = ?
            """, (n_features, n_cells, dataset_id))
            logger.info(f"  ✓ {n_cells:,} cells × {n_features:,} features")

    conn.commit()
    conn.close()
    logger.info("Database updated with feature counts")

def generate_classification_csv():
    """Generate the classification CSV."""
    conn = sqlite3.connect(DB_PATH)

    df = pd.read_sql_query("""
        SELECT
            dataset_id,
            dataset_name AS name,
            description,
            collection_name,
            collection_doi,
            ROUND(file_size_mb, 2) AS size_mb,
            n_cells AS num_points,
            n_features AS num_features,
            phate_plot_path AS phate_plot_image
        FROM datasets
        WHERE downloaded = 1
        ORDER BY size_mb ASC
    """, conn)
    conn.close()

    # Add classification columns
    df['manual_classification'] = ''
    df['notes'] = ''

    # Reorder columns for better readability
    column_order = [
        'dataset_id',
        'name',
        'description',
        'collection_name',
        'size_mb',
        'num_points',
        'num_features',
        'phate_plot_image',
        'manual_classification',
        'notes',
        'collection_doi'
    ]

    df = df[column_order]

    # Save CSV
    CSV_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CSV_OUTPUT, index=False)

    logger.info(f"\n{'='*80}")
    logger.info("CLASSIFICATION CSV GENERATED")
    logger.info(f"{'='*80}")
    logger.info(f"Location: {CSV_OUTPUT}")
    logger.info(f"Datasets: {len(df)}")
    logger.info(f"\nColumns: {', '.join(df.columns)}")
    logger.info(f"\nPreview:")
    print(df.head(10).to_string())
    logger.info(f"\n{'='*80}")

    return df

def main():
    logger.info("="*80)
    logger.info("Creating Classification Sheet for Manual Review")
    logger.info("="*80)

    # Step 1: Extract features from H5AD files
    update_database_features()

    # Step 2: Generate CSV
    df = generate_classification_csv()

    # Summary statistics
    logger.info("\nSummary Statistics:")
    logger.info(f"  Total datasets: {len(df)}")
    logger.info(f"  Total cells: {df['num_points'].sum():,}")
    logger.info(f"  Total size: {df['size_mb'].sum():.1f} MB")
    logger.info(f"  Avg cells per dataset: {df['num_points'].mean():.0f}")
    logger.info(f"  Avg features per dataset: {df['num_features'].mean():.0f}")
    logger.info(f"\nSize distribution:")
    logger.info(f"  Min: {df['size_mb'].min():.2f} MB")
    logger.info(f"  Max: {df['size_mb'].max():.2f} MB")
    logger.info(f"  Median: {df['size_mb'].median():.2f} MB")

if __name__ == "__main__":
    main()
