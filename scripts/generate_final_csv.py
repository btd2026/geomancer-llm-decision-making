#!/usr/bin/env python3
"""Simple script to generate classification CSV with all metadata."""

import scanpy as sc
import pandas as pd
import sqlite3
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DB_PATH = Path("/home/btd8/llm-paper-analyze/data/manylatents_benchmark/manylatents_datasets.db")
DATA_DIR = Path("/nfs/roberts/project/pi_sk2433/shared/Geomancer_2025_Data/manylatents_small_datasets")
CSV_OUTPUT = Path("/home/btd8/llm-paper-analyze/data/manylatents_benchmark/datasets_for_classification.csv")

logger.info("Generating classification CSV with feature extraction...")

# Read from database
conn = sqlite3.connect(DB_PATH)
df = pd.read_sql_query("""
    SELECT
        dataset_id,
        dataset_name,
        collection_name,
        collection_doi,
        file_size_mb,
        n_cells,
        h5ad_url,
        phate_plot_path
    FROM datasets
    WHERE downloaded = 1
    ORDER BY file_size_mb ASC
""", conn)
conn.close()

logger.info(f"Found {len(df)} datasets")

# Extract features from each file
n_features_list = []
for idx, row in df.iterrows():
    # Get version ID from URL
    version_id = row['h5ad_url'].split('/')[-1].replace('.h5ad', '')
    h5ad_path = DATA_DIR / f"{version_id}.h5ad"

    try:
        logger.info(f"[{idx+1}/{len(df)}] Processing {version_id}...")
        adata = sc.read_h5ad(h5ad_path)
        n_features = adata.n_vars
        n_features_list.append(n_features)
        logger.info(f"  ✓ {adata.n_obs} cells × {n_features} features")
        del adata
    except Exception as e:
        logger.error(f"  Error: {e}")
        n_features_list.append(None)

# Add features to dataframe
df['num_features'] = n_features_list

# Prepare final CSV
result_df = pd.DataFrame({
    'dataset_id': df['dataset_id'],
    'name': df['dataset_name'],
    'description': '',
    'collection_name': df['collection_name'],
    'collection_doi': df['collection_doi'],
    'size_mb': df['file_size_mb'].round(2),
    'num_points': df['n_cells'],
    'num_features': df['num_features'],
    'phate_plot_image': df['phate_plot_path'],
    'manual_classification': '',
    'notes': ''
})

# Save CSV
result_df.to_csv(CSV_OUTPUT, index=False)

logger.info(f"\n{'='*80}")
logger.info(f"CSV GENERATED: {CSV_OUTPUT}")
logger.info(f"{'='*80}")
logger.info(f"Total datasets: {len(result_df)}")
logger.info(f"\nFirst 3 rows:")
print(result_df.head(3)[['name', 'size_mb', 'num_points', 'num_features']].to_string())

logger.info(f"\n{'='*80}")
logger.info("Ready for manual classification!")
logger.info(f"CSV location: {CSV_OUTPUT}")
