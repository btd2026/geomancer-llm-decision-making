#!/usr/bin/env python3
"""Extract n_features using anndata backed mode (memory efficient)."""

import anndata
import pandas as pd
import sqlite3
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

DB_PATH = Path("/home/btd8/llm-paper-analyze/data/manylatents_benchmark/manylatents_datasets.db")
DATA_DIR = Path("/nfs/roberts/project/pi_sk2433/shared/Geomancer_2025_Data/manylatents_small_datasets")

logger.info("Extracting n_features using anndata backed mode...")

# Read database
conn = sqlite3.connect(DB_PATH)
df = pd.read_sql_query("""
    SELECT dataset_id, dataset_name, h5ad_url
    FROM datasets WHERE downloaded = 1
    ORDER BY file_size_mb ASC
""", conn)

# Extract features using backed mode
features = []
for idx, row in df.iterrows():
    version_id = row['h5ad_url'].split('/')[-1].replace('.h5ad', '')
    h5ad_path = DATA_DIR / f"{version_id}.h5ad"

    try:
        # Use backed mode - only reads metadata, not data matrix
        adata = anndata.read_h5ad(h5ad_path, backed='r')
        n_features = adata.n_vars
        adata.file.close()  # Close file handle
        features.append(n_features)

        if (idx + 1) % 10 == 0:
            logger.info(f"Processed {idx+1}/{len(df)}")
    except Exception as e:
        logger.error(f"Error {version_id}: {e}")
        features.append(None)

# Update database
cursor = conn.cursor()
for idx, row in df.iterrows():
    if features[idx] is not None:
        cursor.execute("""
            UPDATE datasets
            SET n_features = ?
            WHERE dataset_id = ?
        """, (features[idx], row['dataset_id']))

conn.commit()
logger.info(f"\n✓ Updated database with feature counts")
logger.info(f"✓ Successful: {sum(1 for f in features if f is not None)}/{len(features)}")

# Now generate the final CSV
df_final = pd.read_sql_query("""
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
    ORDER BY file_size_mb ASC
""", conn)
conn.close()

# Add classification columns
df_final['manual_classification'] = ''
df_final['notes'] = ''

# Reorder columns
column_order = [
    'dataset_id', 'name', 'description', 'collection_name', 'collection_doi',
    'size_mb', 'num_points', 'num_features', 'phate_plot_image',
    'manual_classification', 'notes'
]
df_final = df_final[column_order]

# Save CSV
csv_path = Path("/home/btd8/llm-paper-analyze/data/manylatents_benchmark/datasets_for_classification.csv")
df_final.to_csv(csv_path, index=False)

logger.info(f"\n{'='*80}")
logger.info(f"✓ CSV GENERATED: {csv_path}")
logger.info(f"{'='*80}")
logger.info(f"Total datasets: {len(df_final)}")
logger.info(f"With features: {df_final['num_features'].notna().sum()}/{len(df_final)}")
logger.info(f"\nFirst 5 rows:")
print(df_final[['name', 'size_mb', 'num_points', 'num_features']].head(5).to_string(index=False))
