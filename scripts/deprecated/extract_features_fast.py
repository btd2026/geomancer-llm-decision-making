#!/usr/bin/env python3
"""Fast feature extraction using h5py (no full data loading)."""

import h5py
import pandas as pd
import sqlite3
from pathlib import Path

DB_PATH = Path("/home/btd8/llm-paper-analyze/data/manylatents_benchmark/manylatents_datasets.db")
DATA_DIR = Path("/nfs/roberts/project/pi_sk2433/shared/Geomancer_2025_Data/manylatents_small_datasets")
CSV_OUTPUT = Path("/home/btd8/llm-paper-analyze/data/manylatents_benchmark/datasets_for_classification.csv")

print("Extracting features using h5py (fast & low memory)...")

# Read database
conn = sqlite3.connect(DB_PATH)
df = pd.read_sql_query("""
    SELECT dataset_id, dataset_name, collection_name, collection_doi,
           file_size_mb, n_cells, h5ad_url, phate_plot_path
    FROM datasets WHERE downloaded = 1
    ORDER BY file_size_mb ASC
""", conn)
conn.close()

# Extract features using h5py (no memory overhead)
features = []
for idx, row in df.iterrows():
    version_id = row['h5ad_url'].split('/')[-1].replace('.h5ad', '')
    h5ad_path = DATA_DIR / f"{version_id}.h5ad"

    try:
        with h5py.File(h5ad_path, 'r') as f:
            # Get shape without loading data
            if 'X' in f:
                n_features = f['X'].shape[1]
            elif 'raw/X' in f:
                n_features = f['raw/X'].shape[1]
            else:
                n_features = None
        features.append(n_features)
        if (idx + 1) % 10 == 0:
            print(f"Processed {idx+1}/100")
    except Exception as e:
        print(f"Error {version_id}: {e}")
        features.append(None)

df['num_features'] = features

# Create final CSV
result = pd.DataFrame({
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

result.to_csv(CSV_OUTPUT, index=False)

print(f"\n✓ CSV generated: {CSV_OUTPUT}")
print(f"✓ Total datasets: {len(result)}")
print(f"✓ With features: {result['num_features'].notna().sum()}/100")
print(f"\nFirst 3 rows:")
print(result[['name', 'size_mb', 'num_points', 'num_features']].head(3))
