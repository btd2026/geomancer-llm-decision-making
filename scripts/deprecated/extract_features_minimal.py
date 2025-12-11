#!/usr/bin/env python3
"""Extract n_features one file at a time with aggressive memory cleanup."""

import h5py
import sqlite3
from pathlib import Path
import gc

DB_PATH = Path("/home/btd8/llm-paper-analyze/data/manylatents_benchmark/manylatents_datasets.db")
DATA_DIR = Path("/nfs/roberts/project/pi_sk2433/shared/Geomancer_2025_Data/manylatents_small_datasets")

# Get datasets
conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()
cursor.execute("""
    SELECT dataset_id, h5ad_url
    FROM datasets WHERE downloaded = 1
    ORDER BY file_size_mb ASC
""")
datasets = cursor.fetchall()

print(f"Processing {len(datasets)} datasets...")

# Process one at a time
for idx, (dataset_id, h5ad_url) in enumerate(datasets, 1):
    version_id = h5ad_url.split('/')[-1].replace('.h5ad', '')
    h5ad_path = DATA_DIR / f"{version_id}.h5ad"

    try:
        # Use h5py to read just the var dimension
        with h5py.File(h5ad_path, 'r') as f:
            # Try to get n_vars from var dataframe
            if 'var' in f and '_index' in f['var']:
                n_features = len(f['var']['_index'])
            # Fallback: check shape attribute if it exists
            elif 'obs' in f and '_index' in f['obs']:
                # We can infer from the structure
                # Look for any dataset that has shape info
                if 'X' in f:
                    # For CSR sparse matrix
                    if 'indptr' in f['X']:
                        # CSR format: shape stored in attrs
                        if 'shape' in f['X'].attrs:
                            n_features = f['X'].attrs['shape'][1]
                        elif 'h5sparse_shape' in f['X'].attrs:
                            n_features = f['X'].attrs['h5sparse_shape'][1]
                        else:
                            n_features = None
                    else:
                        n_features = None
                else:
                    n_features = None
            else:
                n_features = None

        if n_features is not None:
            cursor.execute("""
                UPDATE datasets
                SET n_features = ?
                WHERE dataset_id = ?
            """, (int(n_features), dataset_id))
            conn.commit()

        if idx % 10 == 0:
            print(f"Processed {idx}/{len(datasets)}")

        # Aggressive cleanup
        gc.collect()

    except Exception as e:
        print(f"Error {version_id}: {e}")
        continue

conn.close()
print(f"\n✓ Feature extraction complete")
