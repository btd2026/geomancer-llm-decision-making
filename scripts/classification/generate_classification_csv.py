#!/usr/bin/env python3
"""Generate final classification CSV with all metadata."""

import pandas as pd
import sqlite3
from pathlib import Path

DB_PATH = Path("/home/btd8/llm-paper-analyze/data/manylatents_benchmark/manylatents_datasets.db")
CSV_OUTPUT = Path("/home/btd8/llm-paper-analyze/data/manylatents_benchmark/datasets_for_classification.csv")

# Read from database
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
    ORDER BY file_size_mb ASC
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
    'collection_doi',
    'size_mb',
    'num_points',
    'num_features',
    'phate_plot_image',
    'manual_classification',
    'notes'
]

df = df[column_order]

# Save CSV
CSV_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(CSV_OUTPUT, index=False)

print(f"{'='*80}")
print(f"CLASSIFICATION CSV GENERATED")
print(f"{'='*80}")
print(f"Location: {CSV_OUTPUT}")
print(f"Total datasets: {len(df)}")
print(f"With features: {df['num_features'].notna().sum()}/{len(df)}")
print(f"\nColumns: {', '.join(df.columns)}")
print(f"\nSample data:")
print(df[['name', 'size_mb', 'num_points', 'num_features']].head(10).to_string(index=False))
print(f"\nSummary statistics:")
print(f"  Total cells: {df['num_points'].sum():,}")
print(f"  Total size: {df['size_mb'].sum():.1f} MB")
print(f"  Avg cells/dataset: {df['num_points'].mean():.0f}")
print(f"  Avg features/dataset: {df['num_features'].mean():.0f}")
print(f"{'='*80}")
print(f"\nReady for manual classification!")
