#!/usr/bin/env python3
"""Check dataset descriptions in database."""

import sqlite3

DB_PATH = "/home/btd8/llm-paper-analyze/data/papers/metadata/papers.db"

conn = sqlite3.connect(DB_PATH)
conn.row_factory = sqlite3.Row

print("=" * 80)
print("DATASET DESCRIPTIONS CHECK")
print("=" * 80)

# Check datasets with CELLxGENE IDs
cursor = conn.execute("""
    SELECT COUNT(*) as total,
           SUM(CASE WHEN llm_description IS NOT NULL THEN 1 ELSE 0 END) as with_desc
    FROM datasets
    WHERE dataset_id IS NOT NULL
""")
row = cursor.fetchone()
print(f"\nCELLxGENE Datasets:")
print(f"  Total: {row['total']}")
print(f"  With LLM description: {row['with_desc']}")
print(f"  Need descriptions: {row['total'] - row['with_desc']}")

# Show sample datasets
print("\n" + "=" * 80)
print("SAMPLE DATASETS (waiting for descriptions)")
print("=" * 80)

cursor = conn.execute("""
    SELECT d.dataset_id, d.dataset_title, d.n_cells, p.title as paper_title
    FROM datasets d
    JOIN papers p ON d.paper_id = p.id
    WHERE d.dataset_id IS NOT NULL
    ORDER BY d.n_cells DESC
    LIMIT 5
""")

for i, row in enumerate(cursor.fetchall(), 1):
    print(f"\n{i}. Dataset: {row['dataset_id']}")
    print(f"   Title: {row['dataset_title'][:60]}...")
    print(f"   Cells: {row['n_cells']:,}")
    print(f"   Paper: {row['paper_title'][:60]}...")

conn.close()
print("\n" + "=" * 80)
