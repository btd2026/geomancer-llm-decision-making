#!/usr/bin/env python3
"""Show sample dataset descriptions."""

import sqlite3

DB_PATH = "/home/btd8/llm-paper-analyze/data/papers/metadata/papers.db"

conn = sqlite3.connect(DB_PATH)
conn.row_factory = sqlite3.Row

print("=" * 80)
print("DATASET DESCRIPTIONS - SAMPLE")
print("=" * 80)

# Get sample descriptions
cursor = conn.execute("""
    SELECT
        d.dataset_id,
        d.dataset_title,
        d.n_cells,
        d.llm_description,
        p.title as paper_title,
        p.journal
    FROM datasets d
    JOIN papers p ON d.paper_id = p.id
    WHERE d.llm_description IS NOT NULL
    AND d.dataset_id IS NOT NULL
    ORDER BY d.n_cells DESC
    LIMIT 10
""")

for i, row in enumerate(cursor.fetchall(), 1):
    print(f"\n{i}. Dataset: {row['dataset_id']}")
    print(f"   Title: {row['dataset_title'][:65]}...")
    print(f"   Cells: {row['n_cells']:,}")
    print(f"   Paper: {row['paper_title'][:65]}...")
    print(f"   Journal: {row['journal']}")
    print(f"   Description: {row['llm_description']}")

conn.close()
print("\n" + "=" * 80)
