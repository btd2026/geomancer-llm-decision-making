#!/usr/bin/env python3
"""Check LLM descriptions in database."""

import sqlite3

DB_PATH = "/home/btd8/llm-paper-analyze/data/papers/metadata/papers.db"

conn = sqlite3.connect(DB_PATH)
conn.row_factory = sqlite3.Row

print("=" * 80)
print("LLM DESCRIPTIONS CHECK")
print("=" * 80)

# Check papers with descriptions
cursor = conn.execute("""
    SELECT COUNT(*) as total,
           SUM(CASE WHEN llm_description IS NOT NULL THEN 1 ELSE 0 END) as with_desc
    FROM papers
    WHERE source = 'cellxgene'
""")
row = cursor.fetchone()
print(f"\nPapers (CELLxGENE source):")
print(f"  Total: {row['total']}")
print(f"  With LLM description: {row['with_desc']}")
print(f"  Percentage: {row['with_desc']/row['total']*100:.1f}%")

# Show sample descriptions
print("\n" + "=" * 80)
print("SAMPLE DESCRIPTIONS")
print("=" * 80)

cursor = conn.execute("""
    SELECT title, llm_description, journal, publication_date
    FROM papers
    WHERE llm_description IS NOT NULL
    AND source = 'cellxgene'
    ORDER BY publication_date DESC
    LIMIT 5
""")

for i, row in enumerate(cursor.fetchall(), 1):
    print(f"\n{i}. {row['title'][:70]}...")
    print(f"   Journal: {row['journal']} ({row['publication_date']})")
    print(f"   Description: {row['llm_description']}")

conn.close()
print("\n" + "=" * 80)
