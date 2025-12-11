#!/usr/bin/env python3
"""Check LLM descriptions in database (all sources)."""

import sqlite3

DB_PATH = "/home/btd8/llm-paper-analyze/data/papers/metadata/papers.db"

conn = sqlite3.connect(DB_PATH)
conn.row_factory = sqlite3.Row

print("=" * 80)
print("LLM DESCRIPTIONS CHECK (ALL SOURCES)")
print("=" * 80)

# Check all papers by source
cursor = conn.execute("""
    SELECT
        COALESCE(source, 'legacy') as source,
        COUNT(*) as total,
        SUM(CASE WHEN llm_description IS NOT NULL THEN 1 ELSE 0 END) as with_desc
    FROM papers
    GROUP BY source
    ORDER BY total DESC
""")

print("\nPapers by source:")
for row in cursor.fetchall():
    print(f"  {row['source']:15} Total: {row['total']:3}  With desc: {row['with_desc']:3}")

# Check total
cursor = conn.execute("""
    SELECT COUNT(*) as total,
           SUM(CASE WHEN llm_description IS NOT NULL THEN 1 ELSE 0 END) as with_desc
    FROM papers
""")
row = cursor.fetchone()
print(f"\n  {'TOTAL':15} Total: {row['total']:3}  With desc: {row['with_desc']:3}")

# Show any papers with descriptions
cursor = conn.execute("""
    SELECT pmid, title, llm_description, source
    FROM papers
    WHERE llm_description IS NOT NULL
    ORDER BY id DESC
    LIMIT 5
""")

results = cursor.fetchall()
if results:
    print("\n" + "=" * 80)
    print("PAPERS WITH DESCRIPTIONS")
    print("=" * 80)
    for i, row in enumerate(results, 1):
        print(f"\n{i}. PMID: {row['pmid']}")
        print(f"   Source: {row['source']}")
        print(f"   Title: {row['title'][:70]}...")
        print(f"   Description: {row['llm_description'][:150]}...")
else:
    print("\n⚠️  No papers with descriptions found")

conn.close()
print("\n" + "=" * 80)
