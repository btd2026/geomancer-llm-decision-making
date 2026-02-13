#!/usr/bin/env python3
"""Show summary of LLM descriptions in database."""

import sqlite3

DB_PATH = "/home/btd8/llm-paper-analyze/data/papers/metadata/papers.db"

conn = sqlite3.connect(DB_PATH)
conn.row_factory = sqlite3.Row

print("=" * 80)
print("LLM DESCRIPTION GENERATION - FINAL SUMMARY")
print("=" * 80)

# Overall statistics
cursor = conn.execute("""
    SELECT
        COUNT(*) as total,
        SUM(CASE WHEN llm_description IS NOT NULL THEN 1 ELSE 0 END) as with_desc,
        COALESCE(source, 'legacy') as source
    FROM papers
    GROUP BY source
""")

print("\nPapers by source:")
total_all = 0
with_desc_all = 0
for row in cursor.fetchall():
    total_all += row['total']
    with_desc_all += row['with_desc']
    pct = (row['with_desc'] / row['total'] * 100) if row['total'] > 0 else 0
    print(f"  {row['source']:15} {row['total']:3} papers  |  {row['with_desc']:3} with descriptions ({pct:.0f}%)")

pct_total = (with_desc_all / total_all * 100) if total_all > 0 else 0
print(f"  {'TOTAL':15} {total_all:3} papers  |  {with_desc_all:3} with descriptions ({pct_total:.0f}%)")

# CELLxGENE papers specifically
print("\n" + "=" * 80)
print("CELLXGENE PAPERS WITH DESCRIPTIONS")
print("=" * 80)

cursor = conn.execute("""
    SELECT
        pmid, title, llm_description, journal, publication_date, collection_name
    FROM papers
    WHERE source = 'cellxgene'
    AND llm_description IS NOT NULL
    ORDER BY publication_date DESC
""")

cellxgene_papers = cursor.fetchall()
if cellxgene_papers:
    for i, row in enumerate(cellxgene_papers, 1):
        print(f"\n{i}. {row['title'][:70]}...")
        print(f"   PMID: {row['pmid']}")
        print(f"   Journal: {row['journal']} ({row['publication_date']})")
        print(f"   Collection: {row['collection_name'][:60]}...")
        print(f"   Description: {row['llm_description']}")
else:
    print("\n⚠️  No CELLxGENE papers with descriptions yet")

# Sample of recent papers with descriptions
print("\n" + "=" * 80)
print("SAMPLE OF ALL PAPERS WITH DESCRIPTIONS (Most Recent)")
print("=" * 80)

cursor = conn.execute("""
    SELECT
        pmid, title, llm_description, journal, publication_date, source
    FROM papers
    WHERE llm_description IS NOT NULL
    ORDER BY publication_date DESC
    LIMIT 10
""")

for i, row in enumerate(cursor.fetchall(), 1):
    print(f"\n{i}. [{row['source']}] {row['title'][:65]}...")
    print(f"   PMID: {row['pmid']} | {row['journal']} ({row['publication_date']})")
    print(f"   Description: {row['llm_description'][:150]}...")

conn.close()
print("\n" + "=" * 80)
