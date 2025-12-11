#!/usr/bin/env python3
"""
Quick verification script to check database population after testing.
"""

import sqlite3
from pathlib import Path

DB_PATH = "/home/btd8/llm-paper-analyze/data/papers/metadata/papers.db"

def main():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    print("=" * 80)
    print("DATABASE VERIFICATION REPORT")
    print("=" * 80)
    print()

    # Papers summary
    print("PAPERS SUMMARY")
    print("-" * 80)
    cursor = conn.execute("""
        SELECT
            COUNT(*) as total,
            SUM(CASE WHEN source = 'cellxgene' THEN 1 ELSE 0 END) as from_cellxgene,
            SUM(CASE WHEN pmid IS NOT NULL THEN 1 ELSE 0 END) as with_pmid,
            SUM(CASE WHEN abstract IS NOT NULL THEN 1 ELSE 0 END) as with_abstract,
            SUM(CASE WHEN has_full_text = 1 THEN 1 ELSE 0 END) as with_full_text,
            SUM(CASE WHEN collection_id IS NOT NULL THEN 1 ELSE 0 END) as with_collection_id
        FROM papers
    """)
    row = cursor.fetchone()
    print(f"Total papers: {row['total']}")
    print(f"From CELLxGENE: {row['from_cellxgene']}")
    print(f"With PMID: {row['with_pmid']}")
    print(f"With abstract: {row['with_abstract']}")
    print(f"With full-text: {row['with_full_text']}")
    print(f"With collection_id: {row['with_collection_id']}")
    print()

    # Datasets summary
    print("DATASETS SUMMARY")
    print("-" * 80)
    cursor = conn.execute("""
        SELECT
            COUNT(*) as total,
            SUM(CASE WHEN dataset_id IS NOT NULL THEN 1 ELSE 0 END) as with_cellxgene_id,
            SUM(CASE WHEN n_cells IS NOT NULL THEN 1 ELSE 0 END) as with_cell_count,
            AVG(CASE WHEN n_cells IS NOT NULL THEN n_cells ELSE 0 END) as avg_cells,
            SUM(CASE WHEN collection_id IS NOT NULL THEN 1 ELSE 0 END) as with_collection_id
        FROM datasets
    """)
    row = cursor.fetchone()
    print(f"Total datasets: {row['total']}")
    print(f"With CELLxGENE ID: {row['with_cellxgene_id']}")
    print(f"With cell count: {row['with_cell_count']}")
    print(f"Average cells: {row['avg_cells']:,.0f}")
    print(f"With collection_id: {row['with_collection_id']}")
    print()

    # Sample papers
    print("SAMPLE PAPERS (Most Recent)")
    print("-" * 80)
    cursor = conn.execute("""
        SELECT pmid, title, journal, publication_date
        FROM papers
        WHERE source = 'cellxgene'
        ORDER BY publication_date DESC
        LIMIT 5
    """)
    for i, row in enumerate(cursor.fetchall(), 1):
        print(f"\n{i}. PMID: {row['pmid']}")
        print(f"   Title: {row['title'][:70]}...")
        print(f"   Journal: {row['journal']}")
        print(f"   Date: {row['publication_date']}")
    print()

    # Sample datasets
    print("\nSAMPLE DATASETS (Largest by Cell Count)")
    print("-" * 80)
    cursor = conn.execute("""
        SELECT dataset_id, dataset_title, n_cells, organism
        FROM datasets
        WHERE dataset_id IS NOT NULL
        ORDER BY n_cells DESC
        LIMIT 5
    """)
    for i, row in enumerate(cursor.fetchall(), 1):
        print(f"\n{i}. Dataset: {row['dataset_id']}")
        print(f"   Title: {row['dataset_title'][:70]}...")
        print(f"   Cells: {row['n_cells']:,}")
        print(f"   Organism: {row['organism']}")
    print()

    # Check dataset-paper linkage
    print("DATASET-PAPER LINKAGE")
    print("-" * 80)
    cursor = conn.execute("""
        SELECT
            COUNT(DISTINCT d.id) as datasets_with_papers,
            COUNT(DISTINCT p.id) as papers_with_datasets
        FROM datasets d
        JOIN papers p ON d.paper_id = p.id
        WHERE d.dataset_id IS NOT NULL
    """)
    row = cursor.fetchone()
    print(f"Datasets linked to papers: {row['datasets_with_papers']}")
    print(f"Papers with linked datasets: {row['papers_with_datasets']}")
    print()

    # Collection summary
    print("COLLECTION SUMMARY")
    print("-" * 80)
    cursor = conn.execute("""
        SELECT
            p.collection_name,
            COUNT(d.id) as dataset_count,
            SUM(d.n_cells) as total_cells
        FROM papers p
        LEFT JOIN datasets d ON p.id = d.paper_id
        WHERE p.collection_id IS NOT NULL
        GROUP BY p.collection_id, p.collection_name
        ORDER BY dataset_count DESC
        LIMIT 5
    """)
    print("Top 5 collections by dataset count:")
    for i, row in enumerate(cursor.fetchall(), 1):
        collection_name = row['collection_name'][:50] if row['collection_name'] else 'Unknown'
        print(f"{i}. {collection_name}...")
        print(f"   Datasets: {row['dataset_count']}, Total cells: {row['total_cells']:,}")
    print()

    print("=" * 80)
    print("✓ Database verification complete")
    print("=" * 80)

    conn.close()

if __name__ == "__main__":
    main()
