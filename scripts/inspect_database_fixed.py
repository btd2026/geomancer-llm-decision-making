#!/usr/bin/env python3
"""
Inspect papers.db to show current state and new CELLxGENE changes.
"""

import sqlite3
from pathlib import Path

DB_PATH = "/home/btd8/llm-paper-analyze/data/papers/metadata/papers.db"

def main():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    print("=" * 80)
    print("PAPERS.DB DATABASE INSPECTION")
    print("=" * 80)
    print()

    # ============================================================================
    # 1. SCHEMA OVERVIEW
    # ============================================================================
    print("1. DATABASE SCHEMA")
    print("-" * 80)

    # Papers table schema
    cursor = conn.execute("PRAGMA table_info(papers)")
    papers_cols = cursor.fetchall()
    print(f"\nPAPERS TABLE ({len(papers_cols)} columns):")
    print("\nOLD COLUMNS (before CELLxGENE integration):")
    old_cols = ['id', 'pmid', 'doi', 'title', 'abstract', 'authors', 'journal',
                'publication_date', 'created_at']
    for col in papers_cols:
        if col['name'] in old_cols:
            print(f"  • {col['name']:20} {col['type']:10} {'' if col['notnull'] else '(nullable)'}")

    print("\n✨ NEW COLUMNS (CELLxGENE integration):")
    new_cols = ['collection_id', 'all_collection_ids', 'collection_name', 'source',
                'llm_description', 'full_text', 'has_full_text']
    for col in papers_cols:
        if col['name'] in new_cols:
            print(f"  • {col['name']:20} {col['type']:10} {'' if col['notnull'] else '(nullable)'}")

    # Datasets table schema
    cursor = conn.execute("PRAGMA table_info(datasets)")
    datasets_cols = cursor.fetchall()
    dataset_col_names = [col['name'] for col in datasets_cols]

    print(f"\n\nDATASETS TABLE ({len(datasets_cols)} columns):")
    print("\nOLD COLUMNS:")
    old_dataset_cols = ['id', 'paper_id', 'title', 'summary',
                        'organism', 'n_samples', 'created_at']
    for col in datasets_cols:
        if col['name'] in old_dataset_cols:
            print(f"  • {col['name']:20} {col['type']:10} {'' if col['notnull'] else '(nullable)'}")

    print("\n✨ NEW COLUMNS (CELLxGENE integration):")
    new_dataset_cols = ['dataset_id', 'collection_id', 'dataset_title',
                        'dataset_version_id', 'dataset_h5ad_path', 'llm_description',
                        'citation', 'downloaded', 'benchmarked', 'n_cells', 'n_genes']
    for col in datasets_cols:
        if col['name'] in new_dataset_cols:
            print(f"  • {col['name']:20} {col['type']:10} {'' if col['notnull'] else '(nullable)'}")

    print("\n")

    # ============================================================================
    # 2. DATA COUNTS BY SOURCE
    # ============================================================================
    print("\n2. DATA INVENTORY")
    print("-" * 80)

    # Papers by source
    cursor = conn.execute("""
        SELECT
            COALESCE(source, 'legacy') as source,
            COUNT(*) as count
        FROM papers
        GROUP BY source
        ORDER BY count DESC
    """)
    print("\nPapers by source:")
    total_papers = 0
    for row in cursor.fetchall():
        print(f"  • {row['source']:15} {row['count']:4} papers")
        total_papers += row['count']
    print(f"  {'TOTAL':15} {total_papers:4} papers")

    # Datasets by type
    cursor = conn.execute("""
        SELECT
            CASE
                WHEN dataset_id IS NOT NULL THEN 'CELLxGENE'
                ELSE 'Legacy'
            END as type,
            COUNT(*) as count
        FROM datasets
        GROUP BY type
        ORDER BY count DESC
    """)
    print("\nDatasets by type:")
    total_datasets = 0
    for row in cursor.fetchall():
        print(f"  • {row['type']:15} {row['count']:4} datasets")
        total_datasets += row['count']
    print(f"  {'TOTAL':15} {total_datasets:4} datasets")

    # ============================================================================
    # 3. NEW CELLXGENE DATA EXAMPLES
    # ============================================================================
    print("\n\n3. NEW CELLxGENE DATA (from recent test)")
    print("-" * 80)

    cursor = conn.execute("""
        SELECT
            pmid,
            title,
            collection_id,
            collection_name,
            journal,
            publication_date,
            source
        FROM papers
        WHERE source = 'cellxgene'
        ORDER BY publication_date DESC
        LIMIT 3
    """)

    print("\nSample CELLxGENE Papers:")
    for i, paper in enumerate(cursor.fetchall(), 1):
        print(f"\n{i}. {paper['title'][:70]}...")
        print(f"   PMID: {paper['pmid']}")
        print(f"   Journal: {paper['journal']} ({paper['publication_date']})")
        print(f"   Collection ID: {paper['collection_id']}")
        print(f"   Collection Name: {paper['collection_name'][:60]}...")
        print(f"   Source: {paper['source']}")

    # Show linked datasets
    print("\n\nSample CELLxGENE Datasets with linkage:")
    cursor = conn.execute("""
        SELECT
            d.dataset_id,
            d.dataset_title,
            d.n_cells,
            d.collection_id,
            p.title as paper_title
        FROM datasets d
        JOIN papers p ON d.paper_id = p.id
        WHERE d.dataset_id IS NOT NULL
        ORDER BY d.n_cells DESC
        LIMIT 3
    """)

    for i, ds in enumerate(cursor.fetchall(), 1):
        print(f"\n{i}. Dataset: {ds['dataset_id']}")
        print(f"   Title: {ds['dataset_title'][:60]}...")
        print(f"   Cells: {ds['n_cells']:,}")
        print(f"   Collection: {ds['collection_id']}")
        print(f"   Linked Paper: {ds['paper_title'][:60]}...")

    # ============================================================================
    # 4. BEFORE vs AFTER COMPARISON
    # ============================================================================
    print("\n\n4. BEFORE vs AFTER COMPARISON")
    print("-" * 80)

    cursor = conn.execute("""
        SELECT COUNT(*) as legacy_papers
        FROM papers
        WHERE source IS NULL OR source != 'cellxgene'
    """)
    legacy_papers = cursor.fetchone()['legacy_papers']

    cursor = conn.execute("""
        SELECT COUNT(*) as cellxgene_papers
        FROM papers
        WHERE source = 'cellxgene'
    """)
    cellxgene_papers = cursor.fetchone()['cellxgene_papers']

    cursor = conn.execute("""
        SELECT COUNT(*) as legacy_datasets
        FROM datasets
        WHERE dataset_id IS NULL
    """)
    legacy_datasets = cursor.fetchone()['legacy_datasets']

    cursor = conn.execute("""
        SELECT COUNT(*) as cellxgene_datasets
        FROM datasets
        WHERE dataset_id IS NOT NULL
    """)
    cellxgene_datasets = cursor.fetchone()['cellxgene_datasets']

    print("\nBEFORE (legacy data):")
    print(f"  • Papers: {legacy_papers}")
    print(f"  • Datasets: {legacy_datasets}")
    print(f"  • Source: Manual searches")
    print(f"  • Linkage: Variable quality")

    print("\nAFTER (with CELLxGENE integration):")
    print(f"  • New Papers: {cellxgene_papers}")
    print(f"  • CELLxGENE Datasets: {cellxgene_datasets}")
    print(f"  • Source: CELLxGENE Census (curated)")
    print(f"  • Linkage: 100% guaranteed (via DOI)")
    print(f"  • New fields: collection_id, dataset_id, citation, etc.")

    # ============================================================================
    # 5. NEW CAPABILITIES
    # ============================================================================
    print("\n\n5. NEW CAPABILITIES ENABLED")
    print("-" * 80)

    print("\n✓ Query by CELLxGENE collection:")
    print("  Example: SELECT * FROM papers WHERE collection_id = 'abc123'")

    print("\n✓ Find all datasets for a paper:")
    print("  Example: SELECT * FROM datasets WHERE collection_id = 'abc123'")

    print("\n✓ Track dataset download status:")
    print("  Example: SELECT * FROM datasets WHERE downloaded = 1")

    print("\n✓ Track benchmarking status:")
    print("  Example: SELECT * FROM datasets WHERE benchmarked = 1")

    print("\n✓ Link to CELLxGENE H5AD files:")
    print("  Example: SELECT dataset_h5ad_path FROM datasets WHERE dataset_id = '...'")

    print("\n✓ Ready for LLM descriptions:")
    print("  Fields: papers.llm_description, datasets.llm_description")

    print("\n✓ Ready for algorithm extraction:")
    print("  Fields: papers.full_text, papers.has_full_text")

    # ============================================================================
    # 6. INDICES FOR PERFORMANCE
    # ============================================================================
    print("\n\n6. DATABASE INDICES (for fast queries)")
    print("-" * 80)

    cursor = conn.execute("""
        SELECT name, tbl_name
        FROM sqlite_master
        WHERE type = 'index'
        AND name LIKE 'idx_%'
        ORDER BY tbl_name, name
    """)

    current_table = None
    for idx in cursor.fetchall():
        if idx['tbl_name'] != current_table:
            print(f"\n{idx['tbl_name'].upper()} table:")
            current_table = idx['tbl_name']
        print(f"  • {idx['name']}")

    # ============================================================================
    # 7. WHERE TO SEE THE CHANGES
    # ============================================================================
    print("\n\n7. WHERE TO SEE THE NEW CHANGES")
    print("-" * 80)

    print("\n📁 Database file location:")
    print(f"   {DB_PATH}")

    print("\n📊 View with Python:")
    print(f"   python3 /home/btd8/llm-paper-analyze/scripts/verify_database_test.py")

    print("\n📄 Full test report:")
    print(f"   cat /home/btd8/llm-paper-analyze/TEST_RESULTS_100_DATASETS.md")

    print("\n🔍 Query examples (save as .py file):")
    print("""
import sqlite3
conn = sqlite3.connect("/home/btd8/llm-paper-analyze/data/papers/metadata/papers.db")

# Find all CELLxGENE papers
papers = conn.execute("SELECT * FROM papers WHERE source='cellxgene'").fetchall()

# Find all CELLxGENE datasets
datasets = conn.execute("SELECT * FROM datasets WHERE dataset_id IS NOT NULL").fetchall()

# Find papers with most datasets
result = conn.execute('''
    SELECT p.title, COUNT(d.id) as dataset_count
    FROM papers p
    JOIN datasets d ON p.id = d.paper_id
    WHERE p.source = 'cellxgene'
    GROUP BY p.id
    ORDER BY dataset_count DESC
''').fetchall()
""")

    # ============================================================================
    # 8. SAMPLE QUERY RESULTS
    # ============================================================================
    print("\n\n8. EXAMPLE QUERY: Papers with Most Datasets")
    print("-" * 80)

    cursor = conn.execute("""
        SELECT
            p.collection_name,
            p.journal,
            COUNT(d.id) as dataset_count,
            SUM(d.n_cells) as total_cells
        FROM papers p
        JOIN datasets d ON p.id = d.paper_id
        WHERE p.source = 'cellxgene'
        GROUP BY p.id
        ORDER BY dataset_count DESC
        LIMIT 5
    """)

    for i, row in enumerate(cursor.fetchall(), 1):
        collection = row['collection_name'][:50] if row['collection_name'] else 'Unknown'
        print(f"\n{i}. {collection}...")
        print(f"   Journal: {row['journal']}")
        print(f"   Datasets: {row['dataset_count']}")
        print(f"   Total cells: {row['total_cells']:,}")

    print("\n")
    print("=" * 80)
    print("✓ DATABASE INSPECTION COMPLETE")
    print("=" * 80)
    print()

    conn.close()

if __name__ == "__main__":
    main()
