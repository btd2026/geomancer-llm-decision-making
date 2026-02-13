#!/usr/bin/env python3
"""
Enrich dataset records with PubTator3 entity information.
Maps species, cell lines, and diseases to datasets.
"""
import json
import sqlite3
from pathlib import Path
from collections import Counter

db_path = Path(__file__).parent.parent / 'data' / 'papers' / 'metadata' / 'papers.db'

def get_paper_entities(cursor, paper_id, entity_type):
    """Get all entities of a specific type for a paper."""
    cursor.execute("""
        SELECT entity_text, entity_id, entity_name, COUNT(*) as count
        FROM pubtator_annotations
        WHERE paper_id = ? AND entity_type = ?
        GROUP BY entity_text, entity_id, entity_name
        ORDER BY count DESC
    """, (paper_id, entity_type))

    return cursor.fetchall()

def add_columns_if_not_exist(cursor):
    """Add new columns to datasets table if they don't exist."""
    new_columns = [
        ("species_ncbi_taxid", "TEXT"),
        ("species_name_normalized", "TEXT"),
        ("cell_line_name", "TEXT"),
        ("cell_line_cellosaurus_id", "TEXT"),
        ("disease_mesh_id", "TEXT"),
        ("disease_name", "TEXT"),
        ("genes_mentioned", "TEXT"),  # JSON array
        ("pubtator_enriched", "INTEGER DEFAULT 0")
    ]

    for col_name, col_type in new_columns:
        try:
            cursor.execute(f"ALTER TABLE datasets ADD COLUMN {col_name} {col_type}")
            print(f"  Added column: {col_name}")
        except sqlite3.OperationalError:
            # Column already exists
            pass

def main():
    """Main execution."""
    print("=" * 80)
    print("ENRICHING DATASETS WITH PUBTATOR3 ANNOTATIONS")
    print("=" * 80)
    print(f"Database: {db_path}")
    print()

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # Add new columns
    print("Checking database schema...")
    add_columns_if_not_exist(cursor)
    conn.commit()
    print()

    # Get all papers with datasets
    cursor.execute("""
        SELECT DISTINCT p.id, p.pmid, p.title
        FROM papers p
        JOIN datasets d ON p.id = d.paper_id
    """)

    papers = cursor.fetchall()
    print(f"Processing {len(papers)} papers with datasets...\n")

    datasets_updated = 0
    species_added = 0
    celllines_added = 0
    diseases_added = 0
    genes_added = 0

    for paper in papers:
        paper_id = paper['id']
        pmid = paper['pmid']

        print(f"[{pmid}] Analyzing annotations...")

        # Get entity annotations by type
        species_list = get_paper_entities(cursor, paper_id, 'Species')
        cellline_list = get_paper_entities(cursor, paper_id, 'CellLine')
        disease_list = get_paper_entities(cursor, paper_id, 'Disease')
        gene_list = get_paper_entities(cursor, paper_id, 'Gene')

        # Update datasets for this paper
        cursor.execute("SELECT id FROM datasets WHERE paper_id = ?", (paper_id,))
        dataset_ids = [row['id'] for row in cursor.fetchall()]

        if not dataset_ids:
            print(f"  → No datasets found for this paper")
            continue

        for dataset_id in dataset_ids:
            updates = {}
            update_flags = []

            # Most common species
            if species_list:
                species = species_list[0]
                updates['species_name_normalized'] = species['entity_name'] or species['entity_text']
                updates['species_ncbi_taxid'] = species['entity_id']
                update_flags.append(f"species: {species['entity_text']}")
                species_added += 1

            # Most common cell line
            if cellline_list:
                cellline = cellline_list[0]
                updates['cell_line_name'] = cellline['entity_name'] or cellline['entity_text']
                updates['cell_line_cellosaurus_id'] = cellline['entity_id']
                update_flags.append(f"cell line: {cellline['entity_text']}")
                celllines_added += 1

            # Most common disease
            if disease_list:
                disease = disease_list[0]
                updates['disease_name'] = disease['entity_name'] or disease['entity_text']
                updates['disease_mesh_id'] = disease['entity_id']
                update_flags.append(f"disease: {disease['entity_text']}")
                diseases_added += 1

            # Top genes (store as JSON)
            if gene_list:
                top_genes = [
                    {
                        'name': g['entity_name'] or g['entity_text'],
                        'text': g['entity_text'],
                        'id': g['entity_id'],
                        'mentions': g['count']
                    }
                    for g in gene_list[:20]  # Top 20 genes
                ]
                updates['genes_mentioned'] = json.dumps(top_genes)
                update_flags.append(f"genes: {len(gene_list)}")
                genes_added += 1

            # Mark as enriched
            updates['pubtator_enriched'] = 1

            # Apply updates
            if updates:
                update_sql = ', '.join([f"{k} = ?" for k in updates.keys()])
                update_values = list(updates.values()) + [dataset_id]
                cursor.execute(
                    f"UPDATE datasets SET {update_sql} WHERE id = ?",
                    update_values
                )
                datasets_updated += 1

                print(f"  → Dataset {dataset_id} updated: {', '.join(update_flags)}")

        print()

    conn.commit()
    conn.close()

    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Papers processed: {len(papers)}")
    print(f"Datasets updated: {datasets_updated}")
    print(f"  - Species information added: {species_added}")
    print(f"  - Cell line information added: {celllines_added}")
    print(f"  - Disease information added: {diseases_added}")
    print(f"  - Gene lists added: {genes_added}")
    print("=" * 80)

    # Show some statistics
    print()
    print("Dataset enrichment statistics:")
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    # Top species
    cursor.execute("""
        SELECT species_name_normalized, COUNT(*) as count
        FROM datasets
        WHERE species_name_normalized IS NOT NULL
        GROUP BY species_name_normalized
        ORDER BY count DESC
        LIMIT 10
    """)
    print("\nTop species:")
    for row in cursor.fetchall():
        print(f"  {row[0]}: {row[1]}")

    # Top cell lines
    cursor.execute("""
        SELECT cell_line_name, COUNT(*) as count
        FROM datasets
        WHERE cell_line_name IS NOT NULL
        GROUP BY cell_line_name
        ORDER BY count DESC
        LIMIT 10
    """)
    cell_lines = cursor.fetchall()
    if cell_lines:
        print("\nTop cell lines:")
        for row in cell_lines:
            print(f"  {row[0]}: {row[1]}")

    # Top diseases
    cursor.execute("""
        SELECT disease_name, COUNT(*) as count
        FROM datasets
        WHERE disease_name IS NOT NULL
        GROUP BY disease_name
        ORDER BY count DESC
        LIMIT 10
    """)
    diseases = cursor.fetchall()
    if diseases:
        print("\nTop diseases:")
        for row in diseases:
            print(f"  {row[0]}: {row[1]}")

    conn.close()

if __name__ == "__main__":
    main()
