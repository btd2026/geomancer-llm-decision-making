#!/usr/bin/env python3
"""
Create dataset catalog with metadata from database and h5ad files.

Generates a comprehensive catalog of all GEO datasets with:
- GEO accession
- Organism, tissue
- Cell and gene counts
- Source paper (PMID)
- File size

Usage:
    python3 create_dataset_catalog.py
"""

import sys
import sqlite3
from pathlib import Path
import pandas as pd
import scanpy as sc

# Configuration
BASE_DIR = Path('/home/btd8/llm-paper-analyze')
DATA_DIR = BASE_DIR / 'data/geo/processed'
DB_PATH = BASE_DIR / 'data/papers/metadata/papers.db'
OUTPUT_DIR = BASE_DIR / 'results/datasets'

DATASETS = [
    'GSE157827',
    'GSE159677',
    'GSE164983',
    'GSE167490',
    'GSE174367',
    'GSE191288',
    'GSE220243',
    'GSE271107'
]


def get_file_info(dataset_name):
    """Get file information for a dataset."""
    filepath = DATA_DIR / f"{dataset_name}.h5ad"

    if not filepath.exists():
        return None

    file_size_mb = filepath.stat().st_size / (1024 ** 2)

    # Try to read only metadata using backed mode (don't load data into memory)
    try:
        adata = sc.read_h5ad(filepath, backed='r')
        n_cells = adata.n_obs
        n_genes = adata.n_vars

        # Get organism and tissue if available
        organism = None
        tissue = None

        if hasattr(adata, 'obs') and adata.obs is not None:
            if 'organism' in adata.obs.columns and len(adata.obs) > 0:
                organism = str(adata.obs['organism'].iloc[0])
            if 'tissue' in adata.obs.columns and len(adata.obs) > 0:
                tissue = str(adata.obs['tissue'].iloc[0])

        # Close file
        if hasattr(adata, 'file'):
            adata.file.close()

        return {
            'n_cells': n_cells,
            'n_genes': n_genes,
            'file_size_mb': file_size_mb,
            'organism': organism,
            'tissue': tissue
        }

    except Exception as e:
        print(f"⚠️  Error loading {dataset_name}: {e}")
        import traceback
        traceback.print_exc()
        return {
            'n_cells': None,
            'n_genes': None,
            'file_size_mb': file_size_mb,
            'organism': None,
            'tissue': None
        }


def get_database_info(dataset_name):
    """Get paper information from database."""

    if not DB_PATH.exists():
        return None

    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        # Find paper with this GEO accession
        cursor.execute("""
            SELECT p.pmid, p.title, p.journal, p.publication_date
            FROM papers p
            JOIN datasets d ON p.id = d.paper_id
            WHERE d.accession_id = ?
            LIMIT 1
        """, (dataset_name,))

        result = cursor.fetchone()
        conn.close()

        if result:
            return {
                'pmid': result[0],
                'title': result[1],
                'journal': result[2],
                'publication_date': result[3]
            }

    except Exception as e:
        print(f"⚠️  Database error for {dataset_name}: {e}")

    return None


def create_catalog():
    """Create complete dataset catalog."""

    print("="*80)
    print("Creating Dataset Catalog")
    print("="*80)

    catalog_data = []

    for dataset in DATASETS:
        print(f"\nProcessing {dataset}...")

        # Get file info
        file_info = get_file_info(dataset)

        # Get database info
        db_info = get_database_info(dataset)

        # Combine
        entry = {
            'geo_accession': dataset,
            'n_cells': file_info['n_cells'] if file_info else None,
            'n_genes': file_info['n_genes'] if file_info else None,
            'file_size_mb': file_info['file_size_mb'] if file_info else None,
            'organism': file_info['organism'] if file_info else None,
            'tissue': file_info['tissue'] if file_info else None,
            'pmid': db_info['pmid'] if db_info else None,
            'title': db_info['title'] if db_info else None,
            'journal': db_info['journal'] if db_info else None,
            'publication_date': db_info['publication_date'] if db_info else None
        }

        catalog_data.append(entry)
        print(f"  ✅ {entry['n_cells']:,} cells, {entry['n_genes']:,} genes, {entry['file_size_mb']:.1f} MB")

    # Create DataFrame
    catalog_df = pd.DataFrame(catalog_data)

    return catalog_df


def save_catalog(catalog_df):
    """Save catalog in multiple formats."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # CSV
    output_path = OUTPUT_DIR / 'geo_datasets.csv'
    catalog_df.to_csv(output_path, index=False)
    print(f"\n✅ Saved CSV: {output_path}")

    # Formatted text
    output_path = OUTPUT_DIR / 'DATASET_CATALOG.txt'
    with open(output_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("GEO DATASET CATALOG\n")
        f.write("="*80 + "\n\n")

        f.write(f"Total datasets: {len(catalog_df)}\n")
        f.write(f"Total cells: {catalog_df['n_cells'].sum():,}\n")
        f.write(f"Total storage: {catalog_df['file_size_mb'].sum():.1f} MB\n\n")

        f.write("="*80 + "\n")
        f.write("DATASET DETAILS\n")
        f.write("="*80 + "\n\n")

        for idx, row in catalog_df.iterrows():
            f.write(f"Dataset: {row['geo_accession']}\n")
            f.write(f"  Cells: {row['n_cells']:,}\n")
            f.write(f"  Genes: {row['n_genes']:,}\n")
            f.write(f"  Size: {row['file_size_mb']:.1f} MB\n")

            if row['organism']:
                f.write(f"  Organism: {row['organism']}\n")
            if row['tissue']:
                f.write(f"  Tissue: {row['tissue']}\n")

            if row['pmid']:
                f.write(f"  PMID: {row['pmid']}\n")
                if row['title']:
                    f.write(f"  Paper: {row['title'][:70]}...\n")

            f.write("\n")

    print(f"✅ Saved text catalog: {output_path}")

    # Markdown
    output_path = OUTPUT_DIR / 'DATASET_CATALOG.md'
    with open(output_path, 'w') as f:
        f.write("# GEO Dataset Catalog\n\n")

        f.write("## Summary\n\n")
        f.write(f"- **Total datasets**: {len(catalog_df)}\n")
        f.write(f"- **Total cells**: {catalog_df['n_cells'].sum():,}\n")
        f.write(f"- **Total storage**: {catalog_df['file_size_mb'].sum():.1f} MB\n\n")

        f.write("---\n\n")
        f.write("## Datasets\n\n")

        for idx, row in catalog_df.iterrows():
            f.write(f"### {row['geo_accession']}\n\n")
            f.write(f"- **Cells**: {row['n_cells']:,}\n")
            f.write(f"- **Genes**: {row['n_genes']:,}\n")
            f.write(f"- **File size**: {row['file_size_mb']:.1f} MB\n")

            if row['organism']:
                f.write(f"- **Organism**: {row['organism']}\n")
            if row['tissue']:
                f.write(f"- **Tissue**: {row['tissue']}\n")

            if row['pmid']:
                f.write(f"- **PMID**: [{row['pmid']}](https://pubmed.ncbi.nlm.nih.gov/{row['pmid']}/)\n")
                if row['title']:
                    f.write(f"- **Paper**: {row['title']}\n")
                if row['journal']:
                    f.write(f"- **Journal**: {row['journal']}\n")

            f.write("\n")

    print(f"✅ Saved markdown catalog: {output_path}")


def main():
    catalog_df = create_catalog()
    save_catalog(catalog_df)

    print("\n"+"="*80)
    print("✅ DATASET CATALOG CREATED")
    print("="*80)

    return 0


if __name__ == '__main__':
    sys.exit(main())
