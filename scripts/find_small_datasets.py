#!/usr/bin/env python3
"""
Find CELLxGENE datasets under 1GB and create a tracking database.
Queries the CELLxGENE Discover API for file sizes and metadata.
"""

import requests
import pandas as pd
import sqlite3
import json
import time
from pathlib import Path
from datetime import datetime
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
MAX_SIZE_BYTES = 1 * 1024 * 1024 * 1024  # 1GB in bytes
TARGET_DATASETS = 100
OUTPUT_DIR = Path("/home/btd8/llm-paper-analyze/data/manylatents_benchmark")
DB_PATH = OUTPUT_DIR / "manylatents_datasets.db"
CSV_PATH = OUTPUT_DIR / "manylatents_datasets.csv"

# CELLxGENE API endpoints
COLLECTIONS_API = "https://api.cellxgene.cziscience.com/curation/v1/collections"
DATASETS_API = "https://api.cellxgene.cziscience.com/curation/v1/datasets"


def create_database():
    """Create SQLite database with schema for tracking datasets."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS datasets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            dataset_id TEXT UNIQUE NOT NULL,
            dataset_name TEXT NOT NULL,
            description TEXT,
            collection_name TEXT,
            organism TEXT,
            tissue TEXT,
            assay TEXT,
            file_size_bytes INTEGER,
            file_size_mb REAL,
            n_cells INTEGER,
            n_features INTEGER,
            h5ad_url TEXT,
            doi TEXT,
            phate_plot_path TEXT,
            downloaded BOOLEAN DEFAULT FALSE,
            benchmarked BOOLEAN DEFAULT FALSE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    conn.commit()
    return conn


def fetch_all_datasets():
    """Fetch all dataset metadata from CELLxGENE API."""
    logger.info("Fetching dataset metadata from CELLxGENE API...")

    all_datasets = []
    page = 1
    per_page = 100

    while True:
        try:
            # Use the datasets index endpoint
            url = f"{DATASETS_API}/index"
            response = requests.get(url, timeout=60)
            response.raise_for_status()
            data = response.json()

            logger.info(f"Retrieved {len(data)} datasets from API")
            all_datasets = data
            break

        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching datasets: {e}")
            # Try alternative approach - fetch collections first
            logger.info("Trying to fetch via collections endpoint...")
            return fetch_datasets_via_collections()

    return all_datasets


def fetch_datasets_via_collections():
    """Alternative: fetch datasets by iterating through collections."""
    logger.info("Fetching all collections...")

    try:
        response = requests.get(COLLECTIONS_API, timeout=60)
        response.raise_for_status()
        collections = response.json()
        logger.info(f"Found {len(collections)} collections")
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching collections: {e}")
        return []

    all_datasets = []
    dataset_ids_to_fetch = []

    # First pass: collect basic info and dataset IDs
    for i, collection in enumerate(collections):
        if i % 50 == 0:
            logger.info(f"Processing collection {i+1}/{len(collections)}")

        collection_id = collection.get('collection_id') or collection.get('id')
        collection_name = collection.get('name', 'Unknown')
        doi = collection.get('doi', '')

        # Get datasets from collection
        datasets = collection.get('datasets', [])

        for dataset in datasets:
            dataset_id = dataset.get('dataset_id') or dataset.get('id')
            # Extract relevant metadata
            dataset_info = {
                'dataset_id': dataset_id,
                'dataset_name': dataset.get('name', dataset.get('title', 'Unknown')),
                'collection_name': collection_name,
                'doi': doi,
                'organism': extract_organism(dataset),
                'tissue': extract_tissue(dataset),
                'assay': extract_assay(dataset),
                'n_cells': dataset.get('cell_count', 0),
            }

            # Get assets for file size (might be present)
            assets = dataset.get('assets', [])
            for asset in assets:
                if asset.get('filetype') == 'H5AD':
                    dataset_info['file_size_bytes'] = asset.get('filesize', 0)
                    dataset_info['h5ad_url'] = asset.get('url', '')
                    break

            if 'file_size_bytes' not in dataset_info or dataset_info['file_size_bytes'] == 0:
                # Need to fetch individual dataset details
                dataset_ids_to_fetch.append((dataset_id, dataset_info))
            else:
                all_datasets.append(dataset_info)

        # Rate limiting
        if i % 10 == 0:
            time.sleep(0.1)

    # Second pass: fetch detailed info for datasets without size info
    if dataset_ids_to_fetch:
        logger.info(f"Fetching detailed info for {len(dataset_ids_to_fetch)} datasets...")

        for i, (dataset_id, base_info) in enumerate(dataset_ids_to_fetch):
            if i % 100 == 0:
                logger.info(f"Fetching dataset details {i+1}/{len(dataset_ids_to_fetch)}")

            try:
                # Fetch individual dataset details
                url = f"{DATASETS_API}/{dataset_id}"
                response = requests.get(url, timeout=30)
                if response.status_code == 200:
                    details = response.json()
                    assets = details.get('assets', [])
                    for asset in assets:
                        if asset.get('filetype') == 'H5AD':
                            base_info['file_size_bytes'] = asset.get('filesize', 0)
                            base_info['h5ad_url'] = asset.get('url', '')
                            break
                    if 'file_size_bytes' not in base_info:
                        base_info['file_size_bytes'] = 0
                        base_info['h5ad_url'] = ''
                else:
                    base_info['file_size_bytes'] = 0
                    base_info['h5ad_url'] = ''
            except Exception as e:
                logger.debug(f"Error fetching dataset {dataset_id}: {e}")
                base_info['file_size_bytes'] = 0
                base_info['h5ad_url'] = ''

            all_datasets.append(base_info)

            # Rate limiting - be nice to the API
            if i % 50 == 0:
                time.sleep(1)
            else:
                time.sleep(0.05)

    return all_datasets


def extract_organism(dataset):
    """Extract organism information from dataset metadata."""
    organisms = dataset.get('organism', [])
    if isinstance(organisms, list) and organisms:
        if isinstance(organisms[0], dict):
            return organisms[0].get('label', 'Unknown')
        return str(organisms[0])
    return 'Unknown'


def extract_tissue(dataset):
    """Extract tissue information from dataset metadata."""
    tissues = dataset.get('tissue', [])
    if isinstance(tissues, list) and tissues:
        if isinstance(tissues[0], dict):
            return tissues[0].get('label', 'Unknown')
        return str(tissues[0])
    return 'Unknown'


def extract_assay(dataset):
    """Extract assay information from dataset metadata."""
    assays = dataset.get('assay', [])
    if isinstance(assays, list) and assays:
        if isinstance(assays[0], dict):
            return assays[0].get('label', 'Unknown')
        return str(assays[0])
    return 'Unknown'


def filter_small_datasets(datasets):
    """Filter datasets to those under 1GB."""
    small_datasets = []

    for dataset in datasets:
        size_bytes = dataset.get('file_size_bytes', 0)
        if 0 < size_bytes < MAX_SIZE_BYTES:
            dataset['file_size_mb'] = size_bytes / (1024 * 1024)
            small_datasets.append(dataset)

    # Sort by file size (ascending) to prioritize smaller files
    small_datasets.sort(key=lambda x: x.get('file_size_bytes', 0))

    logger.info(f"Found {len(small_datasets)} datasets under 1GB")
    return small_datasets


def select_diverse_datasets(datasets, target_count=100):
    """Select a diverse set of datasets by organism, tissue, and size."""
    if len(datasets) <= target_count:
        return datasets

    # Group by organism and tissue to ensure diversity
    by_organism = {}
    for d in datasets:
        org = d.get('organism', 'Unknown')
        if org not in by_organism:
            by_organism[org] = []
        by_organism[org].append(d)

    selected = []

    # Round-robin selection across organisms
    while len(selected) < target_count and any(by_organism.values()):
        for org in list(by_organism.keys()):
            if by_organism[org] and len(selected) < target_count:
                selected.append(by_organism[org].pop(0))
            if not by_organism[org]:
                del by_organism[org]

    return selected


def populate_database(conn, datasets):
    """Insert dataset information into the database."""
    cursor = conn.cursor()

    for dataset in datasets:
        try:
            cursor.execute("""
                INSERT OR REPLACE INTO datasets (
                    dataset_id, dataset_name, description, collection_name,
                    organism, tissue, assay, file_size_bytes, file_size_mb,
                    n_cells, n_features, h5ad_url, doi
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                dataset.get('dataset_id'),
                dataset.get('dataset_name'),
                dataset.get('description', ''),
                dataset.get('collection_name'),
                dataset.get('organism'),
                dataset.get('tissue'),
                dataset.get('assay'),
                dataset.get('file_size_bytes'),
                dataset.get('file_size_mb'),
                dataset.get('n_cells'),
                dataset.get('n_features', None),  # Will be populated after download
                dataset.get('h5ad_url'),
                dataset.get('doi')
            ))
        except sqlite3.Error as e:
            logger.error(f"Error inserting dataset {dataset.get('dataset_id')}: {e}")

    conn.commit()
    logger.info(f"Inserted {len(datasets)} datasets into database")


def export_to_csv(conn):
    """Export database to CSV file."""
    df = pd.read_sql_query("""
        SELECT
            dataset_id,
            dataset_name,
            description,
            collection_name,
            organism,
            tissue,
            assay,
            ROUND(file_size_mb, 2) as size_mb,
            n_cells,
            n_features,
            doi,
            phate_plot_path,
            downloaded,
            benchmarked
        FROM datasets
        ORDER BY file_size_mb ASC
    """, conn)

    df.to_csv(CSV_PATH, index=False)
    logger.info(f"Exported {len(df)} datasets to {CSV_PATH}")
    return df


def main():
    """Main function to find and catalog small datasets."""
    logger.info("Starting dataset discovery for ManyLatents benchmarking")

    # Create database
    conn = create_database()
    logger.info(f"Database created at {DB_PATH}")

    # Fetch all datasets from API
    all_datasets = fetch_all_datasets()

    if not all_datasets:
        logger.error("No datasets retrieved from API")
        return

    logger.info(f"Total datasets found: {len(all_datasets)}")

    # Filter to datasets under 1GB
    small_datasets = filter_small_datasets(all_datasets)

    if len(small_datasets) < TARGET_DATASETS:
        logger.warning(f"Only found {len(small_datasets)} datasets under 1GB (target: {TARGET_DATASETS})")

    # Select diverse subset
    selected = select_diverse_datasets(small_datasets, TARGET_DATASETS)
    logger.info(f"Selected {len(selected)} datasets for benchmarking")

    # Populate database
    populate_database(conn, selected)

    # Export to CSV
    df = export_to_csv(conn)

    # Print summary
    print("\n" + "="*80)
    print("DATASET DISCOVERY SUMMARY")
    print("="*80)
    print(f"Total datasets in CELLxGENE: {len(all_datasets)}")
    print(f"Datasets under 1GB: {len(small_datasets)}")
    print(f"Selected for benchmarking: {len(selected)}")
    print(f"\nDatabase: {DB_PATH}")
    print(f"CSV Export: {CSV_PATH}")
    print(f"\nSize distribution:")
    print(df['size_mb'].describe())
    print(f"\nOrganisms represented:")
    print(df['organism'].value_counts().head(10))
    print(f"\nTissues represented:")
    print(df['tissue'].value_counts().head(10))

    conn.close()
    logger.info("Dataset discovery complete!")


if __name__ == "__main__":
    main()
