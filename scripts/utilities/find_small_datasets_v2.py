#!/usr/bin/env python3
"""
Find CELLxGENE datasets under 1GB using HEAD requests to get actual file sizes.
Uses the existing metadata CSV and checks file sizes directly.
"""

import requests
import pandas as pd
import sqlite3
import csv
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
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
INPUT_CSV = Path("/home/btd8/llm-paper-analyze/cellxgene_full_metadata.csv")

# CELLxGENE download URL pattern
# From citation: https://datasets.cellxgene.cziscience.com/{version_id}.h5ad
BASE_DOWNLOAD_URL = "https://datasets.cellxgene.cziscience.com"


def create_database():
    """Create SQLite database with schema for tracking datasets."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("DROP TABLE IF EXISTS datasets")
    cursor.execute("""
        CREATE TABLE datasets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            dataset_id TEXT UNIQUE NOT NULL,
            dataset_name TEXT NOT NULL,
            description TEXT,
            collection_name TEXT,
            collection_doi TEXT,
            file_size_bytes INTEGER,
            file_size_mb REAL,
            n_cells INTEGER,
            n_features INTEGER,
            h5ad_url TEXT,
            phate_plot_path TEXT,
            downloaded BOOLEAN DEFAULT FALSE,
            benchmarked BOOLEAN DEFAULT FALSE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    conn.commit()
    return conn


def load_metadata():
    """Load the existing CELLxGENE metadata CSV."""
    logger.info(f"Loading metadata from {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV)
    logger.info(f"Loaded {len(df)} datasets")
    return df


def get_file_size(dataset_version_id):
    """Get file size using HTTP HEAD request."""
    url = f"{BASE_DOWNLOAD_URL}/{dataset_version_id}.h5ad"
    try:
        response = requests.head(url, timeout=10, allow_redirects=True)
        if response.status_code == 200:
            size = int(response.headers.get('Content-Length', 0))
            return size
        else:
            return None
    except Exception as e:
        logger.debug(f"Error getting size for {dataset_version_id}: {e}")
        return None


def fetch_file_sizes(df, max_workers=10):
    """Fetch file sizes for all datasets using parallel requests."""
    logger.info(f"Fetching file sizes for {len(df)} datasets...")

    # Create a mapping of version_id to size
    size_map = {}
    version_ids = df['dataset_version_id'].tolist()

    # Use ThreadPoolExecutor for parallel requests
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_id = {
            executor.submit(get_file_size, vid): vid
            for vid in version_ids
        }

        # Collect results
        completed = 0
        for future in as_completed(future_to_id):
            vid = future_to_id[future]
            try:
                size = future.result()
                size_map[vid] = size
            except Exception as e:
                logger.debug(f"Exception for {vid}: {e}")
                size_map[vid] = None

            completed += 1
            if completed % 100 == 0:
                logger.info(f"Checked {completed}/{len(version_ids)} datasets")

    return size_map


def filter_and_select_datasets(df, size_map, target_count=100):
    """Filter datasets under 1GB and select a diverse subset."""
    # Add size information to dataframe
    df['file_size_bytes'] = df['dataset_version_id'].map(size_map)
    df['file_size_mb'] = df['file_size_bytes'] / (1024 * 1024)

    # Filter to datasets with valid size info and under 1GB
    valid_df = df[
        (df['file_size_bytes'].notna()) &
        (df['file_size_bytes'] > 0) &
        (df['file_size_bytes'] < MAX_SIZE_BYTES)
    ].copy()

    logger.info(f"Found {len(valid_df)} datasets under 1GB")

    if len(valid_df) <= target_count:
        return valid_df

    # Sort by size to prioritize smaller files (faster to process)
    valid_df = valid_df.sort_values('file_size_bytes')

    # Select diverse subset - try to get different collections
    # Group by collection and sample proportionally
    selected = []
    collections = valid_df['collection_id'].unique()

    # First pass: take at least one from each collection (up to target)
    for coll_id in collections:
        if len(selected) >= target_count:
            break
        coll_datasets = valid_df[valid_df['collection_id'] == coll_id].iloc[:1]
        selected.append(coll_datasets)

    selected_df = pd.concat(selected, ignore_index=True) if selected else pd.DataFrame()

    # Second pass: fill remaining slots with smallest datasets
    if len(selected_df) < target_count:
        remaining = valid_df[~valid_df['dataset_id'].isin(selected_df['dataset_id'])]
        remaining = remaining.sort_values('file_size_bytes')
        needed = target_count - len(selected_df)
        selected_df = pd.concat([selected_df, remaining.head(needed)], ignore_index=True)

    # Re-sort by size
    selected_df = selected_df.sort_values('file_size_bytes')

    return selected_df


def populate_database(conn, df):
    """Insert dataset information into the database."""
    cursor = conn.cursor()

    count = 0
    for _, row in df.iterrows():
        try:
            cursor.execute("""
                INSERT OR REPLACE INTO datasets (
                    dataset_id, dataset_name, description, collection_name,
                    collection_doi, file_size_bytes, file_size_mb,
                    n_cells, n_features, h5ad_url
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                row['dataset_id'],
                row['dataset_title'],
                '',  # description - will be populated later
                row['collection_name'],
                row['collection_doi'],
                int(row['file_size_bytes']),
                float(row['file_size_mb']),
                int(row['dataset_total_cell_count']),
                None,  # n_features - will be populated after download
                f"{BASE_DOWNLOAD_URL}/{row['dataset_version_id']}.h5ad"
            ))
            count += 1
        except Exception as e:
            logger.error(f"Error inserting dataset {row['dataset_id']}: {e}")

    conn.commit()
    logger.info(f"Inserted {count} datasets into database")


def export_to_csv(conn):
    """Export database to CSV file."""
    df = pd.read_sql_query("""
        SELECT
            dataset_id,
            dataset_name,
            description,
            collection_name,
            collection_doi,
            ROUND(file_size_mb, 2) as size_mb,
            n_cells,
            n_features,
            h5ad_url,
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

    # Load existing metadata
    df = load_metadata()

    # Fetch file sizes (this is the time-consuming part)
    logger.info("Fetching file sizes via HTTP HEAD requests (this may take a few minutes)...")
    size_map = fetch_file_sizes(df, max_workers=20)

    # Filter and select datasets
    selected_df = filter_and_select_datasets(df, size_map, TARGET_DATASETS)
    logger.info(f"Selected {len(selected_df)} datasets for benchmarking")

    if len(selected_df) == 0:
        logger.error("No valid datasets found!")
        return

    # Populate database
    populate_database(conn, selected_df)

    # Export to CSV
    result_df = export_to_csv(conn)

    # Print summary
    print("\n" + "="*80)
    print("DATASET DISCOVERY SUMMARY")
    print("="*80)
    print(f"Total datasets in metadata: {len(df)}")
    print(f"Datasets with valid size info: {sum(1 for s in size_map.values() if s and s > 0)}")
    print(f"Datasets under 1GB: {len(selected_df)}")
    print(f"Selected for benchmarking: {len(result_df)}")
    print(f"\nDatabase: {DB_PATH}")
    print(f"CSV Export: {CSV_PATH}")
    print(f"\nSize distribution (MB):")
    print(result_df['size_mb'].describe())
    print(f"\nSmallest 5 datasets:")
    print(result_df[['dataset_name', 'size_mb', 'n_cells']].head())
    print(f"\nLargest 5 datasets (still under 1GB):")
    print(result_df[['dataset_name', 'size_mb', 'n_cells']].tail())

    conn.close()
    logger.info("Dataset discovery complete!")


if __name__ == "__main__":
    main()
