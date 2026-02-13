#!/usr/bin/env python3
"""
Download H5AD files for the 100 small datasets from CELLxGENE.
Uses parallel downloads with progress tracking.
"""

import sqlite3
import requests
import pandas as pd
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import time
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Paths
DB_PATH = Path("/home/btd8/llm-paper-analyze/data/manylatents_benchmark/manylatents_datasets.db")
DOWNLOAD_DIR = Path("/nfs/roberts/project/pi_sk2433/shared/Geomancer_2025_Data/manylatents_small_datasets")
LOG_FILE = Path("/home/btd8/llm-paper-analyze/logs/download_small_datasets.log")

# Download settings
MAX_WORKERS = 5  # Parallel downloads
CHUNK_SIZE = 8192 * 1024  # 8MB chunks
TIMEOUT = 300  # 5 minute timeout per file


def setup_file_logging():
    """Add file handler for logging."""
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(LOG_FILE)
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(fh)


def get_datasets_to_download():
    """Get list of datasets that haven't been downloaded yet."""
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("""
        SELECT dataset_id, dataset_name, h5ad_url, file_size_mb
        FROM datasets
        WHERE downloaded = 0
        ORDER BY file_size_mb ASC
    """, conn)
    conn.close()
    return df


def download_file(dataset_id, url, dest_path):
    """Download a single H5AD file."""
    try:
        response = requests.get(url, stream=True, timeout=TIMEOUT)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0

        with open(dest_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)

        return True, downloaded
    except Exception as e:
        logger.error(f"Error downloading {dataset_id}: {e}")
        if dest_path.exists():
            dest_path.unlink()  # Clean up partial download
        return False, str(e)


def update_database(dataset_id, downloaded=True):
    """Update the downloaded status in the database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        UPDATE datasets
        SET downloaded = ?
        WHERE dataset_id = ?
    """, (1 if downloaded else 0, dataset_id))
    conn.commit()
    conn.close()


def download_dataset(row):
    """Download a single dataset and update status."""
    dataset_id = row['dataset_id']
    url = row['h5ad_url']
    name = row['dataset_name']
    size_mb = row['file_size_mb']

    # Extract filename from URL
    filename = url.split('/')[-1]
    dest_path = DOWNLOAD_DIR / filename

    # Check if already downloaded
    if dest_path.exists():
        logger.info(f"Already exists: {dataset_id} ({size_mb:.1f} MB)")
        update_database(dataset_id, True)
        return dataset_id, True, "Already exists"

    logger.info(f"Downloading: {name[:50]}... ({size_mb:.1f} MB)")
    start_time = time.time()

    success, result = download_file(dataset_id, url, dest_path)

    if success:
        elapsed = time.time() - start_time
        speed = size_mb / elapsed if elapsed > 0 else 0
        logger.info(f"Completed: {dataset_id} ({size_mb:.1f} MB in {elapsed:.1f}s, {speed:.1f} MB/s)")
        update_database(dataset_id, True)
        return dataset_id, True, f"{elapsed:.1f}s"
    else:
        logger.error(f"Failed: {dataset_id} - {result}")
        return dataset_id, False, result


def main():
    """Main download function."""
    setup_file_logging()

    logger.info("="*60)
    logger.info("Starting download of small CELLxGENE datasets")
    logger.info(f"Download directory: {DOWNLOAD_DIR}")
    logger.info("="*60)

    # Get datasets to download
    df = get_datasets_to_download()
    total_datasets = len(df)
    total_size = df['file_size_mb'].sum()

    logger.info(f"Datasets to download: {total_datasets}")
    logger.info(f"Total size: {total_size:.1f} MB ({total_size/1024:.2f} GB)")

    if total_datasets == 0:
        logger.info("All datasets already downloaded!")
        return

    # Create download directory
    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

    # Download with progress tracking
    start_time = time.time()
    successful = 0
    failed = 0

    # Use ThreadPoolExecutor for parallel downloads
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(download_dataset, row): row['dataset_id']
            for _, row in df.iterrows()
        }

        for i, future in enumerate(as_completed(futures), 1):
            dataset_id = futures[future]
            try:
                did, success, msg = future.result()
                if success:
                    successful += 1
                else:
                    failed += 1
            except Exception as e:
                logger.error(f"Exception for {dataset_id}: {e}")
                failed += 1

            # Progress update
            if i % 10 == 0:
                elapsed = time.time() - start_time
                logger.info(f"Progress: {i}/{total_datasets} ({successful} successful, {failed} failed) - {elapsed/60:.1f} min elapsed")

    # Final summary
    elapsed = time.time() - start_time
    logger.info("="*60)
    logger.info("DOWNLOAD SUMMARY")
    logger.info("="*60)
    logger.info(f"Total time: {elapsed/60:.1f} minutes")
    logger.info(f"Successful: {successful}/{total_datasets}")
    logger.info(f"Failed: {failed}/{total_datasets}")
    logger.info(f"Download location: {DOWNLOAD_DIR}")

    # List downloaded files
    files = list(DOWNLOAD_DIR.glob("*.h5ad"))
    logger.info(f"Files in directory: {len(files)}")
    total_downloaded = sum(f.stat().st_size for f in files) / (1024**3)
    logger.info(f"Total size on disk: {total_downloaded:.2f} GB")

    print(f"\nDownload complete!")
    print(f"Location: {DOWNLOAD_DIR}")
    print(f"Files: {len(files)}")
    print(f"Total size: {total_downloaded:.2f} GB")


if __name__ == "__main__":
    main()
