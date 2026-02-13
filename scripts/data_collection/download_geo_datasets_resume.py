#!/usr/bin/env python3
"""
Resume GEO dataset download - skips already attempted datasets.
"""
import sqlite3
import json
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Import from main script
from download_geo_datasets import process_geo_dataset

# Paths
data_dir = Path(__file__).parent.parent / 'data'
db_path = data_dir / 'papers' / 'metadata' / 'papers.db'
output_dir = Path("/nfs/roberts/project/pi_sk2433/shared/llm-analyze-geo-datasets")
geo_raw_dir = output_dir / 'raw'
geo_processed_dir = output_dir / 'processed'

# Create directories
geo_raw_dir.mkdir(parents=True, exist_ok=True)
geo_processed_dir.mkdir(parents=True, exist_ok=True)

# Set global variables for process_geo_dataset
import download_geo_datasets
download_geo_datasets.geo_raw_dir = geo_raw_dir
download_geo_datasets.geo_processed_dir = geo_processed_dir

def main():
    # Get datasets that have already been attempted
    attempted = set()
    for d in geo_raw_dir.glob('GSE*'):
        if d.is_dir():
            attempted.add(d.name)

    # Get processed datasets
    processed = set()
    for f in geo_processed_dir.glob('*.h5ad'):
        processed.add(f.stem)

    print("="*80)
    print(f"Already attempted: {len(attempted)}")
    print(f"Successfully processed: {len(processed)}")
    print("="*80)

    # Get papers with suitable datasets
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    cursor.execute("""
        SELECT geo_accessions, download_urls
        FROM papers
        WHERE has_suitable_format = 1
        AND geo_accessions IS NOT NULL
    """)

    # Build list of datasets to process
    datasets_to_process = []
    for row in cursor.fetchall():
        if not row[0] or not row[1]:
            continue

        geo_accs = json.loads(row[0]) if isinstance(row[0], str) else row[0]
        download_urls = json.loads(row[1]) if isinstance(row[1], str) else row[1]

        if not isinstance(geo_accs, list):
            geo_accs = [geo_accs]

        # Convert download_urls to dict if it's a list
        if isinstance(download_urls, list):
            download_urls = {}

        for geo_acc in geo_accs:
            # Skip if already attempted OR already processed
            if geo_acc in attempted or geo_acc in processed:
                continue

            datasets_to_process.append((geo_acc, download_urls.get(geo_acc, [])))

    conn.close()

    print(f"Datasets to process: {len(datasets_to_process)}")
    print(f"Target: 100 total datasets")
    print(f"Need: {max(0, 100 - len(processed))} more successful conversions")
    print("="*80)

    # Process datasets
    stats = {'successful': 0, 'failed': 0}

    for i, (geo_acc, urls) in enumerate(datasets_to_process[:100], 1):
        print(f"\n[{i}/{min(100, len(datasets_to_process))}] {geo_acc}")

        try:
            result = process_geo_dataset(geo_acc, urls)

            if result['success']:
                stats['successful'] += 1
                print(f"  ✓ SUCCESS: {result['n_cells']} cells × {result['n_genes']} genes")
            else:
                stats['failed'] += 1
                print(f"  ✗ FAILED: {result.get('error', 'Unknown error')}")
        except KeyboardInterrupt:
            print("\n\nInterrupted by user")
            break
        except Exception as e:
            stats['failed'] += 1
            print(f"  ✗ EXCEPTION: {str(e)}")

    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Successfully converted (this run): {stats['successful']}")
    print(f"Failed (this run): {stats['failed']}")
    print(f"Total processed datasets: {len(list(geo_processed_dir.glob('*.h5ad')))}")
    print(f"Processed datasets saved to: {geo_processed_dir}")
    print("="*80)

if __name__ == '__main__':
    main()
