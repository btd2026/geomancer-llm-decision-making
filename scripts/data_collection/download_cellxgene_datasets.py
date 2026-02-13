#!/usr/bin/env python3
"""
Download top datasets from CELLxGENE Census.
"""
import pandas as pd
import cellxgene_census
from pathlib import Path
import time

# Paths
datasets_file = Path('/home/btd8/llm-paper-analyze/datasets_to_download.csv')
output_dir = Path('/nfs/roberts/project/pi_sk2433/shared/llm-analyze-geo-datasets/processed')

# Ensure output directory exists
output_dir.mkdir(parents=True, exist_ok=True)

# Read dataset list
df = pd.read_csv(datasets_file)

print("=" * 80)
print("CELLXGENE CENSUS - DATASET DOWNLOAD")
print("=" * 80)
print(f"Output directory: {output_dir}")
print(f"Datasets to download: {len(df)}")
print("=" * 80)

# Stats
stats = {'successful': 0, 'failed': 0, 'skipped': 0}

for i, row in df.iterrows():
    dataset_id = row['dataset_id']
    title = row['dataset_title']
    cell_count = row['dataset_total_cell_count']

    # Output filename - use first 50 chars of title + dataset_id for uniqueness
    safe_title = "".join(c if c.isalnum() or c in (' ', '_', '-') else '_' for c in title)
    safe_title = safe_title[:50].strip().replace(' ', '_')
    output_file = output_dir / f"{safe_title}_{dataset_id[:8]}.h5ad"

    print(f"\n[{i+1}/{len(df)}] {dataset_id}")
    print(f"  Title: {title[:70]}")
    print(f"  Cells: {cell_count:,}")

    # Skip if already exists
    if output_file.exists():
        print(f"  ⊙ Already exists: {output_file.name}")
        stats['skipped'] += 1
        continue

    # Download
    try:
        print(f"  Downloading...")
        start_time = time.time()

        cellxgene_census.download_source_h5ad(
            dataset_id=dataset_id,
            to_path=str(output_file),
            census_version="2025-01-30"
        )

        duration = time.time() - start_time
        size_mb = output_file.stat().st_size / (1024**2)

        print(f"  ✓ SUCCESS: {size_mb:.1f} MB downloaded in {duration:.1f}s")
        stats['successful'] += 1

    except Exception as e:
        print(f"  ✗ FAILED: {str(e)}")
        stats['failed'] += 1

        # Clean up partial file if exists
        if output_file.exists():
            output_file.unlink()

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"Successfully downloaded: {stats['successful']}")
print(f"Failed: {stats['failed']}")
print(f"Skipped (already exist): {stats['skipped']}")
print(f"Total in output directory: {len(list(output_dir.glob('*.h5ad')))}")
print(f"\nOutput directory: {output_dir}")
print("=" * 80)
