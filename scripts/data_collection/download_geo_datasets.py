#!/usr/bin/env python3
"""
Download GEO datasets and convert them to h5ad format for ManyLatents benchmarking.

Supports:
- 10x CellRanger format (matrix.mtx, barcodes.tsv, features.tsv) -> h5ad
- h5 format (10x HDF5) -> h5ad  
- h5ad format -> use directly
- TAR archives -> extract and convert

Requires: scanpy, anndata, scipy, pandas
Install with: pip install scanpy anndata
"""
import json
import sqlite3
import urllib.request
import tarfile
import gzip
import shutil
from pathlib import Path
from datetime import datetime
import os
import sys

try:
    import scanpy as sc
    import anndata as ad
    import pandas as pd
    import numpy as np
    HAS_SCANPY = True
except ImportError:
    HAS_SCANPY = False
    print("ERROR: scanpy not installed. Install with: pip install scanpy anndata")
    sys.exit(1)

# Paths
data_dir = Path(__file__).parent.parent / 'data'
db_path = data_dir / 'papers' / 'metadata' / 'papers.db'

# Default directories (can be overridden by command-line args)
default_geo_raw_dir = data_dir / 'geo' / 'raw'
default_geo_processed_dir = data_dir / 'geo' / 'processed'

# These will be set in main() based on args
geo_raw_dir = None
geo_processed_dir = None

def download_file(url, dest_path, show_progress=True):
    """
    Download a file from URL to destination path.
    """
    if dest_path.exists():
        print(f"    File already exists: {dest_path.name}")
        return True

    try:
        print(f"    Downloading: {url}")
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})

        with urllib.request.urlopen(req, timeout=600) as response:
            total_size = int(response.headers.get('content-length', 0))

            with open(dest_path, 'wb') as f:
                downloaded = 0
                chunk_size = 8192

                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)

                    if show_progress and total_size > 0:
                        percent = (downloaded / total_size) * 100
                        print(f"\r    Progress: {percent:.1f}% ({downloaded/1024/1024:.1f} MB)", end='')

                if show_progress:
                    print()

        print(f"    ✓ Downloaded: {dest_path.name}")
        return True

    except Exception as e:
        print(f"    ✗ Download failed: {e}")
        if dest_path.exists():
            dest_path.unlink()
        return False

def extract_tar(tar_path, extract_dir):
    """Extract TAR archive."""
    print(f"    Extracting: {tar_path.name}")
    extract_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Auto-detect compression (mode='r:*' tries all formats)
        with tarfile.open(tar_path, 'r:*') as tar:
            members = tar.getmembers()
            print(f"    Found {len(members)} files in archive")
            tar.extractall(extract_dir)
            extracted = [extract_dir / m.name for m in members if m.isfile()]
            print(f"    ✓ Extracted {len(extracted)} files")
            return extracted

    except Exception as e:
        print(f"    ✗ Extraction failed: {e}")
        import traceback
        traceback.print_exc()
        return []

def find_10x_files(directory):
    """Find 10x CellRanger format files in a directory."""
    directory = Path(directory)
    result = {'filtered': {}, 'raw': {}}

    for filepath in directory.rglob('*'):
        if not filepath.is_file():
            continue

        name = filepath.name.lower()
        dataset_type = 'filtered' if 'filtered' in name else 'raw'

        if 'matrix.mtx' in name:
            result[dataset_type]['matrix'] = filepath
        elif 'barcode' in name and '.tsv' in name:
            result[dataset_type]['barcodes'] = filepath
        elif ('feature' in name or 'gene' in name) and '.tsv' in name:
            result[dataset_type]['features'] = filepath

    if all(k in result['filtered'] for k in ['matrix', 'barcodes']):
        return result['filtered']
    elif all(k in result['raw'] for k in ['matrix', 'barcodes']):
        return result['raw']
    else:
        return {}

def convert_10x_to_h5ad(matrix_path, barcodes_path, features_path, output_path):
    """Convert 10x CellRanger format to h5ad."""
    try:
        print(f"    Converting 10x format to h5ad...")
        parent_dir = matrix_path.parent
        temp_dir = parent_dir / 'temp_10x'
        temp_dir.mkdir(exist_ok=True)

        # Create symlinks with expected names
        (temp_dir / 'matrix.mtx.gz' if matrix_path.suffix == '.gz' else temp_dir / 'matrix.mtx').symlink_to(matrix_path)
        (temp_dir / 'barcodes.tsv.gz' if barcodes_path.suffix == '.gz' else temp_dir / 'barcodes.tsv').symlink_to(barcodes_path)

        features_link_name = 'features.tsv.gz' if features_path.suffix == '.gz' else 'features.tsv'
        if not (temp_dir / features_link_name).exists():
            (temp_dir / features_link_name).symlink_to(features_path)

        adata = sc.read_10x_mtx(temp_dir, var_names='gene_symbols', make_unique=True)
        shutil.rmtree(temp_dir)

        adata.uns['source'] = '10x_cellranger'
        adata.uns['conversion_date'] = datetime.now().isoformat()
        adata.write_h5ad(output_path)

        print(f"    ✓ Converted to h5ad: {adata.n_obs} cells × {adata.n_vars} genes")
        return True

    except Exception as e:
        print(f"    ✗ Conversion failed: {e}")
        return False

def convert_h5_to_h5ad(h5_path, output_path):
    """Convert 10x HDF5 format to h5ad."""
    try:
        print(f"    Converting h5 to h5ad...")
        adata = sc.read_10x_h5(h5_path)
        adata.uns['source'] = '10x_h5'
        adata.uns['conversion_date'] = datetime.now().isoformat()
        adata.write_h5ad(output_path)

        print(f"    ✓ Converted to h5ad: {adata.n_obs} cells × {adata.n_vars} genes")
        return True

    except Exception as e:
        print(f"    ✗ Conversion failed: {e}")
        return False

def process_geo_dataset(geo_accession, download_urls):
    """Download and process a single GEO dataset."""
    result = {
        'success': False,
        'h5ad_path': None,
        'n_cells': None,
        'n_genes': None,
        'error': None
    }

    dataset_raw_dir = geo_raw_dir / geo_accession
    dataset_raw_dir.mkdir(parents=True, exist_ok=True)

    h5ad_output = geo_processed_dir / f"{geo_accession}.h5ad"

    if h5ad_output.exists():
        print(f"  ✓ Already processed: {h5ad_output}")
        try:
            adata = ad.read_h5ad(h5ad_output)
            result['success'] = True
            result['h5ad_path'] = h5ad_output
            result['n_cells'] = adata.n_obs
            result['n_genes'] = adata.n_vars
        except:
            pass
        return result

    print(f"\n  Processing {geo_accession}...")

    # Download files
    downloaded_files = []
    for url in download_urls:
        if not url:
            continue
        filename = url.split('/')[-1]
        dest_path = dataset_raw_dir / filename

        if download_file(url, dest_path):
            downloaded_files.append(dest_path)

    if not downloaded_files:
        result['error'] = "No files downloaded"
        return result

    # Extract TAR files
    extracted_dir = dataset_raw_dir / 'extracted'
    for filepath in downloaded_files:
        if filepath.suffix == '.tar' or filepath.name.endswith('.tar.gz'):
            extract_tar(filepath, extracted_dir)

    # Look for processable files
    search_dirs = [dataset_raw_dir, extracted_dir] if extracted_dir.exists() else [dataset_raw_dir]

    for search_dir in search_dirs:
        # Check for h5ad files
        h5ad_files = list(search_dir.rglob('*.h5ad'))
        if h5ad_files:
            shutil.copy(h5ad_files[0], h5ad_output)
            try:
                adata = ad.read_h5ad(h5ad_output)
                result['success'] = True
                result['h5ad_path'] = h5ad_output
                result['n_cells'] = adata.n_obs
                result['n_genes'] = adata.n_vars
                return result
            except Exception as e:
                result['error'] = f"Failed to read h5ad: {e}"
                return result

        # Check for h5 files
        h5_files = [f for f in search_dir.rglob('*.h5') if not f.name.endswith('.h5ad')]
        if h5_files and convert_h5_to_h5ad(h5_files[0], h5ad_output):
            try:
                adata = ad.read_h5ad(h5ad_output)
                result['success'] = True
                result['h5ad_path'] = h5ad_output
                result['n_cells'] = adata.n_obs
                result['n_genes'] = adata.n_vars
                return result
            except Exception as e:
                result['error'] = f"Failed to read converted h5ad: {e}"
                return result

        # Check for 10x format
        files_10x = find_10x_files(search_dir)
        if files_10x and 'matrix' in files_10x and 'barcodes' in files_10x:
            features = files_10x.get('features')
            if features and convert_10x_to_h5ad(
                files_10x['matrix'],
                files_10x['barcodes'],
                features,
                h5ad_output
            ):
                try:
                    adata = ad.read_h5ad(h5ad_output)
                    result['success'] = True
                    result['h5ad_path'] = h5ad_output
                    result['n_cells'] = adata.n_obs
                    result['n_genes'] = adata.n_vars
                    return result
                except Exception as e:
                    result['error'] = f"Failed to read converted h5ad: {e}"
                    return result

    result['error'] = "No suitable files found for conversion"
    return result

def main():
    """Main execution."""
    import argparse
    global geo_raw_dir, geo_processed_dir

    parser = argparse.ArgumentParser(description='Download and convert GEO datasets to h5ad format')
    parser.add_argument('--geo-accession', help='Specific GEO accession to download')
    parser.add_argument('--limit', type=int, default=3, help='Limit number of datasets to download (default: 3)')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be downloaded')
    parser.add_argument('--output-dir', type=str, help='Custom output directory for processed datasets')

    args = parser.parse_args()

    # Set output directories
    if args.output_dir:
        output_base = Path(args.output_dir)
        geo_raw_dir = output_base / 'raw'
        geo_processed_dir = output_base / 'processed'
    else:
        geo_raw_dir = default_geo_raw_dir
        geo_processed_dir = default_geo_processed_dir

    # Create directories
    geo_raw_dir.mkdir(parents=True, exist_ok=True)
    geo_processed_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("GEO DATASET DOWNLOAD AND CONVERSION")
    print("=" * 80)
    print(f"Output directory: {geo_processed_dir}")
    print("=" * 80)

    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    query = """
        SELECT DISTINCT geo_accessions, download_urls
        FROM papers
        WHERE has_suitable_format = 1
    """

    cursor.execute(query)
    papers = cursor.fetchall()
    conn.close()

    print(f"Found {len(papers)} paper(s) with suitable datasets")

    # Collect all unique GSE accessions
    all_gse = []
    url_map = {}

    for geo_json, urls_json in papers:
        geo_accessions = json.loads(geo_json) if geo_json else []
        download_urls = json.loads(urls_json) if urls_json else []

        for geo_acc in geo_accessions:
            if geo_acc.startswith('GSE') and geo_acc not in all_gse:
                all_gse.append(geo_acc)
                url_map[geo_acc] = [u for u in download_urls if geo_acc in u]

    print(f"Total unique GEO datasets: {len(all_gse)}")

    if args.geo_accession:
        all_gse = [g for g in all_gse if g == args.geo_accession]

    if args.limit:
        all_gse = all_gse[:args.limit]

    print(f"Will process: {len(all_gse)} dataset(s)")
    print("=" * 80)

    stats = {'successful': 0, 'failed': 0, 'already_processed': 0}

    for i, geo_acc in enumerate(all_gse, 1):
        print(f"\n[{i}/{len(all_gse)}] {geo_acc}")

        if args.dry_run:
            print(f"  [DRY RUN] Would download: {geo_acc}")
            continue

        try:
            result = process_geo_dataset(geo_acc, url_map.get(geo_acc, []))

            if result['success']:
                if geo_processed_dir / f"{geo_acc}.h5ad" in list(geo_processed_dir.glob('*.h5ad')):
                    stats['successful'] += 1
                print(f"  ✓ SUCCESS: {result['n_cells']} cells × {result['n_genes']} genes")
            else:
                stats['failed'] += 1
                print(f"  ✗ FAILED: {result.get('error', 'Unknown error')}")
        except Exception as e:
            stats['failed'] += 1
            print(f"  ✗ EXCEPTION: {str(e)}")
            import traceback
            traceback.print_exc()
            continue

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Successfully converted: {stats['successful']}")
    print(f"Failed: {stats['failed']}")
    print(f"Raw data saved to: {geo_raw_dir}")
    print(f"Processed datasets saved to: {geo_processed_dir}")
    print("=" * 80)

if __name__ == "__main__":
    main()
