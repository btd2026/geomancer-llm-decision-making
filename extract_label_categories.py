#!/usr/bin/env python3
"""
Extract actual label category names from h5ad files.

This script reads h5ad files and extracts the actual category names
from the obs column that was used for coloring PHATE plots.
Output: JSON file mapping dataset_id -> list of actual category names.
"""

import argparse
import json
import sys
from pathlib import Path

# Add anndata to path
sys.path.insert(0, '/home/btd8/.conda/envs/claude-env/lib/python3.11/site-packages')

try:
    import anndata as ad
    import h5py
except ImportError as e:
    print(f"Error: {e}")
    print("Installing required packages...")
    import subprocess
    subprocess.check_call(['pip', 'install', 'anndata', 'h5py', '-q'])
    import anndata as ad
    import h5py


def get_label_key_from_config(wandb_metadata_path: str, run_id: str) -> str:
    """Get the label_key used for a run from W&B metadata."""
    try:
        with open(wandb_metadata_path, 'r') as f:
            wandb_data = json.load(f)

        run_info = wandb_data.get(run_id, {})
        config = run_info.get('config', {})
        data_config = config.get('data', {}).get('value', {})

        label_key = data_config.get('label_key', None)
        return label_key
    except Exception as e:
        print(f"Warning: Could not get label_key for {run_id}: {e}")
        return None


def extract_label_categories(h5ad_path: str, label_key: str, max_categories: int = 50) -> list:
    """
    Extract unique category values from h5ad file for a given label_key.

    Args:
        h5ad_path: Path to h5ad file
        label_key: Column name in obs to extract categories from
        max_categories: Maximum number of categories to return (for sanity)

    Returns:
        List of unique category values
    """
    try:
        adata = ad.read_h5ad(h5ad_path)

        # Check if label_key exists in obs
        if label_key not in adata.obs:
            return None

        # Get unique values
        categories = adata.obs[label_key].unique()

        # Convert to strings and filter out NaN/None
        result = []
        for cat in categories:
            cat_str = str(cat)
            if cat_str not in ['nan', 'None', 'NA', '<NA>']:
                result.append(cat_str)

        # Limit to max_categories for sanity
        if len(result) > max_categories:
            result = result[:max_categories]

        return result

    except Exception as e:
        print(f"Error processing {h5ad_path}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description='Extract label categories from h5ad files')
    parser.add_argument('--wandb-metadata', type=str,
                        default='/home/btd8/geomancer-llm-decision-making/wandb_gallery_replit/wandb_metadata.json',
                        help='Path to W&B metadata JSON file')
    parser.add_argument('--gallery-data', type=str,
                        default='/home/btd8/geomancer-llm-decision-making/wandb_gallery_replit/gallery_data.json',
                        help='Path to gallery_data JSON file')
    parser.add_argument('--subsampled-dir', type=str,
                        default='/nfs/roberts/project/pi_sk2433/shared/Geomancer_2025_Data/subsampled',
                        help='Directory containing subsampled h5ad files')
    parser.add_argument('--output', type=str,
                        default='/home/btd8/geomancer-llm-decision-making/wandb_gallery_replit/label_categories.json',
                        help='Output JSON file with label categories')
    parser.add_argument('--max-categories', type=int, default=50,
                        help='Maximum categories per dataset (default: 50)')

    args = parser.parse_args()

    # Load W&B metadata to get all run IDs and their label_keys
    with open(args.wandb_metadata, 'r') as f:
        wandb_metadata = json.load(f)

    # Load gallery_data to see what datasets we need
    with open(args.gallery_data, 'r') as f:
        gallery_data = json.load(f)

    # Build mapping from adata filename to run_id
    # Gallery may have dataset_info with runs, or run_specific_info
    filename_to_run_id = {}

    # Method 1: Extract from run_specific_info if available
    if 'run_specific_info' in gallery_data:
        for run_id, info in gallery_data['run_specific_info'].items():
            if 'image_path' in info:
                # Extract filename from image_path like "images/abc123.png"
                image_path = info['image_path']
                filename = Path(image_path).stem  # removes .png
                filename_to_run_id[filename] = run_id

    # Method 2: Extract from dataset_info groups
    if 'dataset_info' in gallery_data:
        for label_group, group_info in gallery_data['dataset_info'].items():
            for run_id in group_info.get('runs', []):
                filename_to_run_id[run_id] = run_id

    print(f"Found {len(filename_to_run_id)} run IDs in gallery data")

    # Extract categories for each run
    results = {}
    not_found = []

    for run_id, run_data in wandb_metadata.items():
        # Get the label_key used for this run
        label_key = get_label_key_from_config(args.wandb_metadata, run_id)

        if label_key is None:
            print(f"Warning: No label_key found for {run_id}")
            continue

        # Find the corresponding h5ad file
        # Run ID matches filename (e.g., "hgftnuic" -> "hgftnuic.h5ad")
        h5ad_filename = f"{run_id}.h5ad"
        h5ad_path = Path(args.subsampled_dir) / h5ad_filename

        if not h5ad_path.exists():
            # Try alternative naming patterns
            # Some datasets might have different naming
            found = False
            for h5ad_file in Path(args.subsampled_dir).glob("*.h5ad"):
                if run_id in h5ad_file.stem or h5ad_file.stem.replace('-', '_'):
                    h5ad_path = h5ad_file
                    found = True
                    break

            if not found:
                not_found.append(run_id)
                continue

        # Extract categories
        categories = extract_label_categories(str(h5ad_path), label_key, args.max_categories)

        if categories:
            results[run_id] = {
                'label_key': label_key,
                'categories': categories,
                'num_categories': len(categories),
                'dataset_name': run_data.get('dataset_name', 'Unknown')
            }
            print(f"{run_id}: {label_key} -> {len(categories)} categories")
        else:
            print(f"Warning: Could not extract categories for {run_id}")
            not_found.append(run_id)

    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved label categories for {len(results)} runs to {args.output}")

    if not_found:
        print(f"\nWarning: Could not find h5ad files for {len(not_found)} runs:")
        for run_id in not_found[:10]:
            print(f"  - {run_id}")
        if len(not_found) > 10:
            print(f"  ... and {len(not_found) - 10} more")


if __name__ == '__main__':
    main()
