#!/usr/bin/env python3
"""
Subsample large H5AD datasets to make them manageable for PHATE.

For datasets with >50,000 cells, randomly sample 50,000 cells.
This makes memory requirements predictable and reasonable.
"""

import scanpy as sc
import numpy as np
from pathlib import Path
import argparse
import sys

def subsample_dataset(input_path, output_dir, max_cells=50000, seed=42):
    """Subsample a dataset if it's too large."""
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / input_path.name

    # Skip if already processed
    if output_path.exists():
        print(f"✓ Already exists: {input_path.name}")
        return "skipped"

    try:
        # Read dataset
        print(f"Reading: {input_path.name}...")
        adata = sc.read_h5ad(input_path)
        n_cells = adata.n_obs

        # Subsample if needed
        if n_cells > max_cells:
            print(f"  {n_cells:,} cells -> subsampling to {max_cells:,}")
            np.random.seed(seed)
            indices = np.random.choice(n_cells, size=max_cells, replace=False)
            indices = np.sort(indices)  # Keep order for reproducibility
            adata = adata[indices, :].copy()
        else:
            print(f"  {n_cells:,} cells -> keeping all")

        # Save
        print(f"  Saving to: {output_path.name}")
        adata.write_h5ad(output_path)
        print(f"✓ Completed: {input_path.name}")
        return "success"

    except Exception as e:
        print(f"✗ Error processing {input_path.name}: {e}")
        return "error"

def main():
    parser = argparse.ArgumentParser(description="Subsample large H5AD datasets")
    parser.add_argument("--input-dir", type=str, required=True, help="Input directory with H5AD files")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory for subsampled files")
    parser.add_argument("--max-cells", type=int, default=50000, help="Maximum cells per dataset (default: 50000)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--file", type=str, help="Process single file instead of all")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if args.file:
        # Process single file
        input_path = input_dir / args.file
        if not input_path.exists():
            print(f"Error: File not found: {input_path}")
            sys.exit(1)
        files = [input_path]
    else:
        # Process all files
        files = sorted(list(input_dir.glob("*.h5ad")))

    print(f"Found {len(files)} H5AD files")
    print(f"Max cells per dataset: {args.max_cells:,}")
    print(f"Output directory: {output_dir}")
    print("=" * 60)

    results = {"success": 0, "skipped": 0, "error": 0}

    for i, file_path in enumerate(files, 1):
        print(f"\n[{i}/{len(files)}]")
        result = subsample_dataset(file_path, output_dir, args.max_cells, args.seed)
        results[result] += 1

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Success:  {results['success']}")
    print(f"Skipped:  {results['skipped']}")
    print(f"Errors:   {results['error']}")
    print(f"Total:    {len(files)}")

if __name__ == "__main__":
    main()
