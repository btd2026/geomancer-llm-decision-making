#!/usr/bin/env python3
"""
Download a GEO dataset and convert to AnnData format for manylatents.

This script:
1. Downloads raw data from GEO using GEOparse
2. Converts to AnnData (.h5ad) format
3. Performs basic quality control
4. Saves to data/geo/processed/
"""

import sys
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

def download_geo_dataset(geo_id, output_dir):
    """
    Download a GEO dataset and convert to AnnData format.

    Args:
        geo_id: GEO accession ID (e.g., 'GSE152048')
        output_dir: Directory to save processed data
    """
    print(f"="*80)
    print(f"Downloading GEO Dataset: {geo_id}")
    print(f"="*80)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Output path
    output_path = output_dir / f"{geo_id}.h5ad"

    if output_path.exists():
        print(f"\n✅ Dataset already exists at: {output_path}")
        return str(output_path)

    try:
        import GEOparse
        import scanpy as sc
        import anndata as ad

        print(f"\n📥 Step 1: Downloading from GEO...")
        print(f"   This may take several minutes...")

        # Download GEO dataset
        gse = GEOparse.get_GEO(geo=geo_id, destdir=str(output_dir / "raw"))

        print(f"   ✅ Downloaded {geo_id}")
        print(f"   Title: {gse.metadata.get('title', ['Unknown'])[0]}")
        print(f"   Samples: {len(gse.gsms)}")
        print(f"   Platform: {list(gse.gpls.keys())}")

        # For single-cell data, we need to handle it differently
        # This is a simplified approach - may need customization per dataset

        print(f"\n📊 Step 2: Converting to AnnData format...")

        # Try to extract expression matrix
        # This is dataset-specific - different GEO datasets have different structures

        # Approach 1: Try supplementary files (common for scRNA-seq)
        supp_files = gse.download_supplementary_files(directory=str(output_dir / "raw"), download_sra=False)

        if supp_files:
            print(f"   Found {len(supp_files)} supplementary files")

            # Look for common scRNA-seq file formats
            h5_files = list((output_dir / "raw").rglob("*.h5"))
            h5ad_files = list((output_dir / "raw").rglob("*.h5ad"))
            mtx_files = list((output_dir / "raw").rglob("*.mtx*"))

            if h5ad_files:
                print(f"   ✅ Found .h5ad file: {h5ad_files[0]}")
                # Just copy it
                import shutil
                shutil.copy(h5ad_files[0], output_path)
                adata = ad.read_h5ad(output_path)

            elif h5_files:
                print(f"   ✅ Found .h5 file: {h5_files[0]}")
                # Try to read as 10x format
                try:
                    adata = sc.read_10x_h5(h5_files[0])
                    print(f"   Successfully read as 10x format")
                except:
                    print(f"   ⚠️  Not 10x format, trying generic h5 read...")
                    # Fallback to generic read
                    adata = ad.read_h5ad(h5_files[0])

            elif mtx_files:
                print(f"   ✅ Found .mtx file: {mtx_files[0]}")
                # Read matrix market format
                mtx_dir = mtx_files[0].parent
                adata = sc.read_10x_mtx(mtx_dir)

            else:
                print(f"   ⚠️  No standard scRNA-seq files found")
                print(f"   Available files: {list((output_dir / 'raw').rglob('*'))[:10]}")
                raise ValueError("Could not find compatible data files")

        else:
            # Approach 2: Build from sample data (for older GEO datasets)
            print(f"   No supplementary files, trying sample data...")

            # Extract expression data from samples
            sample_data = []
            for gsm_name, gsm in gse.gsms.items():
                if hasattr(gsm, 'table'):
                    sample_data.append(gsm.table)

            if sample_data:
                # Combine samples
                expr_matrix = pd.concat(sample_data, axis=1)
                adata = ad.AnnData(X=expr_matrix.T)
                print(f"   ✅ Built AnnData from {len(sample_data)} samples")
            else:
                raise ValueError("Could not extract expression data from GEO")

        print(f"\n✅ Step 3: Quality control...")
        print(f"   Shape: {adata.shape} (cells × genes)")
        print(f"   Sparsity: {(adata.X == 0).sum() / adata.X.size:.2%}")

        # Basic QC
        if adata.n_obs > 100000:
            print(f"   ⚠️  Large dataset ({adata.n_obs} cells), consider downsampling")

        if adata.n_vars > 50000:
            print(f"   ⚠️  Many genes ({adata.n_vars}), consider filtering")

        # Save
        print(f"\n💾 Step 4: Saving to {output_path}...")
        adata.write_h5ad(output_path)

        print(f"\n✅ SUCCESS!")
        print(f"   Saved to: {output_path}")
        print(f"   Dataset shape: {adata.shape}")

        return str(output_path)

    except ImportError as e:
        print(f"\n❌ Missing dependency: {e}")
        print(f"   Install with: pip install GEOparse scanpy anndata")
        sys.exit(1)

    except Exception as e:
        print(f"\n❌ Error downloading {geo_id}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Download GEO dataset and convert to AnnData format"
    )
    parser.add_argument(
        "--geo",
        type=str,
        required=True,
        help="GEO accession ID (e.g., GSE152048)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/geo",
        help="Output directory (default: data/geo)"
    )

    args = parser.parse_args()

    # Ensure GEO ID is uppercase
    geo_id = args.geo.upper()

    output_path = download_geo_dataset(geo_id, args.output_dir)

    print(f"\n{'='*80}")
    print(f"Next steps:")
    print(f"  1. Verify the dataset: scanpy.read_h5ad('{output_path}')")
    print(f"  2. Run benchmarks: python scripts/simple_benchmark.py --dataset {output_path}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
