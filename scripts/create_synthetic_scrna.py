#!/usr/bin/env python3
"""
Create a realistic synthetic scRNA-seq dataset.

This creates a dataset with realistic characteristics:
- Sparse matrix (90%+ zeros, like real scRNA-seq)
- Multiple cell types with distinct expression patterns
- Realistic cell/gene counts
- Saved in AnnData format for benchmarking

This is useful for testing the pipeline without GEO download issues.
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path


def create_synthetic_scrna(
    n_cells=5000,
    n_genes=2000,
    n_cell_types=5,
    sparsity=0.90,
    output_path="data/synthetic/scrna_test.h5ad"
):
    """
    Create synthetic scRNA-seq data with realistic characteristics.

    Args:
        n_cells: Number of cells
        n_genes: Number of genes
        n_cell_types: Number of distinct cell types
        sparsity: Fraction of zeros (typical: 0.85-0.95 for scRNA-seq)
        output_path: Where to save the .h5ad file
    """
    import scanpy as sc
    import anndata as ad

    print(f"="*80)
    print(f"Creating Synthetic scRNA-seq Dataset")
    print(f"="*80)
    print(f"  Cells: {n_cells}")
    print(f"  Genes: {n_genes}")
    print(f"  Cell types: {n_cell_types}")
    print(f"  Target sparsity: {sparsity:.1%}")

    np.random.seed(42)

    # Step 1: Generate cell type assignments
    cells_per_type = n_cells // n_cell_types
    cell_types = np.repeat(range(n_cell_types), cells_per_type)
    if len(cell_types) < n_cells:
        cell_types = np.concatenate([cell_types, np.random.choice(n_cell_types, n_cells - len(cell_types))])
    np.random.shuffle(cell_types)

    print(f"\n✅ Step 1: Generated {n_cell_types} cell types")

    # Step 2: Generate expression matrix
    # Each cell type has some marker genes with higher expression
    X = np.zeros((n_cells, n_genes))

    markers_per_type = n_genes // (n_cell_types * 2)  # Each type has ~10% of genes as markers

    for cell_type in range(n_cell_types):
        # Select cells of this type
        type_mask = cell_types == cell_type
        n_type_cells = type_mask.sum()

        # Select marker genes for this type
        marker_start = cell_type * markers_per_type
        marker_end = marker_start + markers_per_type

        # Generate expression for marker genes (higher for this type)
        X[type_mask, marker_start:marker_end] = np.random.negative_binomial(
            n=5, p=0.3, size=(n_type_cells, markers_per_type)
        )

        # Generate background expression for all genes (lower, sparser)
        background = np.random.negative_binomial(
            n=2, p=0.7, size=(n_type_cells, n_genes)
        )
        X[type_mask, :] += background * 0.1

    print(f"✅ Step 2: Generated expression matrix")

    # Step 3: Apply sparsity (set values below threshold to 0)
    threshold = np.percentile(X[X > 0], (sparsity - (X == 0).mean()) / (1 - (X == 0).mean()) * 100)
    X[X < threshold] = 0
    actual_sparsity = (X == 0).sum() / X.size

    print(f"✅ Step 3: Applied sparsity (actual: {actual_sparsity:.1%})")

    # Step 4: Create AnnData object
    obs = pd.DataFrame({
        'cell_id': [f'Cell_{i}' for i in range(n_cells)],
        'cell_type': [f'Type_{ct}' for ct in cell_types],
        'n_genes_detected': (X > 0).sum(axis=1)
    })
    obs.index = obs['cell_id']

    var = pd.DataFrame({
        'gene_id': [f'Gene_{i}' for i in range(n_genes)],
        'n_cells_expressed': (X > 0).sum(axis=0)
    })
    var.index = var['gene_id']

    adata = ad.AnnData(X=X, obs=obs, var=var)

    print(f"✅ Step 4: Created AnnData object")

    # Step 5: Add some QC metrics (like real data would have)
    sc.pp.calculate_qc_metrics(adata, inplace=True)

    print(f"✅ Step 5: Calculated QC metrics")

    # Step 6: Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(output_path)

    print(f"\n{'='*80}")
    print(f"✅ SUCCESS - Synthetic Dataset Created!")
    print(f"{'='*80}")
    print(f"  Saved to: {output_path}")
    print(f"  Shape: {adata.shape} (cells × genes)")
    print(f"  Sparsity: {actual_sparsity:.2%}")
    print(f"  Cell types: {n_cell_types}")
    print(f"  File size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")

    # Print summary stats
    print(f"\n📊 Dataset Summary:")
    print(f"  Mean genes per cell: {adata.obs['n_genes_detected'].mean():.0f}")
    print(f"  Median genes per cell: {adata.obs['n_genes_detected'].median():.0f}")
    print(f"  Mean cells per gene: {adata.var['n_cells_expressed'].mean():.0f}")
    print(f"  Cell type distribution:")
    for ct, count in zip(*np.unique(cell_types, return_counts=True)):
        print(f"    Type_{ct}: {count} cells")

    print(f"\n💡 Next step:")
    print(f"  python scripts/simple_benchmark_v2.py --dataset {output_path}")

    return str(output_path)


def main():
    parser = argparse.ArgumentParser(
        description="Create synthetic scRNA-seq dataset for testing"
    )
    parser.add_argument(
        "--n-cells",
        type=int,
        default=5000,
        help="Number of cells (default: 5000)"
    )
    parser.add_argument(
        "--n-genes",
        type=int,
        default=2000,
        help="Number of genes (default: 2000)"
    )
    parser.add_argument(
        "--n-cell-types",
        type=int,
        default=5,
        help="Number of cell types (default: 5)"
    )
    parser.add_argument(
        "--sparsity",
        type=float,
        default=0.90,
        help="Target sparsity (default: 0.90)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/synthetic/scrna_test.h5ad",
        help="Output path (default: data/synthetic/scrna_test.h5ad)"
    )

    args = parser.parse_args()

    create_synthetic_scrna(
        n_cells=args.n_cells,
        n_genes=args.n_genes,
        n_cell_types=args.n_cell_types,
        sparsity=args.sparsity,
        output_path=args.output
    )


if __name__ == "__main__":
    main()
