#!/usr/bin/env python3
"""
Explore CELLxGENE Census to find available datasets.
"""
import cellxgene_census
import pandas as pd

print("=" * 80)
print("CELLxGENE CENSUS - DATASET EXPLORATION")
print("=" * 80)

# Open the census
with cellxgene_census.open_soma(census_version="latest") as census:
    print("\nOpening Census...")

    # Get experiment metadata
    print("\nFetching dataset metadata...")

    # Access the experiment (human data)
    human = census["census_data"]["homo_sapiens"]

    # Get dataset information
    datasets = human.obs.read(column_names=["dataset_id", "tissue_general", "assay", "suspension_type", "cell_type"]).concat()
    datasets_df = datasets.to_pandas()

    # Get unique datasets
    unique_datasets = datasets_df.groupby('dataset_id').agg({
        'tissue_general': lambda x: x.mode()[0] if len(x.mode()) > 0 else x.iloc[0],
        'assay': lambda x: x.mode()[0] if len(x.mode()) > 0 else x.iloc[0],
        'suspension_type': lambda x: x.mode()[0] if len(x.mode()) > 0 else x.iloc[0],
        'cell_type': 'count'
    }).reset_index()

    unique_datasets.columns = ['dataset_id', 'tissue', 'assay', 'suspension_type', 'n_cells']

    # Filter for single-cell RNA-seq only
    rna_seq_assays = ['10x 3\' v2', '10x 3\' v3', '10x 5\' v2', 'Smart-seq2', '10x 5\' v1', '10x 3\' v1']
    rna_datasets = unique_datasets[unique_datasets['assay'].isin(rna_seq_assays)]

    print(f"\nTotal datasets in Census: {len(unique_datasets)}")
    print(f"RNA-seq datasets: {len(rna_datasets)}")

    # Sort by number of cells (descending)
    rna_datasets = rna_datasets.sort_values('n_cells', ascending=False)

    print(f"\nTop 20 RNA-seq datasets by cell count:")
    print("=" * 80)
    for i, row in rna_datasets.head(20).iterrows():
        print(f"{row['dataset_id']}: {row['n_cells']:,} cells | {row['tissue']} | {row['assay']}")

    # Save full list
    output_file = "/home/btd8/llm-paper-analyze/cellxgene_datasets.csv"
    rna_datasets.to_csv(output_file, index=False)
    print(f"\n✓ Full dataset list saved to: {output_file}")
    print(f"  Total RNA-seq datasets available: {len(rna_datasets)}")

    print("\n" + "=" * 80)
    print("RECOMMENDATION:")
    print("=" * 80)
    print(f"Select top {min(92, len(rna_datasets))} datasets by cell count for download")
    print("=" * 80)
