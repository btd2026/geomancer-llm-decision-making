#!/usr/bin/env python3
"""
Explore CELLxGENE Census dataset metadata to find publication/citation information.
"""

import cellxgene_census
import pandas as pd

print("=" * 80)
print("CELLxGENE CENSUS - DATASET METADATA EXPLORATION")
print("=" * 80)

# Open the census
with cellxgene_census.open_soma(census_version="2025-01-30") as census:
    print("\n1. Exploring census_info.datasets table...")

    # Get dataset information from census_info
    datasets = census["census_info"]["datasets"].read().concat().to_pandas()

    print(f"\n   Total datasets: {len(datasets)}")
    print(f"\n   Available metadata fields ({len(datasets.columns)} total):")
    for i, col in enumerate(datasets.columns, 1):
        print(f"      {i:2}. {col}")

    print("\n2. Examining first dataset record...")
    first_dataset = datasets.iloc[0]

    print("\n   Dataset ID:", first_dataset.get('dataset_id', 'N/A'))
    print("   Dataset Title:", first_dataset.get('dataset_title', 'N/A'))

    # Check for citation/publication fields
    publication_fields = [
        'citation', 'doi', 'publication_doi', 'collection_doi',
        'publication', 'journal', 'authors', 'publication_date',
        'collection_name', 'collection_id', 'collection_url'
    ]

    print("\n3. Publication/Citation fields:")
    for field in publication_fields:
        if field in datasets.columns:
            value = first_dataset.get(field, 'N/A')
            if pd.notna(value):
                # Truncate if too long
                value_str = str(value)
                if len(value_str) > 150:
                    value_str = value_str[:150] + "..."
                print(f"      ✓ {field}: {value_str}")
            else:
                print(f"      ○ {field}: (empty)")
        else:
            print(f"      ✗ {field}: (not available)")

    print("\n4. Sample of 3 datasets with full metadata:")
    print("=" * 80)

    for idx in range(min(3, len(datasets))):
        row = datasets.iloc[idx]
        print(f"\n   Dataset #{idx+1}:")
        print(f"   ID: {row.get('dataset_id', 'N/A')}")
        print(f"   Title: {row.get('dataset_title', 'N/A')}")

        # Show any citation/publication info
        for field in ['citation', 'collection_name', 'collection_doi']:
            if field in datasets.columns and pd.notna(row.get(field)):
                value = str(row[field])
                if len(value) > 200:
                    value = value[:200] + "..."
                print(f"   {field}: {value}")

    print("\n" + "=" * 80)
    print("5. Summary:")
    print("=" * 80)

    # Check which datasets have citation info
    if 'citation' in datasets.columns:
        has_citation = datasets['citation'].notna().sum()
        print(f"   Datasets with citation info: {has_citation}/{len(datasets)} ({has_citation/len(datasets)*100:.1f}%)")

    if 'collection_doi' in datasets.columns:
        has_doi = datasets['collection_doi'].notna().sum()
        print(f"   Datasets with collection DOI: {has_doi}/{len(datasets)} ({has_doi/len(datasets)*100:.1f}%)")

    # Save full metadata to CSV
    output_file = "/home/btd8/llm-paper-analyze/cellxgene_full_metadata.csv"
    datasets.to_csv(output_file, index=False)
    print(f"\n   ✓ Full metadata saved to: {output_file}")
    print(f"   Total fields: {len(datasets.columns)}")
    print(f"   Total datasets: {len(datasets)}")

print("\n" + "=" * 80)
print("EXPLORATION COMPLETE")
print("=" * 80)
