#!/usr/bin/env python3
"""
Analyze the new manual classifications and determine the best strategy to integrate them.
"""

import pandas as pd
import numpy as np
from pathlib import Path

def analyze_new_labels():
    """Analyze the new manual classification data."""
    print("="*70)
    print("ANALYZING NEW MANUAL CLASSIFICATIONS")
    print("="*70)

    # Load both label files
    print("Loading label files...")
    old_labels = pd.read_csv("/home/btd8/Documents/phate_labels_rich.csv")
    new_labels = pd.read_csv("/home/btd8/Documents/phate_labels_rich (1).csv")

    print(f"Old labels: {len(old_labels)} datasets")
    print(f"New labels: {len(new_labels)} datasets")

    # Analyze labeled samples in each
    old_labeled = old_labels[old_labels['primary_structure'].notna() & (old_labels['primary_structure'] != '')]
    new_labeled = new_labels[new_labels['primary_structure'].notna() & (new_labels['primary_structure'] != '')]

    print(f"\nLabeled samples:")
    print(f"Old file: {len(old_labeled)}/{len(old_labels)} ({len(old_labeled)/len(old_labels)*100:.1f}%)")
    print(f"New file: {len(new_labeled)}/{len(new_labels)} ({len(new_labeled)/len(new_labels)*100:.1f}%)")

    # Compare label distributions
    print(f"\nOld label distribution:")
    for label, count in old_labeled['primary_structure'].value_counts().items():
        print(f"  {label}: {count}")

    print(f"\nNew label distribution:")
    for label, count in new_labeled['primary_structure'].value_counts().items():
        print(f"  {label}: {count}")

    # Check if any datasets overlap
    old_ids = set(old_labeled['dataset_id'])
    new_ids = set(new_labeled['dataset_id'])
    overlap = old_ids & new_ids

    print(f"\nDataset ID overlap:")
    print(f"Common labeled datasets: {len(overlap)}")

    if overlap:
        print("Overlapping datasets:")
        for dataset_id in list(overlap)[:10]:
            old_label = old_labeled[old_labeled['dataset_id'] == dataset_id]['primary_structure'].iloc[0]
            new_label = new_labeled[new_labeled['dataset_id'] == dataset_id]['primary_structure'].iloc[0]
            print(f"  {dataset_id[:36]}: {old_label} -> {new_label}")

    return old_labeled, new_labeled, overlap

def check_data_sources():
    """Check what data sources we have available."""
    print(f"\n" + "="*50)
    print("CHECKING AVAILABLE DATA SOURCES")
    print("="*50)

    sources = {}

    # Check original training metrics
    metrics_path = "/home/btd8/llm-paper-analyze/data/manylatents_benchmark/embedding_metrics.csv"
    if Path(metrics_path).exists():
        metrics_df = pd.read_csv(metrics_path)
        sources['original_metrics'] = len(metrics_df)
        print(f"✓ Original training metrics: {len(metrics_df)} datasets")

    # Check computed PHATE metrics
    phate_metrics_path = "/home/btd8/llm-paper-analyze/data/phate_basic_metrics.csv"
    if Path(phate_metrics_path).exists():
        phate_metrics = pd.read_csv(phate_metrics_path)
        sources['phate_metrics'] = len(phate_metrics)
        print(f"✓ Computed PHATE metrics: {len(phate_metrics)} datasets")

    # Check PHATE embeddings directory
    phate_dir = Path("/home/btd8/manylatents/outputs/phate_k100_benchmark")
    if phate_dir.exists():
        phate_dirs = list(phate_dir.glob("*"))
        sources['phate_embeddings'] = len(phate_dirs)
        print(f"✓ PHATE embeddings: {len(phate_dirs)} datasets")

    # Check source data directory
    source_dir = Path("/nfs/roberts/project/pi_sk2433/shared/Geomancer_2025_Data/manylatents_small_datasets")
    if source_dir.exists():
        source_files = list(source_dir.glob("*.h5ad"))
        sources['source_data'] = len(source_files)
        print(f"✓ Source data files: {len(source_files)} datasets")
    else:
        print("✗ Source data directory not accessible")

    return sources

def create_integration_strategy(old_labeled, new_labeled, overlap, sources):
    """Create a strategy for integrating the new manual labels."""
    print(f"\n" + "="*50)
    print("INTEGRATION STRATEGY")
    print("="*50)

    total_labeled = len(old_labeled) + len(new_labeled) - len(overlap)
    print(f"Total unique labeled datasets available: {total_labeled}")

    print(f"\n🎯 RECOMMENDED APPROACHES:")

    # Strategy 1: Combine all available labels
    print(f"\n1. **COMBINE ALL LABELS** (Recommended)")
    print(f"   • Merge {len(old_labeled)} old + {len(new_labeled)} new = {total_labeled} unique labeled samples")
    print(f"   • Handle {len(overlap)} overlapping datasets (use newer labels)")
    print(f"   • Significantly increase training data size")
    print(f"   • Requires computing metrics for new datasets")

    # Strategy 2: Use new labels only if we can compute embeddings
    print(f"\n2. **NEW LABELS ONLY**")
    print(f"   • Focus on {len(new_labeled)} newly labeled datasets")
    print(f"   • Compute PHATE embeddings and metrics for these")
    print(f"   • Train fresh model with consistent methodology")
    print(f"   • Requires access to source data")

    # Strategy 3: Current approach with improved features
    print(f"\n3. **ENHANCE CURRENT MODEL**")
    print(f"   • Keep current {len(old_labeled)} training samples")
    print(f"   • Improve feature extraction for better performance")
    print(f"   • Add more sophisticated metrics")
    print(f"   • Apply to current 100 classified datasets")

    return total_labeled

def check_new_datasets_availability(new_labeled):
    """Check if the newly labeled datasets are available for processing."""
    print(f"\n" + "="*50)
    print("CHECKING NEW DATASET AVAILABILITY")
    print("="*50)

    new_ids = set(new_labeled['dataset_id'])
    print(f"Checking availability for {len(new_ids)} newly labeled datasets...")

    # Check in source directory
    source_dir = Path("/nfs/roberts/project/pi_sk2433/shared/Geomancer_2025_Data/manylatents_small_datasets")
    if source_dir.exists():
        source_files = list(source_dir.glob("*.h5ad"))
        source_ids = set([f.stem for f in source_files])

        available_new = new_ids & source_ids
        print(f"✓ Found {len(available_new)} newly labeled datasets in source directory")

        if len(available_new) > 0:
            print(f"Available datasets (first 10):")
            for dataset_id in list(available_new)[:10]:
                structure = new_labeled[new_labeled['dataset_id'] == dataset_id]['primary_structure'].iloc[0]
                print(f"  {dataset_id[:36]}: {structure}")

            return available_new
        else:
            print("✗ No newly labeled datasets found in source directory")
            return set()
    else:
        print("✗ Source directory not accessible")
        return set()

def main():
    print("Starting analysis of new manual classifications...")

    # Analyze the new labels
    old_labeled, new_labeled, overlap = analyze_new_labels()

    # Check available data sources
    sources = check_data_sources()

    # Create integration strategy
    total_labeled = create_integration_strategy(old_labeled, new_labeled, overlap, sources)

    # Check if new datasets are available for processing
    available_new = check_new_datasets_availability(new_labeled)

    # Summary and recommendations
    print(f"\n" + "="*70)
    print("SUMMARY AND NEXT STEPS")
    print("="*70)

    print(f"📊 **DATA SUMMARY**")
    print(f"   • Old labeled: {len(old_labeled)} datasets")
    print(f"   • New labeled: {len(new_labeled)} datasets")
    print(f"   • Overlap: {len(overlap)} datasets")
    print(f"   • Available for processing: {len(available_new)} new datasets")

    print(f"\n🎯 **RECOMMENDED ACTION**")
    if len(available_new) >= 20:
        print(f"   ✅ COMPUTE EMBEDDINGS FOR NEW LABELS")
        print(f"   • Process {len(available_new)} newly labeled datasets")
        print(f"   • Compute PHATE embeddings and metrics")
        print(f"   • Combine with existing training data")
        print(f"   • Retrain model with {total_labeled} total samples")
    elif len(available_new) > 0:
        print(f"   ⚠️  LIMITED NEW DATA AVAILABLE")
        print(f"   • Only {len(available_new)} new datasets found")
        print(f"   • Consider enhancing current model instead")
        print(f"   • Or manually verify data locations")
    else:
        print(f"   ❌ ENHANCE CURRENT APPROACH")
        print(f"   • New datasets not accessible")
        print(f"   • Focus on improving feature extraction")
        print(f"   • Add more sophisticated structural metrics")

    print(f"\n📝 **FILES CREATED**")

    # Save analysis results
    analysis_results = {
        'old_labeled_count': len(old_labeled),
        'new_labeled_count': len(new_labeled),
        'overlap_count': len(overlap),
        'available_new_count': len(available_new),
        'total_potential_labels': total_labeled,
        'recommendation': 'compute_new_embeddings' if len(available_new) >= 20 else 'enhance_current'
    }

    import json
    with open('/home/btd8/llm-paper-analyze/data/label_analysis_summary.json', 'w') as f:
        json.dump(analysis_results, f, indent=2)

    print(f"   • Analysis summary: /home/btd8/llm-paper-analyze/data/label_analysis_summary.json")

    return old_labeled, new_labeled, available_new, analysis_results

if __name__ == "__main__":
    old_labeled, new_labeled, available_new, results = main()