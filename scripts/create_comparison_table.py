#!/usr/bin/env python3
"""
Create comprehensive comparison table from all benchmark results.

Reads all individual CSV metric files and creates:
1. Master comparison table (all datasets × all algorithms)
2. Summary statistics
3. Presentation-ready formatted tables

Usage:
    python3 create_comparison_table.py
"""

import pandas as pd
from pathlib import Path
import json

# Configuration
BASE_DIR = Path('/home/btd8/llm-paper-analyze')
RESULTS_DIR = BASE_DIR / 'results/benchmarks'
OUTPUT_DIR = BASE_DIR / 'results'

DATASETS = [
    'GSE157827',
    'GSE159677',
    'GSE164983',
    'GSE167490',
    'GSE174367',
    'GSE191288',
    'GSE220243',
    'GSE271107'
]

ALGORITHMS = ['PCA', 'UMAP', 't-SNE', 'PHATE']


def collect_all_metrics():
    """Collect all metric CSV files into a master DataFrame."""
    all_data = []

    for dataset in DATASETS:
        for algo in ALGORITHMS:
            # Find metric file
            metric_file = RESULTS_DIR / dataset / 'metrics' / f'{algo.lower()}_metrics.csv'

            if not metric_file.exists():
                print(f"⚠️  Missing: {dataset} × {algo}")
                continue

            # Load CSV
            df = pd.read_csv(metric_file)
            all_data.append(df)
            print(f"✅ Loaded: {dataset} × {algo}")

    if not all_data:
        print("\n❌ No metric files found!")
        return None

    # Combine all data
    master_df = pd.concat(all_data, ignore_index=True)

    print(f"\n✅ Collected {len(master_df)} benchmark results")

    return master_df


def create_summary_table(master_df):
    """Create summary table with key metrics."""

    # Select key columns
    key_cols = [
        'dataset',
        'algorithm',
        'n_cells',
        'n_genes',
        'trustworthiness_k10',
        'continuity_k10',
        'lid_k10',
        'participation_ratio',
        'runtime_sec'
    ]

    # Filter to available columns
    available_cols = [col for col in key_cols if col in master_df.columns]
    summary = master_df[available_cols].copy()

    # Sort by dataset and algorithm
    summary = summary.sort_values(['dataset', 'algorithm'])

    return summary


def create_pivot_tables(master_df):
    """Create pivot tables for easy comparison."""

    pivots = {}

    # Trustworthiness by dataset and algorithm
    if 'trustworthiness_k10' in master_df.columns:
        pivots['trustworthiness'] = pd.pivot_table(
            master_df,
            values='trustworthiness_k10',
            index='dataset',
            columns='algorithm',
            aggfunc='mean'
        )

    # Continuity by dataset and algorithm
    if 'continuity_k10' in master_df.columns:
        pivots['continuity'] = pd.pivot_table(
            master_df,
            values='continuity_k10',
            index='dataset',
            columns='algorithm',
            aggfunc='mean'
        )

    # LID by dataset and algorithm
    if 'lid_k10' in master_df.columns:
        pivots['lid'] = pd.pivot_table(
            master_df,
            values='lid_k10',
            index='dataset',
            columns='algorithm',
            aggfunc='mean'
        )

    # Runtime by dataset and algorithm
    if 'runtime_sec' in master_df.columns:
        pivots['runtime'] = pd.pivot_table(
            master_df,
            values='runtime_sec',
            index='dataset',
            columns='algorithm',
            aggfunc='mean'
        )

    return pivots


def save_outputs(master_df, summary_df, pivots):
    """Save all outputs to files."""

    OUTPUT_DIR.mkdir(exist_ok=True)

    # 1. Master CSV (all data)
    output_path = OUTPUT_DIR / 'all_benchmarks.csv'
    master_df.to_csv(output_path, index=False)
    print(f"\n✅ Saved master table: {output_path}")

    # 2. Summary CSV
    output_path = OUTPUT_DIR / 'benchmark_summary.csv'
    summary_df.to_csv(output_path, index=False)
    print(f"✅ Saved summary table: {output_path}")

    # 3. Pivot tables
    for metric_name, pivot_df in pivots.items():
        output_path = OUTPUT_DIR / f'comparison_{metric_name}.csv'
        pivot_df.to_csv(output_path)
        print(f"✅ Saved pivot table: {output_path}")

    # 4. Formatted text report
    output_path = OUTPUT_DIR / 'BENCHMARK_REPORT.txt'
    with open(output_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("BENCHMARK COMPARISON REPORT\n")
        f.write("="*80 + "\n\n")

        f.write(f"Total benchmarks: {len(master_df)}\n")
        f.write(f"Datasets: {master_df['dataset'].nunique()}\n")
        f.write(f"Algorithms: {master_df['algorithm'].nunique()}\n\n")

        f.write("="*80 + "\n")
        f.write("SUMMARY TABLE\n")
        f.write("="*80 + "\n\n")
        f.write(summary_df.to_string(index=False))
        f.write("\n\n")

        for metric_name, pivot_df in pivots.items():
            f.write("="*80 + "\n")
            f.write(f"{metric_name.upper()} COMPARISON\n")
            f.write("="*80 + "\n\n")
            f.write(pivot_df.to_string())
            f.write("\n\n")

    print(f"✅ Saved text report: {output_path}")

    # 5. Statistics JSON
    stats = {
        'total_benchmarks': len(master_df),
        'datasets': master_df['dataset'].unique().tolist(),
        'algorithms': master_df['algorithm'].unique().tolist(),
        'metrics': {
            col: {
                'mean': float(master_df[col].mean()) if pd.api.types.is_numeric_dtype(master_df[col]) else None,
                'std': float(master_df[col].std()) if pd.api.types.is_numeric_dtype(master_df[col]) else None,
                'min': float(master_df[col].min()) if pd.api.types.is_numeric_dtype(master_df[col]) else None,
                'max': float(master_df[col].max()) if pd.api.types.is_numeric_dtype(master_df[col]) else None
            }
            for col in master_df.select_dtypes(include=['number']).columns
        }
    }

    output_path = OUTPUT_DIR / 'benchmark_statistics.json'
    with open(output_path, 'w') as f:
        json.dump(stats, f, indent=2)

    print(f"✅ Saved statistics: {output_path}")


def main():
    print("="*80)
    print("Creating Comparison Tables from Benchmark Results")
    print("="*80)

    # Collect all metrics
    master_df = collect_all_metrics()

    if master_df is None:
        return 1

    # Create summary
    summary_df = create_summary_table(master_df)

    # Create pivots
    pivots = create_pivot_tables(master_df)

    # Save outputs
    save_outputs(master_df, summary_df, pivots)

    print("\n"+"="*80)
    print("✅ ALL COMPARISON TABLES CREATED")
    print("="*80)

    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
