#!/usr/bin/env python3
"""
Comprehensive benchmarking script for all GEO datasets using ManyLatents framework.

Runs multiple dimensionality reduction algorithms (PHATE, PCA, UMAP, t-SNE) on each
dataset and generates organized outputs with metrics and visualizations.

Usage:
    python3 benchmark_all_datasets.py --dataset GSE157827 --algorithm phate
    python3 benchmark_all_datasets.py --dataset all --algorithm all
"""

import sys
import os
import argparse
import json
import time
from pathlib import Path
from datetime import datetime

# Add ManyLatents to path
sys.path.insert(0, '/home/btd8/manylatents')

import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns
import torch

# ManyLatents imports
from manylatents.algorithms.latent.phate import PHATEModule
from manylatents.algorithms.latent.pca import PCAModule
from manylatents.algorithms.latent.umap import UMAPModule
from manylatents.algorithms.latent.tsne import TSNEModule

# ManyLatents metrics
from manylatents.metrics.trustworthiness import Trustworthiness
from manylatents.metrics.continuity import Continuity
from manylatents.metrics.lid import LocalIntrinsicDimensionality
from manylatents.metrics.participation_ratio import ParticipationRatio
from manylatents.metrics.knn_preservation import KNNPreservation

# Configuration
BASE_DIR = Path('/home/btd8/llm-paper-analyze')
DATA_DIR = BASE_DIR / 'data/geo/processed'
RESULTS_DIR = BASE_DIR / 'results/benchmarks'

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

ALGORITHMS = {
    'pca': {
        'class': PCAModule,
        'params': {'n_components': 2},
        'name': 'PCA'
    },
    'umap': {
        'class': UMAPModule,
        'params': {'n_components': 2, 'n_neighbors': 15, 'min_dist': 0.1},
        'name': 'UMAP'
    },
    'tsne': {
        'class': TSNEModule,
        'params': {'n_components': 2, 'perplexity': 30},
        'name': 't-SNE'
    },
    'phate': {
        'class': PHATEModule,
        'params': {'n_components': 2, 'knn': 5, 't': 'auto'},
        'name': 'PHATE'
    }
}


class SimpleDataset:
    """Wrapper class for ManyLatents metrics that expect a dataset object."""
    def __init__(self, data):
        self.data = data
        self.original_data = data


def load_dataset(dataset_name, subsample=10000):
    """Load and preprocess a GEO dataset.

    Args:
        dataset_name: GEO accession (e.g., 'GSE157827')
        subsample: Max number of cells to use (for memory efficiency)

    Returns:
        adata: AnnData object
        X: Dense numpy array for algorithms
    """
    print(f"\n{'='*80}")
    print(f"Loading dataset: {dataset_name}")
    print(f"{'='*80}")

    filepath = DATA_DIR / f"{dataset_name}.h5ad"

    if not filepath.exists():
        raise FileNotFoundError(f"Dataset not found: {filepath}")

    # Load dataset
    adata = sc.read_h5ad(filepath)
    print(f"Original shape: {adata.n_obs} cells × {adata.n_vars} genes")

    # Subsample if needed
    if adata.n_obs > subsample:
        print(f"Subsampling to {subsample} cells...")
        sc.pp.subsample(adata, n_obs=subsample)

    # Convert to dense array if sparse
    if hasattr(adata.X, 'toarray'):
        X = adata.X.toarray()
    else:
        X = adata.X.copy()

    # Convert to PyTorch tensor for ManyLatents
    X_tensor = torch.from_numpy(X.astype(np.float32))

    print(f"Final shape: {X.shape[0]} cells × {X.shape[1]} genes")
    print(f"Data type: {X.dtype} → {X_tensor.dtype}")
    print(f"Memory: {X.nbytes / 1024**2:.1f} MB")

    return adata, X, X_tensor


def run_algorithm(algorithm_name, X_data, algorithm_config):
    """Run a dimensionality reduction algorithm.

    Args:
        algorithm_name: Algorithm identifier (e.g., 'phate')
        X_data: Input data matrix (PyTorch tensor)
        algorithm_config: Algorithm configuration dict

    Returns:
        embedding: 2D embedding array (numpy)
        runtime: Execution time in seconds
    """
    print(f"\n{'='*80}")
    print(f"Running {algorithm_config['name']}")
    print(f"{'='*80}")

    # Initialize algorithm
    AlgoClass = algorithm_config['class']
    params = algorithm_config['params']

    print(f"Parameters: {params}")
    print(f"Input type: {type(X_data)}, shape: {X_data.shape}")

    # Run algorithm
    start_time = time.time()

    try:
        # Create algorithm instance
        algo = AlgoClass(**params)

        # Fit and transform (expects PyTorch tensor)
        embedding_tensor = algo.fit_transform(X_data)

        # Convert to numpy for metrics
        if isinstance(embedding_tensor, torch.Tensor):
            embedding = embedding_tensor.detach().cpu().numpy()
        else:
            embedding = embedding_tensor

        runtime = time.time() - start_time

        print(f"✅ Success!")
        print(f"Embedding shape: {embedding.shape}")
        print(f"Runtime: {runtime:.2f} seconds")

        return embedding, runtime

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        raise


def calculate_metrics(X_original, embedding):
    """Calculate comprehensive quality metrics using ManyLatents.

    Args:
        X_original: Original high-dimensional data
        embedding: Low-dimensional embedding

    Returns:
        metrics: Dictionary of metric names and values
    """
    print(f"\n{'='*80}")
    print(f"Calculating Quality Metrics")
    print(f"{'='*80}")

    metrics = {}
    dataset = SimpleDataset(X_original)

    # Trustworthiness (k=10, 25, 50)
    print("Computing Trustworthiness...")
    for k in [10, 25, 50]:
        try:
            trust = Trustworthiness(embedding, dataset, n_neighbors=k)
            metrics[f'trustworthiness_k{k}'] = float(trust)
            print(f"  k={k}: {float(trust):.4f}")
        except Exception as e:
            print(f"  k={k}: ERROR - {e}")
            metrics[f'trustworthiness_k{k}'] = None

    # Continuity (k=10, 25, 50)
    print("Computing Continuity...")
    for k in [10, 25, 50]:
        try:
            cont = Continuity(embedding, dataset, n_neighbors=k)
            metrics[f'continuity_k{k}'] = float(cont)
            print(f"  k={k}: {float(cont):.4f}")
        except Exception as e:
            print(f"  k={k}: ERROR - {e}")
            metrics[f'continuity_k{k}'] = None

    # Local Intrinsic Dimensionality (k=10, 20, 30)
    print("Computing Local Intrinsic Dimensionality...")
    for k in [10, 20, 30]:
        try:
            lid = LocalIntrinsicDimensionality(embedding, k=k)
            metrics[f'lid_k{k}'] = float(lid)
            print(f"  k={k}: {float(lid):.4f}")
        except Exception as e:
            print(f"  k={k}: ERROR - {e}")
            metrics[f'lid_k{k}'] = None

    # Participation Ratio
    print("Computing Participation Ratio...")
    try:
        pr = ParticipationRatio(embedding)
        metrics['participation_ratio'] = float(pr)
        print(f"  PR: {float(pr):.4f}")
    except Exception as e:
        print(f"  PR: ERROR - {e}")
        metrics['participation_ratio'] = None

    print(f"\n✅ Metrics calculation complete")

    return metrics


def create_visualization(embedding, dataset_name, algorithm_name, output_dir):
    """Create visualization of the embedding.

    Args:
        embedding: 2D embedding array
        dataset_name: Dataset identifier
        algorithm_name: Algorithm name
        output_dir: Directory to save figure
    """
    print(f"\nCreating visualization...")

    fig, ax = plt.subplots(figsize=(10, 8))

    # Create scatter plot with gradient coloring
    scatter = ax.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=np.arange(len(embedding)),
        s=5,
        alpha=0.6,
        cmap='viridis',
        rasterized=True
    )

    ax.set_xlabel(f'{algorithm_name} 1', fontsize=12)
    ax.set_ylabel(f'{algorithm_name} 2', fontsize=12)
    ax.set_title(f'{dataset_name} - {algorithm_name} Embedding', fontsize=14, fontweight='bold')

    plt.colorbar(scatter, ax=ax, label='Cell Index')
    plt.tight_layout()

    # Save
    output_path = output_dir / f'{algorithm_name.lower()}_embedding.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  Saved: {output_path}")


def save_metrics_table(metrics, dataset_info, algorithm_name, runtime, output_dir):
    """Save metrics in 2-column table format (Metric | Value).

    Args:
        metrics: Dictionary of metrics
        dataset_info: Dataset metadata (cells, genes)
        algorithm_name: Algorithm name
        runtime: Execution time
        output_dir: Directory to save tables
    """
    print(f"\nSaving metrics tables...")

    # Prepare data
    rows = []
    rows.append(('Dataset', dataset_info['name']))
    rows.append(('Cells', f"{dataset_info['n_cells']:,}"))
    rows.append(('Genes', f"{dataset_info['n_genes']:,}"))
    rows.append(('Algorithm', algorithm_name))
    rows.append(('Embedding Dimensions', '2'))
    rows.append(('', ''))  # Separator

    # Add metrics
    for key, value in sorted(metrics.items()):
        if value is not None:
            formatted_key = key.replace('_', ' ').title()
            if isinstance(value, float):
                rows.append((formatted_key, f"{value:.4f}"))
            else:
                rows.append((formatted_key, str(value)))

    rows.append(('', ''))  # Separator
    rows.append(('Execution Time (s)', f"{runtime:.2f}"))

    # Save as formatted text table
    output_path_txt = output_dir / f'{algorithm_name.lower()}_metrics.txt'
    with open(output_path_txt, 'w') as f:
        f.write(f"{'='*80}\n")
        f.write(f"{dataset_info['name']} - {algorithm_name} Metrics\n")
        f.write(f"{'='*80}\n\n")

        # Calculate column width
        max_key_len = max(len(row[0]) for row in rows)

        for key, value in rows:
            if key == '':
                f.write('\n')
            else:
                f.write(f"{key:<{max_key_len+2}}| {value}\n")

    print(f"  Saved: {output_path_txt}")

    # Save as CSV
    output_path_csv = output_dir / f'{algorithm_name.lower()}_metrics.csv'

    metrics_with_meta = {
        'dataset': dataset_info['name'],
        'n_cells': dataset_info['n_cells'],
        'n_genes': dataset_info['n_genes'],
        'algorithm': algorithm_name,
        'embedding_dims': 2,
        'runtime_sec': runtime,
        **metrics
    }

    df = pd.DataFrame([metrics_with_meta])
    df.to_csv(output_path_csv, index=False)

    print(f"  Saved: {output_path_csv}")

    # Save as JSON
    output_path_json = output_dir / f'{algorithm_name.lower()}_metrics.json'
    with open(output_path_json, 'w') as f:
        json.dump(metrics_with_meta, f, indent=2)

    print(f"  Saved: {output_path_json}")


def benchmark_single(dataset_name, algorithm_name):
    """Run benchmark for a single dataset-algorithm combination.

    Args:
        dataset_name: GEO accession
        algorithm_name: Algorithm identifier (pca, umap, tsne, phate)
    """
    print(f"\n{'#'*80}")
    print(f"# BENCHMARK: {dataset_name} × {ALGORITHMS[algorithm_name]['name']}")
    print(f"{'#'*80}")

    # Create output directory
    output_dir = RESULTS_DIR / dataset_name / 'metrics'
    viz_dir = RESULTS_DIR / dataset_name / 'visualizations'
    output_dir.mkdir(parents=True, exist_ok=True)
    viz_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Load dataset
        adata, X, X_tensor = load_dataset(dataset_name)

        dataset_info = {
            'name': dataset_name,
            'n_cells': X.shape[0],
            'n_genes': X.shape[1]
        }

        # Run algorithm (use tensor for ManyLatents)
        embedding, runtime = run_algorithm(
            algorithm_name,
            X_tensor,
            ALGORITHMS[algorithm_name]
        )

        # Calculate metrics
        metrics = calculate_metrics(X, embedding)

        # Create visualization
        create_visualization(
            embedding,
            dataset_name,
            ALGORITHMS[algorithm_name]['name'],
            viz_dir
        )

        # Save metrics
        save_metrics_table(
            metrics,
            dataset_info,
            ALGORITHMS[algorithm_name]['name'],
            runtime,
            output_dir
        )

        print(f"\n{'='*80}")
        print(f"✅ BENCHMARK COMPLETE: {dataset_name} × {ALGORITHMS[algorithm_name]['name']}")
        print(f"{'='*80}\n")

        return True

    except Exception as e:
        print(f"\n{'='*80}")
        print(f"❌ BENCHMARK FAILED: {dataset_name} × {ALGORITHMS[algorithm_name]['name']}")
        print(f"Error: {str(e)}")
        print(f"{'='*80}\n")

        import traceback
        traceback.print_exc()

        return False


def main():
    parser = argparse.ArgumentParser(description='Benchmark GEO datasets with ManyLatents algorithms')
    parser.add_argument('--dataset', type=str, required=True,
                       help='Dataset name (e.g., GSE157827) or "all"')
    parser.add_argument('--algorithm', type=str, required=True,
                       help='Algorithm name (pca, umap, tsne, phate) or "all"')

    args = parser.parse_args()

    # Determine datasets and algorithms to run
    datasets_to_run = DATASETS if args.dataset == 'all' else [args.dataset]
    algorithms_to_run = list(ALGORITHMS.keys()) if args.algorithm == 'all' else [args.algorithm]

    # Validate
    for dataset in datasets_to_run:
        if dataset not in DATASETS:
            print(f"❌ Unknown dataset: {dataset}")
            print(f"Available datasets: {', '.join(DATASETS)}")
            return 1

    for algo in algorithms_to_run:
        if algo not in ALGORITHMS:
            print(f"❌ Unknown algorithm: {algo}")
            print(f"Available algorithms: {', '.join(ALGORITHMS.keys())}")
            return 1

    # Run benchmarks
    print(f"\n{'#'*80}")
    print(f"# STARTING BENCHMARKS")
    print(f"# Datasets: {len(datasets_to_run)}")
    print(f"# Algorithms: {len(algorithms_to_run)}")
    print(f"# Total runs: {len(datasets_to_run) * len(algorithms_to_run)}")
    print(f"{'#'*80}\n")

    results = []

    for dataset in datasets_to_run:
        for algo in algorithms_to_run:
            success = benchmark_single(dataset, algo)
            results.append({
                'dataset': dataset,
                'algorithm': algo,
                'success': success
            })

    # Summary
    print(f"\n{'#'*80}")
    print(f"# BENCHMARK SUMMARY")
    print(f"{'#'*80}\n")

    successes = sum(1 for r in results if r['success'])
    failures = len(results) - successes

    print(f"Total runs: {len(results)}")
    print(f"Successes: {successes}")
    print(f"Failures: {failures}")

    if failures > 0:
        print(f"\nFailed runs:")
        for r in results:
            if not r['success']:
                print(f"  - {r['dataset']} × {r['algorithm']}")

    return 0 if failures == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
