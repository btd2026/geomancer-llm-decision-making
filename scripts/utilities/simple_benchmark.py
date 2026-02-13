#!/usr/bin/env python3
"""
Simple benchmark script for proof of concept.

This script:
1. Loads a dataset (synthetic or GEO)
2. Runs multiple algorithms via manyagents
3. Stores results in papers.db → manylatents_results table
4. Prints summary

Usage:
    # Use synthetic data (fast, for testing)
    python scripts/simple_benchmark.py --dataset synthetic

    # Use real GEO data
    python scripts/simple_benchmark.py --dataset data/geo/GSE152048.h5ad
"""

import asyncio
import sys
import argparse
import time
import json
import sqlite3
from pathlib import Path
import numpy as np

# Add manyagents and manylatents to path
sys.path.insert(0, '/home/btd8/manylatents')
sys.path.insert(0, '/home/btd8/manyagents')


async def run_benchmark(dataset_name, dataset_data, algorithms, db_path):
    """
    Run benchmark: test multiple algorithms on a dataset.

    Args:
        dataset_name: Name/ID of dataset
        dataset_data: Numpy array of data, or None to use built-in dataset
        algorithms: List of algorithm configs to test
        db_path: Path to database to store results
    """
    from manyagents.adapters import ManyLatentsAdapter

    print(f"="*80)
    print(f"BENCHMARK: {dataset_name}")
    print(f"="*80)

    adapter = ManyLatentsAdapter()
    results = []

    for algo_config in algorithms:
        algo_name = algo_config['name']
        print(f"\n{'─'*80}")
        print(f"Running: {algo_name}")
        print(f"{'─'*80}")

        try:
            start_time = time.time()

            # Run algorithm
            if dataset_data is not None:
                # Use provided data
                result = await adapter.run(
                    task_config=algo_config['config'],
                    input_data=dataset_data
                )
            else:
                # Use built-in dataset from manylatents
                result = await adapter.run(
                    task_config=algo_config['config'],
                    input_files={}
                )

            end_time = time.time()
            execution_time = end_time - start_time

            # Extract results
            if result['success']:
                embeddings = result['output_files']['embeddings']
                scores = result['output_files'].get('scores', {})

                print(f"   ✅ Success")
                print(f"   Embeddings shape: {embeddings.shape}")
                print(f"   Execution time: {execution_time:.2f}s")

                if scores:
                    print(f"   Metrics:")
                    for metric_name, metric_value in scores.items():
                        if isinstance(metric_value, (int, float)):
                            print(f"      {metric_name}: {metric_value:.4f}")
                        else:
                            print(f"      {metric_name}: {metric_value}")

                # Store result
                results.append({
                    'dataset_name': dataset_name,
                    'algorithm_name': algo_name,
                    'success': True,
                    'embeddings_shape': list(embeddings.shape),
                    'execution_time': execution_time,
                    'scores': scores,
                    'config': algo_config['config']
                })

            else:
                print(f"   ❌ Failed: {result.get('summary', 'Unknown error')}")
                results.append({
                    'dataset_name': dataset_name,
                    'algorithm_name': algo_name,
                    'success': False,
                    'error': result.get('summary', 'Unknown error')
                })

        except Exception as e:
            print(f"   ❌ Exception: {e}")
            import traceback
            traceback.print_exc()

            results.append({
                'dataset_name': dataset_name,
                'algorithm_name': algo_name,
                'success': False,
                'error': str(e)
            })

    # Store in database
    print(f"\n{'='*80}")
    print(f"Storing results in database...")
    print(f"{'='*80}")

    store_results_in_db(results, db_path)

    # Print summary
    print_summary(results)

    return results


def store_results_in_db(results, db_path):
    """Store benchmark results in database."""

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # For proof of concept, create a simplified results table
    # We'll use a test_benchmarks table instead of manylatents_results for now

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS test_benchmarks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            dataset_name TEXT NOT NULL,
            algorithm_name TEXT NOT NULL,
            success BOOLEAN NOT NULL,
            embeddings_shape TEXT,
            execution_time REAL,
            scores TEXT,
            config TEXT,
            error TEXT
        )
    ''')

    for result in results:
        cursor.execute('''
            INSERT INTO test_benchmarks (
                dataset_name,
                algorithm_name,
                success,
                embeddings_shape,
                execution_time,
                scores,
                config,
                error
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            result['dataset_name'],
            result['algorithm_name'],
            result['success'],
            json.dumps(result.get('embeddings_shape')),
            result.get('execution_time'),
            json.dumps(result.get('scores', {})),
            json.dumps(result.get('config', {})),
            result.get('error')
        ))

    conn.commit()

    print(f"   ✅ Stored {len(results)} results in database")
    print(f"   Table: test_benchmarks")

    # Show what was stored
    cursor.execute('SELECT COUNT(*) FROM test_benchmarks')
    total_count = cursor.fetchone()[0]
    print(f"   Total benchmarks in DB: {total_count}")

    conn.close()


def print_summary(results):
    """Print benchmark summary."""

    print(f"\n{'='*80}")
    print(f"BENCHMARK SUMMARY")
    print(f"{'='*80}")

    successful = [r for r in results if r['success']]
    failed = [r for r in results if not r['success']]

    print(f"\n✅ Successful: {len(successful)}/{len(results)}")
    print(f"❌ Failed: {len(failed)}/{len(results)}")

    if successful:
        print(f"\n{'Algorithm':<15} {'Time (s)':<12} {'Shape':<15} {'Metrics'}")
        print(f"{'─'*70}")

        for result in successful:
            algo = result['algorithm_name']
            time_s = result.get('execution_time', 0)
            shape = result.get('embeddings_shape', [])
            scores = result.get('scores', {})

            # Get first metric for summary
            metric_str = ""
            if scores:
                first_metric = list(scores.items())[0]
                if isinstance(first_metric[1], (int, float)):
                    metric_str = f"{first_metric[0]}={first_metric[1]:.3f}"
                else:
                    metric_str = f"{first_metric[0]}={first_metric[1]}"

            print(f"{algo:<15} {time_s:<12.2f} {str(shape):<15} {metric_str}")

    if failed:
        print(f"\nFailed algorithms:")
        for result in failed:
            print(f"   ❌ {result['algorithm_name']}: {result.get('error', 'Unknown')}")


async def main_async(args):
    """Main async function."""

    # Dataset configuration
    if args.dataset == "synthetic":
        print(f"Using synthetic 'swissroll' dataset from manylatents")
        dataset_name = "swissroll"
        dataset_data = None  # Will use manylatents' built-in dataset
    else:
        print(f"Loading dataset from: {args.dataset}")
        dataset_name = Path(args.dataset).stem

        # Load data
        try:
            import anndata as ad
            adata = ad.read_h5ad(args.dataset)
            dataset_data = adata.X

            if hasattr(dataset_data, 'toarray'):
                dataset_data = dataset_data.toarray()

            print(f"   Loaded: {dataset_data.shape}")

        except Exception as e:
            print(f"❌ Could not load dataset: {e}")
            sys.exit(1)

    # Algorithm configurations
    algorithms = [
        {
            'name': 'PCA',
            'config': {
                'algorithm': 'pca',
                'data': 'swissroll' if dataset_data is None else None,
                'n_components': 2
            }
        },
        {
            'name': 'UMAP',
            'config': {
                'algorithm': 'umap',
                'data': 'swissroll' if dataset_data is None else None,
                'n_components': 2
            }
        },
        {
            'name': 't-SNE',
            'config': {
                'algorithm': 'tsne',
                'data': 'swissroll' if dataset_data is None else None,
                'n_components': 2
            }
        }
    ]

    # Run benchmark
    results = await run_benchmark(
        dataset_name=dataset_name,
        dataset_data=dataset_data,
        algorithms=algorithms,
        db_path=args.db_path
    )

    print(f"\n{'='*80}")
    print(f"PROOF OF CONCEPT COMPLETE!")
    print(f"{'='*80}")
    print(f"\nNext steps:")
    print(f"  1. Query results: sqlite3 {args.db_path} 'SELECT * FROM test_benchmarks'")
    print(f"  2. Scale to more datasets")
    print(f"  3. Add more algorithms")
    print(f"  4. Train ML model")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run simple benchmark for proof of concept"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="synthetic",
        help="Dataset to use: 'synthetic' or path to .h5ad file (default: synthetic)"
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default="data/papers/metadata/papers.db",
        help="Path to database (default: data/papers/metadata/papers.db)"
    )

    args = parser.parse_args()

    # Run async main
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
