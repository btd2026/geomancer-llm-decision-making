#!/usr/bin/env python3
"""
Run benchmark on a specific .h5ad dataset file.

This version loads external data files and runs them through manylatents.
"""

import sys
import argparse
import time
import json
import sqlite3
from pathlib import Path
import numpy as np

# Add manylatents to path
sys.path.insert(0, '/home/btd8/manylatents')


def run_benchmark_on_file(dataset_path, db_path):
    """Run benchmark on a specific dataset file."""
    from manylatents.api import run
    import anndata as ad

    print(f"="*80)
    print(f"BENCHMARK: External Dataset")
    print(f"="*80)

    # Load dataset
    dataset_path = Path(dataset_path)
    if not dataset_path.exists():
        print(f"❌ Dataset not found: {dataset_path}")
        sys.exit(1)

    print(f"\n📂 Loading dataset: {dataset_path.name}")
    adata = ad.read_h5ad(dataset_path)
    print(f"   Shape: {adata.shape} (cells × genes)")

    # Convert to numpy if sparse
    if hasattr(adata.X, 'toarray'):
        X_data = adata.X.toarray()
    else:
        X_data = np.array(adata.X)

    print(f"   Sparsity: {(X_data == 0).sum() / X_data.size:.2%}")
    print(f"   Data type: {X_data.dtype}")

    # Algorithms to test
    algorithms = [
        {
            'name': 'PCA',
            'manylatents_config': {
                'input_data': X_data,
                'algorithms': {
                    'latent': {
                        '_target_': 'manylatents.algorithms.latent.pca.PCAModule',
                        'n_components': 2
                    }
                }
            }
        },
        {
            'name': 'UMAP',
            'manylatents_config': {
                'input_data': X_data,
                'algorithms': {
                    'latent': {
                        '_target_': 'manylatents.algorithms.latent.umap.UMAPModule',
                        'n_components': 2
                    }
                }
            }
        },
        {
            'name': 't-SNE',
            'manylatents_config': {
                'input_data': X_data,
                'algorithms': {
                    'latent': {
                        '_target_': 'manylatents.algorithms.latent.tsne.TSNEModule',
                        'n_components': 2
                    }
                }
            }
        }
    ]

    results = []

    for algo_config in algorithms:
        algo_name = algo_config['name']
        print(f"\n{'─'*80}")
        print(f"Running: {algo_name}")
        print(f"{'─'*80}")

        try:
            start_time = time.time()

            # Run algorithm
            result = run(**algo_config['manylatents_config'])

            end_time = time.time()
            execution_time = end_time - start_time

            # Extract results
            embeddings = result['embeddings']
            scores = result.get('scores', {})

            print(f"   ✅ Success!")
            print(f"   Embeddings shape: {embeddings.shape}")
            print(f"   Execution time: {execution_time:.2f}s")

            if scores:
                print(f"   Metrics:")
                for metric_name, metric_value in scores.items():
                    if isinstance(metric_value, (int, float)):
                        print(f"      {metric_name}: {metric_value:.4f}")

            # Store result
            results.append({
                'dataset_name': dataset_path.stem,
                'dataset_path': str(dataset_path),
                'n_cells': adata.shape[0],
                'n_genes': adata.shape[1],
                'algorithm_name': algo_name,
                'success': True,
                'embeddings_shape': list(embeddings.shape),
                'execution_time': execution_time,
                'scores': scores
            })

        except Exception as e:
            print(f"   ❌ Exception: {e}")
            import traceback
            traceback.print_exc()

            results.append({
                'dataset_name': dataset_path.stem,
                'dataset_path': str(dataset_path),
                'n_cells': adata.shape[0],
                'n_genes': adata.shape[1],
                'algorithm_name': algo_name,
                'success': False,
                'error': str(e)
            })

    # Store in database
    print(f"\n{'='*80}")
    print(f"Storing results...")
    print(f"{'='*80}")
    store_results(results, db_path)

    # Print summary
    print_summary(results)

    return results


def store_results(results, db_path):
    """Store results in database."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Create table if not exists
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS file_benchmarks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            dataset_name TEXT NOT NULL,
            dataset_path TEXT,
            n_cells INTEGER,
            n_genes INTEGER,
            algorithm_name TEXT NOT NULL,
            success BOOLEAN NOT NULL,
            embeddings_shape TEXT,
            execution_time REAL,
            scores TEXT,
            error TEXT
        )
    ''')

    for result in results:
        cursor.execute('''
            INSERT INTO file_benchmarks (
                dataset_name, dataset_path, n_cells, n_genes,
                algorithm_name, success, embeddings_shape,
                execution_time, scores, error
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            result['dataset_name'],
            result.get('dataset_path'),
            result.get('n_cells'),
            result.get('n_genes'),
            result['algorithm_name'],
            result['success'],
            json.dumps(result.get('embeddings_shape')),
            result.get('execution_time'),
            json.dumps(result.get('scores', {})),
            result.get('error')
        ))

    conn.commit()
    print(f"   ✅ Stored {len(results)} results in 'file_benchmarks' table")

    cursor.execute('SELECT COUNT(*) FROM file_benchmarks')
    total = cursor.fetchone()[0]
    print(f"   Total file benchmarks: {total}")

    conn.close()


def print_summary(results):
    """Print summary."""
    print(f"\n{'='*80}")
    print(f"BENCHMARK SUMMARY")
    print(f"{'='*80}")

    successful = [r for r in results if r['success']]
    failed = [r for r in results if not r['success']]

    print(f"\n✅ Successful: {len(successful)}/{len(results)}")
    print(f"❌ Failed: {len(failed)}/{len(results)}")

    if successful:
        print(f"\n{'Algorithm':<15} {'Time (s)':<12} {'Cells':<10} {'Genes':<10}")
        print(f"{'─'*50}")

        for result in successful:
            algo = result['algorithm_name']
            time_s = result.get('execution_time', 0)
            n_cells = result.get('n_cells', 0)
            n_genes = result.get('n_genes', 0)

            print(f"{algo:<15} {time_s:<12.2f} {n_cells:<10} {n_genes:<10}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark a specific .h5ad file")
    parser.add_argument("dataset", type=str, help="Path to .h5ad dataset file")
    parser.add_argument(
        "--db-path",
        type=str,
        default="data/papers/metadata/papers.db",
        help="Database path"
    )

    args = parser.parse_args()

    run_benchmark_on_file(args.dataset, args.db_path)

    print(f"\n{'='*80}")
    print(f"✅ BENCHMARK COMPLETE!")
    print(f"={'='*80}\n")


if __name__ == "__main__":
    main()
