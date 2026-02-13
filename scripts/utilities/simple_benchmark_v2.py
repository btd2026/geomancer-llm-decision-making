#!/usr/bin/env python3
"""
Simple benchmark - Direct manylatents calls (bypassing manyagents for now).

This proves the concept works end-to-end:
1. Run algorithms directly via manylatents.api
2. Store results in database
3. Show it works

Once this works, we can add the manyagents layer.
"""

import sys
import argparse
import time
import json
import sqlite3
from pathlib import Path

# Add manylatents to path
sys.path.insert(0, '/home/btd8/manylatents')


def run_benchmark(dataset_config, algorithms, db_path):
    """
    Run benchmark: test multiple algorithms on a dataset.
    """
    from manylatents.api import run

    print(f"="*80)
    print(f"PROOF OF CONCEPT BENCHMARK")
    print(f"="*80)

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
                    else:
                        print(f"      {metric_name}: {metric_value}")
            else:
                print(f"   ⚠️  No metrics computed")

            # Store result
            results.append({
                'dataset_name': dataset_config['name'],
                'algorithm_name': algo_name,
                'success': True,
                'embeddings_shape': list(embeddings.shape),
                'execution_time': execution_time,
                'scores': scores,
                'config': algo_config['manylatents_config']
            })

        except Exception as e:
            print(f"   ❌ Exception: {e}")
            import traceback
            traceback.print_exc()

            results.append({
                'dataset_name': dataset_config['name'],
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

    # Create results table if not exists
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS poc_benchmarks (
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
            INSERT INTO poc_benchmarks (
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

    print(f"   ✅ Stored {len(results)} results in 'poc_benchmarks' table")

    # Show total
    cursor.execute('SELECT COUNT(*) FROM poc_benchmarks')
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
        print(f"\n{'Algorithm':<15} {'Time (s)':<12} {'Shape':<20}")
        print(f"{'─'*50}")

        for result in successful:
            algo = result['algorithm_name']
            time_s = result.get('execution_time', 0)
            shape = result.get('embeddings_shape', [])

            print(f"{algo:<15} {time_s:<12.2f} {str(shape):<20}")

    if failed:
        print(f"\nFailed algorithms:")
        for result in failed:
            print(f"   ❌ {result['algorithm_name']}: {result.get('error', 'Unknown')[:60]}...")


def main():
    parser = argparse.ArgumentParser(
        description="Run simple benchmark proof of concept (direct manylatents)"
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default="data/papers/metadata/papers.db",
        help="Path to database"
    )

    args = parser.parse_args()

    # Dataset configuration (using built-in swissroll)
    dataset_config = {
        'name': 'swissroll',
        'description': 'Synthetic swissroll manifold dataset'
    }

    # Algorithm configurations (direct manylatents API calls)
    algorithms = [
        {
            'name': 'PCA',
            'manylatents_config': {
                'data': 'swissroll',
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
                'data': 'swissroll',
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
                'data': 'swissroll',
                'algorithms': {
                    'latent': {
                        '_target_': 'manylatents.algorithms.latent.tsne.TSNEModule',
                        'n_components': 2
                    }
                }
            }
        }
    ]

    # Run benchmark
    results = run_benchmark(dataset_config, algorithms, args.db_path)

    print(f"\n{'='*80}")
    print(f"✅ PROOF OF CONCEPT COMPLETE!")
    print(f"{'='*80}")
    print(f"\n📊 Results:")
    print(f"   - Ran {len(algorithms)} algorithms on {dataset_config['name']} dataset")
    print(f"   - Stored results in {args.db_path}")
    print(f"   - Table: poc_benchmarks")
    print(f"\n💡 Next steps:")
    print(f"   1. Query: sqlite3 {args.db_path} 'SELECT * FROM poc_benchmarks'")
    print(f"   2. Add real GEO datasets")
    print(f"   3. Scale to more algorithms")
    print(f"   4. Train ML recommendation model")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
