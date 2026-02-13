#!/usr/bin/env python3
"""
Run all manyLatents experiments sequentially or in parallel.

This script provides a way to run all CELLxGENE PHATE experiments
either sequentially or using multiprocessing.
"""

import subprocess
import yaml
from pathlib import Path
import argparse
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
import time

MANYLATENTS_DIR = Path("/home/btd8/manylatents")
EXPERIMENT_LIST = MANYLATENTS_DIR / "cellxgene_experiments.yaml"
OUTPUT_BASE = Path("/nfs/roberts/project/pi_sk2433/shared/Geomancer_2025_Data/manylatents_outputs")

def run_experiment(exp_name, verbose=False):
    """Run a single experiment."""
    cmd = [
        "python3", "-m", "manylatents.main",
        f"experiment=cellxgene/{exp_name}",
        "logger=none",
        f"hydra.run.dir={OUTPUT_BASE}/{exp_name}"
    ]

    if verbose:
        print(f"\n{'='*60}")
        print(f"Running: {exp_name}")
        print(f"Command: {' '.join(cmd)}")
        print(f"{'='*60}\n")

    try:
        result = subprocess.run(
            cmd,
            cwd=str(MANYLATENTS_DIR),
            capture_output=True,
            text=True,
            timeout=3600  # 1 hour timeout per experiment
        )

        if result.returncode == 0:
            if verbose:
                print(f"✓ Completed: {exp_name}")
            return exp_name, "success", None
        else:
            error_msg = f"Exit code {result.returncode}"
            if verbose:
                print(f"✗ Failed: {exp_name} - {error_msg}")
                print(f"stderr: {result.stderr[-500:]}")  # Last 500 chars
            return exp_name, "failed", error_msg

    except subprocess.TimeoutExpired:
        if verbose:
            print(f"✗ Timeout: {exp_name}")
        return exp_name, "timeout", "Experiment timed out after 1 hour"
    except Exception as e:
        if verbose:
            print(f"✗ Error: {exp_name} - {str(e)}")
        return exp_name, "error", str(e)

def main():
    parser = argparse.ArgumentParser(description="Run manyLatents experiments")
    parser.add_argument("--parallel", type=int, default=1,
                       help="Number of parallel processes (default: 1 for sequential)")
    parser.add_argument("--start", type=int, default=0,
                       help="Start index (inclusive)")
    parser.add_argument("--end", type=int, default=None,
                       help="End index (exclusive)")
    parser.add_argument("--verbose", action="store_true",
                       help="Print detailed progress")
    parser.add_argument("--dry-run", action="store_true",
                       help="Print experiments without running")
    args = parser.parse_args()

    # Load experiments
    if not EXPERIMENT_LIST.exists():
        print(f"Error: Experiment list not found: {EXPERIMENT_LIST}")
        print("Run generate_manylatents_configs.py first!")
        sys.exit(1)

    with open(EXPERIMENT_LIST, 'r') as f:
        data = yaml.safe_load(f)
        experiments = data['experiments']

    # Select subset
    end = args.end if args.end is not None else len(experiments)
    experiments = experiments[args.start:end]

    print(f"Total experiments: {len(experiments)}")
    print(f"Parallel workers: {args.parallel}")
    print(f"Output directory: {OUTPUT_BASE}")

    if args.dry_run:
        print("\nExperiments to run:")
        for i, exp in enumerate(experiments, start=args.start):
            print(f"  {i:3d}. {exp['name']}")
        sys.exit(0)

    # Create output directory
    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

    start_time = time.time()
    results = {"success": [], "failed": [], "timeout": [], "error": []}

    # Run experiments
    if args.parallel == 1:
        # Sequential execution
        for i, exp in enumerate(experiments, start=args.start):
            print(f"\n[{i+1}/{len(experiments)}] Running: {exp['name']}")
            name, status, error = run_experiment(exp['name'], verbose=args.verbose)
            results[status].append((name, error))
    else:
        # Parallel execution
        print(f"\nRunning {len(experiments)} experiments with {args.parallel} workers...")
        with ProcessPoolExecutor(max_workers=args.parallel) as executor:
            futures = {
                executor.submit(run_experiment, exp['name'], args.verbose): exp
                for exp in experiments
            }

            for i, future in enumerate(as_completed(futures), 1):
                exp = futures[future]
                name, status, error = future.result()
                results[status].append((name, error))
                print(f"[{i}/{len(experiments)}] {status.upper()}: {name}")

    # Print summary
    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Total time: {elapsed/60:.1f} minutes")
    print(f"Success: {len(results['success'])}")
    print(f"Failed:  {len(results['failed'])}")
    print(f"Timeout: {len(results['timeout'])}")
    print(f"Errors:  {len(results['error'])}")

    if results['failed'] or results['timeout'] or results['error']:
        print(f"\n{'='*60}")
        print("FAILURES")
        print(f"{'='*60}")
        for status in ['failed', 'timeout', 'error']:
            if results[status]:
                print(f"\n{status.upper()}:")
                for name, error in results[status]:
                    print(f"  - {name}: {error}")

    # Save results
    results_file = OUTPUT_BASE / "run_results.yaml"
    with open(results_file, 'w') as f:
        yaml.dump(results, f)
    print(f"\nResults saved to: {results_file}")

if __name__ == "__main__":
    main()
