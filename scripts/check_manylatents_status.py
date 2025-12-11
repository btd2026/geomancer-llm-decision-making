#!/usr/bin/env python3
"""
Check the status of manyLatents experiments.

This script verifies the setup and checks which experiments have completed.
"""

from pathlib import Path
import yaml
import sys

MANYLATENTS_DIR = Path("/home/btd8/manylatents")
DATA_DIR = Path("/nfs/roberts/project/pi_sk2433/shared/Geomancer_2025_Data/processed")
OUTPUT_BASE = Path("/nfs/roberts/project/pi_sk2433/shared/Geomancer_2025_Data/manylatents_outputs")
EXPERIMENT_LIST = MANYLATENTS_DIR / "cellxgene_experiments.yaml"

def check_setup():
    """Verify the setup is complete."""
    print("=" * 60)
    print("SETUP VERIFICATION")
    print("=" * 60)

    checks = {
        "ManyLatents directory": MANYLATENTS_DIR.exists(),
        "ManyLatents virtual environment": (MANYLATENTS_DIR / ".venv").exists(),
        "Experiment list": EXPERIMENT_LIST.exists(),
        "Data directory": DATA_DIR.exists(),
        "Output directory": OUTPUT_BASE.exists(),
    }

    for check, status in checks.items():
        icon = "✓" if status else "✗"
        print(f"{icon} {check}: {'OK' if status else 'MISSING'}")

    if not all(checks.values()):
        print("\n⚠ Some components are missing!")
        return False

    # Count H5AD files
    h5ad_files = list(DATA_DIR.glob("*.h5ad"))
    print(f"\n✓ Found {len(h5ad_files)} H5AD files in {DATA_DIR}")

    # Count experiment configs
    if EXPERIMENT_LIST.exists():
        with open(EXPERIMENT_LIST, 'r') as f:
            data = yaml.safe_load(f)
            experiments = data['experiments']
        print(f"✓ Found {len(experiments)} experiment configs")

    print("\n✓ Setup is complete!")
    return True

def check_progress():
    """Check experiment progress."""
    print("\n" + "=" * 60)
    print("EXPERIMENT PROGRESS")
    print("=" * 60)

    if not EXPERIMENT_LIST.exists():
        print("✗ Experiment list not found")
        return

    with open(EXPERIMENT_LIST, 'r') as f:
        data = yaml.safe_load(f)
        experiments = data['experiments']

    completed = []
    running = []
    pending = []

    for exp in experiments:
        exp_name = exp['name']
        exp_dir = OUTPUT_BASE / exp_name

        if exp_dir.exists():
            # Check for output files
            csv_file = list(exp_dir.glob("*.csv"))
            png_file = list(exp_dir.glob("*.png"))

            if csv_file and png_file:
                completed.append(exp_name)
            else:
                running.append(exp_name)
        else:
            pending.append(exp_name)

    print(f"Completed: {len(completed)}/{len(experiments)}")
    print(f"Running:   {len(running)}/{len(experiments)}")
    print(f"Pending:   {len(pending)}/{len(experiments)}")

    if completed:
        print(f"\n✓ {len(completed)} experiments completed successfully")
        if len(completed) <= 10:
            print("  Completed experiments:")
            for exp in completed[:10]:
                print(f"    - {exp}")

    if running:
        print(f"\n⚠ {len(running)} experiments in progress (or incomplete)")
        if len(running) <= 5:
            print("  Running/incomplete experiments:")
            for exp in running[:5]:
                print(f"    - {exp}")

    if pending:
        print(f"\n⏳ {len(pending)} experiments pending")

    # Show example output
    if completed:
        example = OUTPUT_BASE / completed[0]
        print(f"\nExample output directory: {example}")
        if example.exists():
            files = list(example.glob("*"))
            print(f"  Files: {', '.join([f.name for f in files[:5]])}")

def main():
    print("\nManyLatents CELLxGENE Pipeline Status Check")
    print("=" * 60)

    setup_ok = check_setup()
    if setup_ok:
        check_progress()

    print("\n" + "=" * 60)
    print("NEXT STEPS")
    print("=" * 60)
    print("\nTo start experiments:")
    print("  1. SLURM array job:  sbatch run_manylatents_array.slurm")
    print("  2. Python runner:    python3 scripts/run_all_manylatents.py --verbose")
    print("  3. Single test:      cd /home/btd8/manylatents && source .venv/bin/activate")
    print("                       python3 -m manylatents.main experiment=cellxgene/Blood_d86edd6a")
    print("\nFor more info, see: MANYLATENTS_SETUP.md")

if __name__ == "__main__":
    main()
