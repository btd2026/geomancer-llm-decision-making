#!/usr/bin/env python3
"""
Check if enhanced features are ready and run training when available.
"""

import time
import subprocess
from pathlib import Path

def check_features_ready():
    """Check if enhanced features file exists and is non-empty."""
    features_path = Path("/home/btd8/llm-paper-analyze/data/enhanced_phate_metrics.csv")

    if not features_path.exists():
        return False

    # Check if file has content (more than just header)
    try:
        with open(features_path, 'r') as f:
            lines = f.readlines()
            return len(lines) > 1  # Header + at least one data row
    except:
        return False

def run_training():
    """Run the enhanced training script."""
    print("🚀 Enhanced features ready! Starting training...")

    result = subprocess.run([
        'python3',
        '/home/btd8/llm-paper-analyze/scripts/train_structure_classifier_v2.py'
    ], cwd='/home/btd8/llm-paper-analyze')

    if result.returncode == 0:
        print("✅ Training completed successfully!")
    else:
        print("❌ Training failed!")

    return result.returncode

def main():
    print("Checking for enhanced features...")

    max_wait = 1800  # 30 minutes
    check_interval = 30  # 30 seconds

    for i in range(max_wait // check_interval):
        if check_features_ready():
            run_training()
            return

        print(f"⏳ Waiting for enhanced features... ({i*check_interval}s elapsed)")
        time.sleep(check_interval)

    print("⚠️ Timeout waiting for enhanced features. Running with original features only...")
    run_training()

if __name__ == "__main__":
    main()