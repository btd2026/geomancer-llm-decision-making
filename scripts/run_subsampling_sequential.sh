#!/bin/bash
# Sequential subsampling script - more reliable than SLURM array
set -e

export WANDB_API_KEY="c2504bb9a04dfbf35d668f90ec5893001673c128"
export PATH="$HOME/.local/bin:$PATH"

cd /home/btd8/manylatents
source .venv/bin/activate

echo "Starting sequential subsampling at $(date)"
echo "This will process all 101 datasets, skipping 22 already completed"
echo "Estimated time: 8-16 hours"
echo "Output: /nfs/roberts/project/pi_sk2433/shared/Geomancer_2025_Data/subsampled/"
echo "================================================"

python3 /home/btd8/llm-paper-analyze/scripts/subsample_datasets.py \
    --input-dir /nfs/roberts/project/pi_sk2433/shared/Geomancer_2025_Data/processed \
    --output-dir /nfs/roberts/project/pi_sk2433/shared/Geomancer_2025_Data/subsampled \
    --max-cells 50000 \
    2>&1 | tee /nfs/roberts/project/pi_sk2433/shared/Geomancer_2025_Data/logs/subsample_sequential.log

echo "================================================"
echo "Subsampling completed at $(date)"
