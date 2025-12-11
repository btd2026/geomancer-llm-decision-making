#!/bin/bash
#SBATCH --job-name=phate_cellxgene
#SBATCH --output=/nfs/roberts/project/pi_sk2433/shared/Geomancer_2025_Data/logs/phate_%A_%a.out
#SBATCH --error=/nfs/roberts/project/pi_sk2433/shared/Geomancer_2025_Data/logs/phate_%A_%a.err
#SBATCH --array=0-100%15  # Run 101 jobs, max 15 concurrent (subsampled data)
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G  # Sufficient for 50K cells
#SBATCH --time=4:00:00
#SBATCH --partition=day

# Exit on error
set -e

# Load necessary modules (adjust based on your cluster)
# module load python/3.11
# module load cuda/12.1  # if using GPU

# Create output and logs directories
mkdir -p /nfs/roberts/project/pi_sk2433/shared/Geomancer_2025_Data/logs
mkdir -p /nfs/roberts/project/pi_sk2433/shared/Geomancer_2025_Data/manylatents_outputs

# Activate virtual environment
export PATH="$HOME/.local/bin:$PATH"
cd /home/btd8/manylatents
source .venv/bin/activate

# Read experiment list
EXPERIMENT_LIST="/home/btd8/manylatents/cellxgene_experiments.yaml"

# Extract experiment name for this array task
# Parse YAML to get the experiment name at index $SLURM_ARRAY_TASK_ID
EXPERIMENT_NAME=$(python3 -c "
import yaml
import sys
with open('${EXPERIMENT_LIST}', 'r') as f:
    data = yaml.safe_load(f)
    experiments = data['experiments']
    if ${SLURM_ARRAY_TASK_ID} < len(experiments):
        print(experiments[${SLURM_ARRAY_TASK_ID}]['name'])
    else:
        sys.exit(1)
")

if [ -z "$EXPERIMENT_NAME" ]; then
    echo "Error: Could not find experiment for task ID ${SLURM_ARRAY_TASK_ID}"
    exit 1
fi

echo "========================================="
echo "Running experiment: ${EXPERIMENT_NAME}"
echo "Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Host: $(hostname)"
echo "Date: $(date)"
echo "========================================="

# Run the experiment
python3 -m manylatents.main \
    experiment=cellxgene/${EXPERIMENT_NAME} \
    logger=none \
    hydra.run.dir=/nfs/roberts/project/pi_sk2433/shared/Geomancer_2025_Data/manylatents_outputs/${EXPERIMENT_NAME}

echo "========================================="
echo "Experiment completed: ${EXPERIMENT_NAME}"
echo "Date: $(date)"
echo "========================================="
