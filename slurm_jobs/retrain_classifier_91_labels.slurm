#!/bin/bash
#SBATCH --job-name=retrain_classifier_91
#SBATCH --partition=day
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=logs/retrain_classifier_91_%j.log
#SBATCH --error=logs/retrain_classifier_91_%j.err

# Load required modules
module load miniconda
conda activate claude-env

# Set Python path
export PYTHONPATH="/home/btd8/llm-paper-analyze/scripts:$PYTHONPATH"

# Create logs directory if it doesn't exist
mkdir -p logs

# Print job information
echo "=== JOB INFORMATION ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURMD_NODENAME"
echo "Started at: $(date)"
echo "Working directory: $(pwd)"
echo "Memory allocated: ${SLURM_MEM_PER_NODE}MB"
echo "CPUs allocated: $SLURM_CPUS_PER_TASK"

# Print environment info
echo -e "\n=== ENVIRONMENT ==="
echo "Python: $(which python3)"
echo "Conda environment: $CONDA_DEFAULT_ENV"
echo "Available memory: $(free -h | grep '^Mem')"

# Check input files exist
echo -e "\n=== INPUT FILES CHECK ==="
if [ -f "/home/btd8/llm-paper-analyze/data/manylatents_benchmark/embedding_metrics.csv" ]; then
    echo "✓ Metrics file exists"
    wc -l /home/btd8/llm-paper-analyze/data/manylatents_benchmark/embedding_metrics.csv
else
    echo "✗ Metrics file missing"
    exit 1
fi

if [ -f "/home/btd8/Documents/phate_labels_rich.csv" ]; then
    echo "✓ Main labels file exists"
    wc -l /home/btd8/Documents/phate_labels_rich.csv
else
    echo "✗ Main labels file missing"
    exit 1
fi

if [ -f "/home/btd8/Documents/phate_labels_rich_remainders.csv" ]; then
    echo "✓ Remainder labels file exists"
    wc -l /home/btd8/Documents/phate_labels_rich_remainders.csv
else
    echo "✗ Remainder labels file missing"
    exit 1
fi

# Run the training script
echo -e "\n=== STARTING TRAINING ==="
cd /home/btd8/llm-paper-analyze

python3 scripts/train_classifier_all_labels.py

# Check if training completed successfully
if [ $? -eq 0 ]; then
    echo -e "\n=== TRAINING COMPLETED SUCCESSFULLY ==="

    # Print output files
    echo "Output files created:"
    if [ -d "data/manylatents_benchmark/ml_results_91_labels" ]; then
        ls -la data/manylatents_benchmark/ml_results_91_labels/
    fi

    # Print summary if available
    if [ -f "data/manylatents_benchmark/ml_results_91_labels/training_summary_91_labels.json" ]; then
        echo -e "\n=== TRAINING SUMMARY ==="
        python3 -c "
import json
with open('data/manylatents_benchmark/ml_results_91_labels/training_summary_91_labels.json') as f:
    summary = json.load(f)
print(f'Samples used: {summary[\"n_samples\"]}')
print(f'Features used: {summary[\"n_features\"]}')
print(f'Classes: {summary[\"n_classes\"]}')
print(f'Best model: {summary[\"best_model\"]}')
print(f'Best accuracy: {summary[\"best_accuracy\"]:.4f}')
print('\\nLabel distribution:')
for label, count in summary['label_distribution'].items():
    print(f'  {label}: {count}')
"
    fi

else
    echo -e "\n=== TRAINING FAILED ==="
    echo "Check the log file for errors"
    exit 1
fi

echo -e "\n=== JOB COMPLETED ==="
echo "Finished at: $(date)"
echo "Duration: $SECONDS seconds"

# Print resource usage
echo -e "\n=== RESOURCE USAGE ==="
echo "Peak memory usage:"
grep "MaxRSS" /proc/$$/status 2>/dev/null || echo "Memory info not available"

# Final status
echo -e "\n=== FINAL STATUS ==="
echo "Exit code: $?"
echo "Log file: logs/retrain_classifier_91_${SLURM_JOB_ID}.log"