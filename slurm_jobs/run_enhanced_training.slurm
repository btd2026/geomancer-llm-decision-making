#!/bin/bash
#SBATCH --job-name=enhanced_train
#SBATCH --output=logs/enhanced_train_%j.log
#SBATCH --error=logs/enhanced_train_%j.err
#SBATCH --time=02:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --partition=devel

# Ensure logs directory exists
mkdir -p logs

echo "Starting enhanced classifier training at $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on node: $(hostname)"
echo "Memory allocated: 16G"
echo "CPUs allocated: 4"

# Change to working directory
cd /home/btd8/llm-paper-analyze

# Set environment variables
export PYTHONPATH=$PYTHONPATH:/home/btd8/llm-paper-analyze/scripts
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "Training enhanced structure classifier with:"
echo "- Feature selection using univariate F-test and Random Forest importance"
echo "- Multiple classifier comparison (SVM, Random Forest, Gradient Boosting, etc.)"
echo "- Cross-validation with stratified K-fold or Leave-One-Out"
echo "- Enhanced feature set from PHATE embeddings"

# Check if enhanced features are ready
if [ ! -f "data/enhanced_phate_metrics.csv" ]; then
    echo "⚠️  Enhanced features not found. Training with original features only."
    echo "    To get enhanced results, ensure enhanced feature extraction completes first."
fi

python3 scripts/train_structure_classifier_v2.py

echo "Enhanced training completed at $(date)"

# Check outputs
echo ""
echo "=== TRAINING OUTPUTS ==="
if [ -f "models/enhanced_structure_classifier.pkl" ]; then
    echo "✓ Enhanced model saved successfully"
    ls -lh models/enhanced_structure_classifier.pkl
else
    echo "✗ Enhanced model not created"
fi

if [ -f "data/enhanced_structure_predictions.csv" ]; then
    echo "✓ Enhanced predictions generated"
    echo "✓ Predictions for $(wc -l < data/enhanced_structure_predictions.csv) datasets"
    echo "✓ File size: $(du -h data/enhanced_structure_predictions.csv | cut -f1)"
else
    echo "ℹ️  Enhanced predictions not generated (enhanced features may not be ready)"
fi