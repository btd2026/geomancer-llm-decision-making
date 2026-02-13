#!/bin/bash
#SBATCH --partition=day
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --job-name=extract_labels
#SBATCH --output=logs/extract_labels_%A.log
#SBATCH --error=logs/extract_labels_%A.err

# Extract actual label categories from h5ad files for PHATE gallery

cd /home/btd8/geomancer-llm-decision-making

# Activate conda environment
source /home/btd8/.conda/envs/claude-env/bin/activate

# Run the extraction script
python3 extract_label_categories.py \
    --wandb-metadata wandb_gallery_replit/wandb_metadata.json \
    --gallery-data wandb_gallery_replit/gallery_data.json \
    --subsampled-dir /nfs/roberts/project/pi_sk2433/shared/Geomancer_2025_Data/subsampled \
    --output wandb_gallery_replit/label_categories.json \
    --max-categories 50

echo "Label extraction completed"
echo "Output: wandb_gallery_replit/label_categories.json"
