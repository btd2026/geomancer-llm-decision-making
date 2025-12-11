# ManyLatents CELLxGENE Pipeline Setup

This document explains how to run PHATE dimensionality reduction on all 101 CELLxGENE datasets using the manyLatents framework.

## Overview

We've set up a complete pipeline to run PHATE experiments on all your H5AD datasets with organized output for presentation. The pipeline includes:

1. **101 experiment configurations** - One for each CELLxGENE dataset
2. **SLURM array job** - Run all experiments concurrently on the cluster
3. **Python sequential runner** - Alternative for local or sequential execution
4. **Organized outputs** - All results saved to shared directory with proper structure

## Directory Structure

```
/home/btd8/manylatents/
├── manylatents/configs/
│   ├── data/
│   │   └── cellxgene_dataset.yaml          # Data config for H5AD files
│   └── experiment/
│       └── cellxgene/                       # 101 experiment configs
│           ├── A_Single_Cell_Atlas_of_Mouse_White_Adipose_Tissue_a2da8d7b.yaml
│           ├── Blood_d86edd6a.yaml
│           └── ...
├── cellxgene_experiments.yaml              # Master list of experiments
└── .venv/                                   # Python environment (already set up)

/nfs/roberts/project/pi_sk2433/shared/Geomancer_2025_Data/
├── processed/                               # Input H5AD files (101 files)
├── manylatents_outputs/                     # Output directory
│   ├── dataset_name_1/
│   │   ├── phate_dataset_name_1.csv         # PHATE embeddings
│   │   ├── phate_dataset_name_1.png         # Visualization
│   │   └── metrics.yaml                     # Quality metrics
│   └── ...
└── logs/                                    # SLURM logs
    ├── phate_JOBID_0.out
    ├── phate_JOBID_1.out
    └── ...
```

## Quick Start

### Option 1: SLURM Array Job (Recommended for Cluster)

Run all 101 experiments concurrently using SLURM:

```bash
# Create logs directory
mkdir -p /nfs/roberts/project/pi_sk2433/shared/Geomancer_2025_Data/logs

# Submit array job
cd /home/btd8/llm-paper-analyze
sbatch run_manylatents_array.slurm
```

This will run all experiments in parallel. Each experiment will:
- Load the H5AD file
- Run PHATE dimensionality reduction (2D)
- Save embeddings as CSV
- Generate visualization plot
- Save quality metrics

**Monitor progress:**
```bash
# Check job status
squeue -u $USER

# Check specific job
squeue -j JOBID

# View logs (real-time)
tail -f /nfs/roberts/project/pi_sk2433/shared/Geomancer_2025_Data/logs/phate_JOBID_*.out
```

### Option 2: Python Sequential Runner

For testing or running on a single machine:

```bash
# Activate environment
export PATH="$HOME/.local/bin:$PATH"
cd /home/btd8/manylatents
source .venv/bin/activate

# Dry run (see what will be executed)
python3 /home/btd8/llm-paper-analyze/scripts/run_all_manylatents.py --dry-run

# Run first 5 experiments (for testing)
python3 /home/btd8/llm-paper-analyze/scripts/run_all_manylatents.py --start 0 --end 5 --verbose

# Run all experiments sequentially
python3 /home/btd8/llm-paper-analyze/scripts/run_all_manylatents.py --verbose

# Run with 4 parallel workers
python3 /home/btd8/llm-paper-analyze/scripts/run_all_manylatents.py --parallel 4 --verbose
```

### Option 3: Run Single Experiment

Test a single experiment:

```bash
export PATH="$HOME/.local/bin:$PATH"
cd /home/btd8/manylatents
source .venv/bin/activate

# Example: Run PHATE on blood dataset
python3 -m manylatents.main \
    experiment=cellxgene/Blood_d86edd6a \
    logger=none
```

## Output Organization

Each experiment creates a dedicated output directory:

```
/nfs/roberts/project/pi_sk2433/shared/Geomancer_2025_Data/manylatents_outputs/Blood_d86edd6a/
├── phate_Blood_d86edd6a.csv              # Embeddings (2 columns: PHATE1, PHATE2)
├── phate_Blood_d86edd6a.png              # 2D visualization
├── metrics.yaml                           # Quality metrics
└── .hydra/                                # Hydra configuration logs
    ├── config.yaml
    └── hydra.yaml
```

### Embeddings CSV Format

```csv
PHATE1,PHATE2,cell_type
-12.34,5.67,T_cell
-11.23,6.78,B_cell
...
```

### Metrics YAML Format

```yaml
test_metric:
  k: 25
  score: 0.85
```

## Customization

### Modify PHATE Parameters

Edit the generated config files or modify the generator script:

```yaml
# In /home/btd8/manylatents/manylatents/configs/experiment/cellxgene/YOUR_DATASET.yaml
algorithms:
  latent:
    n_components: 2       # Number of dimensions
    knn: 5                # Number of nearest neighbors
    decay: 40             # Decay rate
    t: "auto"             # Diffusion time (or integer)
```

### Change Output Location

Edit `run_manylatents_array.slurm` or `run_all_manylatents.py`:

```bash
# Change this line:
OUTPUT_BASE = Path("/nfs/roberts/project/pi_sk2433/shared/Geomancer_2025_Data/manylatents_outputs")
```

### Add More Algorithms

To run additional algorithms (UMAP, t-SNE, etc.), modify the experiment configs:

```yaml
defaults:
  - override /algorithms/latent: umap  # Change from phate to umap
```

Or create new configs using the generator script as a template.

## Troubleshooting

### Memory Issues

If experiments fail with OOM errors, adjust SLURM memory:

```bash
# In run_manylatents_array.slurm
#SBATCH --mem=64G  # Increase from 32G
```

### Timeout Issues

For large datasets that take longer than 2 hours:

```bash
# In run_manylatents_array.slurm
#SBATCH --time=6:00:00  # Increase from 2:00:00
```

### Failed Experiments

Check logs for errors:

```bash
# Find failed jobs
grep -l "Error" /nfs/roberts/project/pi_sk2433/shared/Geomancer_2025_Data/logs/*.err

# View specific error
cat /nfs/roberts/project/pi_sk2433/shared/Geomancer_2025_Data/logs/phate_JOBID_5.err
```

Re-run specific experiments:

```bash
# Re-run experiment #5
export PATH="$HOME/.local/bin:$PATH"
cd /home/btd8/manylatents
source .venv/bin/activate

# Get experiment name from master list
EXPERIMENT=$(python3 -c "import yaml; data=yaml.safe_load(open('cellxgene_experiments.yaml')); print(data['experiments'][5]['name'])")

# Run it
python3 -m manylatents.main experiment=cellxgene/${EXPERIMENT} logger=none
```

## Results Summary

After all experiments complete, you can generate a summary:

```python
import pandas as pd
from pathlib import Path
import yaml

output_dir = Path("/nfs/roberts/project/pi_sk2433/shared/Geomancer_2025_Data/manylatents_outputs")
results = []

for exp_dir in output_dir.iterdir():
    if exp_dir.is_dir():
        metrics_file = exp_dir / "metrics.yaml"
        if metrics_file.exists():
            with open(metrics_file) as f:
                metrics = yaml.safe_load(f)
            results.append({
                "dataset": exp_dir.name,
                "metric_score": metrics.get("test_metric", {}).get("score", None)
            })

df = pd.DataFrame(results)
print(df.describe())
df.to_csv("phate_results_summary.csv", index=False)
```

## Next Steps

1. **Run experiments**: Start with SLURM array job or sequential runner
2. **Monitor progress**: Check logs and output directories
3. **Analyze results**: Visualize embeddings and compare metrics
4. **Create presentation**: Use organized outputs for your presentation

## Files Created

- `/home/btd8/manylatents/cellxgene_experiments.yaml` - Master experiment list
- `/home/btd8/manylatents/manylatents/configs/data/cellxgene_dataset.yaml` - Data config
- `/home/btd8/manylatents/manylatents/configs/experiment/cellxgene/*.yaml` - 101 experiment configs
- `/home/btd8/llm-paper-analyze/run_manylatents_array.slurm` - SLURM array job script
- `/home/btd8/llm-paper-analyze/scripts/generate_manylatents_configs.py` - Config generator
- `/home/btd8/llm-paper-analyze/scripts/run_all_manylatents.py` - Python runner script

## Additional Resources

- [ManyLatents Documentation](https://github.com/manylatents)
- [PHATE Algorithm](https://github.com/KrishnaswamyLab/PHATE)
- [Hydra Configuration](https://hydra.cc/)
