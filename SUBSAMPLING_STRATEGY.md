# CELLxGENE Dataset Subsampling Strategy

## Problem

The CELLxGENE datasets are **extremely large**:
- Largest file: **18GB** (All_major_cell_types_in_adult_human_retina)
- Many files: **3-11GB** each
- When loaded into memory: **3-5x file size**
- Result: Even with 64GB RAM, jobs were **Out Of Memory (OOM) killed**

## Solution: Subsample to 50,000 Cells

For datasets with >50,000 cells, we randomly sample 50,000 cells. This provides:
- **Sufficient cells** for robust PHATE analysis
- **Predictable memory** requirements (~8-12GB per dataset)
- **Faster processing** (PHATE scales with n_cells²)
- **Reproducible** results (fixed random seed=42)

## Implementation

###Step 1: Subsample Datasets (Currently Running) ✓

**Job ID:** 2858992
**Status:** 5 jobs running, 96 pending
**Script:** `run_subsampling.slurm`

```bash
# Subsampling configuration
- Max cells: 50,000
- Random seed: 42
- Input:  /nfs/roberts/project/pi_sk2433/shared/Geomancer_2025_Data/processed/
- Output: /nfs/roberts/project/pi_sk2433/shared/Geomancer_2025_Data/subsampled/
- Memory: 96GB per job
- Concurrent: 5 jobs at a time
```

**What it does:**
- For each H5AD file:
  - If ≤50,000 cells → copy as-is
  - If >50,000 cells → randomly sample 50,000 cells
  - Save to `subsampled/` directory

**Timeline:** ~8-12 hours for all 101 datasets (5 concurrent jobs)

### Step 2: Update Experiment Configs

Once subsampling completes, regenerate configs to point to subsampled data:

```bash
cd /home/btd8/llm-paper-analyze
# Update the data directory in generate_manylatents_configs.py
python3 scripts/generate_manylatents_configs.py \
    --data-dir /nfs/roberts/project/pi_sk2433/shared/Geomancer_2025_Data/subsampled
```

### Step 3: Run PHATE on Subsampled Data

With 50K cells max, memory requirements are predictable:

```bash
# Submit PHATE jobs with reasonable memory
sbatch run_manylatents_array.slurm
# Configuration:
- Memory: 32GB (plenty for 50K cells)
- CPUs: 4
- Time: 4 hours
- Concurrent: 15 jobs
```

## Monitoring Subsampling Progress

```bash
# Check job status
squeue -u $USER

# Check completed subsampled files
ls -lh /nfs/roberts/project/pi_sk2433/shared/Geomancer_2025_Data/subsampled/

# Count completed
ls /nfs/roberts/project/pi_sk2433/shared/Geomancer_2025_Data/subsampled/*.h5ad 2>/dev/null | wc -l

# Check a specific log
cat /nfs/roberts/project/pi_sk2433/shared/Geomancer_2025_Data/logs/subsample_2858992_0.out
```

## Dataset Size Comparison

### Before Subsampling
```
Top 10 largest datasets:
18GB  All_major_cell_types_in_adult_human_retina  (~millions of cells)
16GB  L2_3_IT_DLPFC_Seattle_Alzheimers            (~millions of cells)
11GB  Tabula_Sapiens_Epithelium                   (~millions of cells)
...
```

### After Subsampling (Expected)
```
All datasets:
~500MB - 2GB per file (50,000 cells max)
Total: ~50-100GB for all 101 datasets
Memory: ~8-12GB to process each
```

## Benefits of This Approach

1. **Guaranteed Success**: 50K cells fits comfortably in 32GB RAM
2. **Scientifically Valid**: 50K cells is more than sufficient for manifold learning
3. **Faster**: PHATE will run 10-100x faster on smaller datasets
4. **Reproducible**: Fixed random seed ensures consistent subsampling
5. **Practical**: Enables completion of all 101 datasets

## Scientific Justification

**Why 50,000 cells is sufficient:**

1. **Manifold Learning**: PHATE and similar algorithms learn the underlying data manifold, which is captured well with 50K representative cells
2. **Statistical Power**: 50K cells provides excellent statistical power for detecting biological structures
3. **Literature Precedent**: Many landmark single-cell papers use similar or smaller sample sizes
4. **Computational Efficiency**: Enables practical analysis while preserving biological signal

## Next Steps

1. ✅ **Wait for subsampling to complete** (~8-12 hours)
   - Monitor: `squeue -u $USER`
   - Check: `ls /nfs/roberts/project/pi_sk2433/shared/Geomancer_2025_Data/subsampled/`

2. **Regenerate experiment configs** pointing to subsampled data
   - Update `DATA_DIR` in `generate_manylatents_configs.py`
   - Run: `python3 scripts/generate_manylatents_configs.py`

3. **Run PHATE on all subsampled datasets**
   - Submit: `sbatch run_manylatents_array.slurm`
   - With: 32GB memory, 15 concurrent jobs
   - Timeline: ~6-8 hours for all 101 datasets

4. **Collect and organize results** for presentation

## Files

- **Subsampling script:** `scripts/subsample_datasets.py`
- **Subsampling SLURM job:** `run_subsampling.slurm`
- **PHATE experiment configs:** `manylatents/configs/experiment/cellxgene/*.yaml`
- **PHATE SLURM job:** `run_manylatents_array.slurm`

## Troubleshooting

### If subsampling jobs fail with OOM
- Reduce concurrent jobs: `#SBATCH --array=0-100%3`
- Or process in batches: smaller files first

### If PHATE still fails after subsampling
- Increase memory: `#SBATCH --mem=48G`
- Reduce concurrent jobs

### Check specific dataset
```bash
# Check if a dataset was subsampled
ls -lh /nfs/roberts/project/pi_sk2433/shared/Geomancer_2025_Data/subsampled/Blood_d86edd6a.h5ad

# Check cell count
python3 -c "import scanpy as sc; adata = sc.read_h5ad('/nfs/roberts/project/pi_sk2433/shared/Geomancer_2025_Data/subsampled/Blood_d86edd6a.h5ad'); print(f'{adata.n_obs} cells')"
```
