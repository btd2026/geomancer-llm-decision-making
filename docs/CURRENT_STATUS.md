# Current Status - CELLxGENE PHATE Pipeline

**Date:** November 10, 2025
**Status:** Subsampling in progress

## Current Situation

### ✅ What's Working

1. **ManyLatents Setup** - Complete
   - Environment configured with `uv`
   - 101 experiment configs generated
   - PHATE algorithm configured correctly

2. **Problem Identified** - Memory Issues
   - CELLxGENE datasets are **HUGE** (up to 18GB files, millions of cells)
   - Even 64GB RAM caused Out Of Memory (OOM) errors
   - All 101 initial PHATE jobs failed due to OOM

3. **Solution Implemented** - Subsampling
   - Created subsampling pipeline to reduce datasets to 50,000 cells max
   - **Currently running:** Job ID 2858992

## Active Jobs

### Subsampling Job (Job ID: 2858992)

**Status:** 🔄 **Running** (5 concurrent jobs)

```
Configuration:
- Input:  /nfs/roberts/project/pi_sk2433/shared/Geomancer_2025_Data/processed/
- Output: /nfs/roberts/project/pi_sk2433/shared/Geomancer_2025_Data/subsampled/
- Max cells: 50,000 per dataset
- Memory: 96GB per job
- Concurrent: 5 jobs
- Total: 101 datasets
```

**Progress:**
```bash
# Check status
squeue -u $USER

# Check completed files
ls /nfs/roberts/project/pi_sk2433/shared/Geomancer_2025_Data/subsampled/ | wc -l
```

**Expected completion:** 8-12 hours from start (started ~23:50)

## Next Steps

### 1. Monitor Subsampling ⏳

Wait for all 101 datasets to be subsampled.

**Check progress:**
```bash
# Job status
squeue -u $USER

# Completed count
ls /nfs/roberts/project/pi_sk2433/shared/Geomancer_2025_Data/subsampled/*.h5ad 2>/dev/null | wc -l

# Should show: 101 when complete
```

### 2. Regenerate Experiment Configs

Once subsampling completes, update configs to point to subsampled data:

```bash
cd /home/btd8/llm-paper-analyze

# Regenerate with subsampled data directory (now default)
python3 scripts/generate_manylatents_configs.py

# Or explicitly specify:
python3 scripts/generate_manylatents_configs.py \
    --data-dir /nfs/roberts/project/pi_sk2433/shared/Geomancer_2025_Data/subsampled
```

### 3. Run PHATE on Subsampled Data

Submit the updated PHATE jobs:

```bash
# Clean up old failed outputs
rm -rf /nfs/roberts/project/pi_sk2433/shared/Geomancer_2025_Data/manylatents_outputs/*

# Submit PHATE jobs
sbatch run_manylatents_array.slurm
```

**New configuration (optimized for subsampled data):**
- Memory: 32GB (plenty for 50K cells)
- CPUs: 4
- Concurrent: 15 jobs
- Time: 4 hours per job

**Expected timeline:** 6-8 hours for all 101 datasets

### 4. Verify and Collect Results

```bash
# Check completion status
python3 scripts/check_manylatents_status.py

# Browse results
ls /nfs/roberts/project/pi_sk2433/shared/Geomancer_2025_Data/manylatents_outputs/

# Example result structure:
# manylatents_outputs/
# ├── Blood_d86edd6a/
# │   ├── phate_Blood_d86edd6a.csv    # PHATE embeddings
# │   ├── phate_Blood_d86edd6a.png    # Visualization
# │   └── metrics.yaml                # Quality metrics
# └── ...
```

## Why Subsampling?

### Problem
- Largest file: **18GB** → ~50-100GB in memory
- Even 64GB RAM failed
- Processing time: hours per dataset

### Solution
- **50,000 cells** is scientifically valid for manifold learning
- Fits comfortably in 32GB RAM
- **10-100x faster** PHATE execution
- **Reproducible** (fixed seed=42)

### Scientific Justification
1. PHATE learns the data manifold - captured well with 50K representative cells
2. 50K cells provides excellent statistical power
3. Many landmark single-cell papers use similar sizes
4. Enables practical completion of all 101 datasets

## Timeline Summary

| Phase | Status | Duration |
|-------|--------|----------|
| **Setup** | ✅ Complete | Done |
| **Initial PHATE attempt** | ❌ Failed (OOM) | Done |
| **Subsampling** | 🔄 Running | 8-12 hours |
| **Regenerate configs** | ⏳ Pending | 5 minutes |
| **PHATE on subsampled** | ⏳ Pending | 6-8 hours |
| **Collect results** | ⏳ Pending | - |

**Total remaining time:** ~16-24 hours from now

## Files and Documentation

### Key Files
- **Subsampling script:** `scripts/subsample_datasets.py`
- **Subsampling job:** `run_subsampling.slurm`
- **Config generator:** `scripts/generate_manylatents_configs.py`
- **PHATE job:** `run_manylatents_array.slurm`
- **Status checker:** `scripts/check_manylatents_status.py`

### Documentation
- **Setup guide:** `MANYLATENTS_SETUP.md`
- **Subsampling strategy:** `SUBSAMPLING_STRATEGY.md`
- **Setup complete:** `SETUP_COMPLETE.md`
- **This file:** `CURRENT_STATUS.md`

## Monitoring Commands

```bash
# Check running jobs
squeue -u $USER

# Check subsampling progress
ls /nfs/roberts/project/pi_sk2433/shared/Geomancer_2025_Data/subsampled/ | wc -l

# Check specific log
cat /nfs/roberts/project/pi_sk2433/shared/Geomancer_2025_Data/logs/subsample_2858992_0.out

# Check for errors
cat /nfs/roberts/project/pi_sk2433/shared/Geomancer_2025_Data/logs/subsample_2858992_0.err

# Full status check
python3 scripts/check_manylatents_status.py
```

## What to Expect

### After Subsampling Completes
You'll have 101 H5AD files in the `subsampled/` directory, each with ≤50,000 cells.

### After PHATE Completes
You'll have 101 organized result directories with:
- **CSV files:** PHATE coordinates for visualization
- **PNG files:** 2D PHATE plots
- **Metrics:** Quality assessment

All organized and ready for your presentation! 🎉
