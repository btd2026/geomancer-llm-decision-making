# SLURM Guide for Running PHATE on Your Datasets

## What is SLURM?

**SLURM** (Simple Linux Utility for Resource Management) is a job scheduling system used on HPC (High Performance Computing) clusters like Yale's computing infrastructure.

### Why You Need SLURM for This:

**Current Problem:**
- Your datasets are large (200K-500K cells × 30K genes)
- Interactive sessions have limited memory (~8-16GB)
- Processes keep getting killed (OOM - Out of Memory)

**SLURM Solution:**
- Request specific resources (e.g., 32GB RAM, 8 CPU cores)
- Jobs run on dedicated compute nodes with guaranteed resources
- Jobs can run for hours without timing out
- Can submit multiple jobs in parallel

---

## SLURM Basics

### Key Concepts:

1. **Job Script**: A bash script that tells SLURM what to run
2. **Resource Request**: Memory, CPUs, time limit, etc.
3. **Partition**: Queue/group of compute nodes (e.g., `general`, `gpu`, `bigmem`)
4. **Job Submission**: Use `sbatch` to submit jobs
5. **Job Monitoring**: Use `squeue` to check job status

---

## Essential SLURM Commands

```bash
# 1. Submit a job
sbatch my_job.sh

# 2. Check job status
squeue -u $USER
# Shows: JOBID, PARTITION, NAME, STATE, TIME, NODES

# 3. Check specific job
squeue -j <job_id>

# 4. Cancel a job
scancel <job_id>

# 5. Cancel all your jobs
scancel -u $USER

# 6. Check job details
scontrol show job <job_id>

# 7. View completed job info
sacct -j <job_id>

# 8. Check available partitions
sinfo

# 9. Check your recent jobs
sacct -u $USER --starttime=today
```

---

## Job Script Anatomy

Here's a typical SLURM job script:

```bash
#!/bin/bash
#SBATCH --job-name=my_job        # Job name (shows in squeue)
#SBATCH --partition=general      # Which queue to use
#SBATCH --ntasks=1               # Number of tasks (usually 1)
#SBATCH --cpus-per-task=8        # Number of CPU cores
#SBATCH --mem=32G                # Memory allocation
#SBATCH --time=2:00:00           # Time limit (HH:MM:SS)
#SBATCH --output=logs/%x_%j.out  # Output file (%x=name, %j=jobid)
#SBATCH --error=logs/%x_%j.err   # Error file

# Load environment
source ~/.bashrc
conda activate claude-env

# Your commands here
echo "Job started at $(date)"
echo "Running on node: $(hostname)"

python3 my_script.py

echo "Job finished at $(date)"
```

### SBATCH Directives Explained:

- `--job-name`: Give your job a meaningful name
- `--partition`: Choose compute node type (usually `general`)
- `--cpus-per-task`: Number of CPU cores (8 is good for PHATE)
- `--mem`: Total memory (32G should work for your datasets)
- `--time`: Max runtime (format: days-HH:MM:SS)
- `--output`: Where to save console output
- `--error`: Where to save errors

---

## For Your PHATE Testing

### Step 1: Create Log Directory

```bash
mkdir -p ~/llm-paper-analyze/logs
```

### Step 2: Create SLURM Job Script

I'll create this for you below.

### Step 3: Submit Job

```bash
cd ~/llm-paper-analyze
sbatch run_phate_test.slurm
```

### Step 4: Monitor Job

```bash
# Check if job is running
squeue -u $USER

# Watch job status in real-time
watch -n 5 'squeue -u $USER'

# Check output while running
tail -f logs/phate_test_*.out
```

### Step 5: View Results

```bash
# After job completes, check output
cat logs/phate_test_*.out

# Check for errors
cat logs/phate_test_*.err
```

---

## Common Job States

When you run `squeue -u $USER`, you'll see job states:

- **PD** (Pending): Job is waiting for resources
- **R** (Running): Job is currently running
- **CG** (Completing): Job is finishing up
- **CD** (Completed): Job finished successfully

If job doesn't appear in `squeue`, it's completed. Check with:
```bash
sacct -u $USER --starttime=today
```

---

## Typical Workflow for Your Use Case

### Single Dataset Test:
```bash
# 1. Create job script
nano run_phate_test.slurm

# 2. Submit job
sbatch run_phate_test.slurm

# 3. Get job ID (e.g., 12345)
# Watch progress
tail -f logs/phate_test_12345.out

# 4. After completion, check results
cat logs/phate_test_12345.out
```

### Multiple Datasets (Job Array):
```bash
# Submit 100 jobs at once (1 per dataset)
sbatch --array=1-101 run_phate_batch.slurm

# SLURM will run jobs as resources become available
# Check how many are running
squeue -u $USER | grep phate_batch
```

---

## Memory and Time Estimation

### For Your Datasets (200K-500K cells):

**Quick test (subsample 5K cells):**
- Memory: 8-16GB
- CPUs: 4-8
- Time: 5-15 minutes

**Medium test (subsample 50K cells):**
- Memory: 16-32GB
- CPUs: 8-16
- Time: 30-60 minutes

**Full dataset (500K cells):**
- Memory: 64-128GB
- CPUs: 16-32
- Time: 2-6 hours

**Recommendation**: Start with subsample, then scale up.

---

## Yale-Specific Information

On Yale HPC clusters:

### Available Partitions:
```bash
sinfo
# Common partitions:
# - general: Standard compute nodes
# - bigmem: High-memory nodes (128GB+)
# - gpu: GPU nodes (for deep learning)
# - day: Short jobs (24h limit)
```

### Check Partition Limits:
```bash
scontrol show partition general
# Shows: MaxTime, MaxMemPerNode, MaxCPUsPerNode
```

### Your Allocation:
```bash
# Check your account/allocation
sacctmgr show user $USER -s
```

---

## Troubleshooting

### Job Pending for Long Time?
```bash
# Check why
squeue -j <job_id> --start

# Possible reasons:
# - Requesting too much memory
# - All nodes busy
# - Partition limits
```

### Job Failed?
```bash
# Check error log
cat logs/<job_name>_<job_id>.err

# Check job efficiency
seff <job_id>

# Common issues:
# - Out of memory (request more --mem)
# - Out of time (increase --time)
# - Module/environment issues
```

### Job Killed?
```bash
# If killed for memory, you'll see in error log:
# "slurmstepd: error: Detected 1 oom-kill event"

# Solution: Request more memory
#SBATCH --mem=64G  # Instead of 32G
```

---

## Best Practices

1. **Start small**: Test with 1 dataset before batch processing
2. **Use logs**: Always specify `--output` and `--error`
3. **Be conservative**: Request slightly more memory/time than you think you need
4. **Monitor jobs**: Check logs while running to catch issues early
5. **Clean up**: Delete old log files periodically
6. **Use job arrays**: For processing many datasets efficiently

---

## Example Commands for Your Project

```bash
# Check available resources
sinfo -o "%P %C %m %l"
# Shows: Partition, CPUs (alloc/idle/other/total), Memory, TimeLimit

# Submit test job
sbatch run_phate_test.slurm

# Check queue
squeue -u btd8

# Cancel if needed
scancel <job_id>

# Check completed jobs today
sacct -u btd8 --starttime=today --format=JobID,JobName,Partition,State,Elapsed,MaxRSS

# View efficiency report
seff <job_id>
```

---

## Interactive Sessions (Alternative)

If you want to test interactively with more resources:

```bash
# Request interactive session with 32GB RAM
salloc --partition=general --mem=32G --cpus-per-task=8 --time=2:00:00

# Once allocated, you'll be on a compute node
# Activate environment and run commands
conda activate claude-env
python3 test_phate_simple.py

# Exit when done
exit
```

---

## Next Steps for Your Project

I'll create practical SLURM scripts for you:

1. **Test script**: Test PHATE on 1 dataset
2. **Batch script**: Process all 101 datasets
3. **Monitoring script**: Check progress of batch jobs

See the scripts I'm creating in the next section!
