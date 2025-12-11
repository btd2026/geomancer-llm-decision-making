#!/bin/bash
# Monitor the W&B confusion matrix SLURM job

JOB_ID=3456218

echo "Monitoring SLURM job $JOB_ID for W&B confusion matrix analysis"
echo "=========================================="

# Function to check job status
check_job_status() {
    squeue -j $JOB_ID --noheader --format="%i %t %M %l %N %r" 2>/dev/null || echo "Job $JOB_ID not found in queue (may have completed)"
}

# Function to show logs if available
show_logs() {
    if [ -f "logs/wandb_confusion_matrix_${JOB_ID}.log" ]; then
        echo ""
        echo "=== Current Output Log ==="
        tail -20 "logs/wandb_confusion_matrix_${JOB_ID}.log"
    fi

    if [ -f "logs/wandb_confusion_matrix_${JOB_ID}.err" ]; then
        echo ""
        echo "=== Current Error Log ==="
        tail -20 "logs/wandb_confusion_matrix_${JOB_ID}.err"
    fi
}

# Check initial status
echo "Job Status: $(check_job_status)"
echo ""

# Monitor loop
while true; do
    status=$(check_job_status)
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Job Status: $status"

    # Show logs if available
    show_logs

    # Check if job is no longer in queue
    if ! squeue -j $JOB_ID &>/dev/null; then
        echo ""
        echo "Job $JOB_ID has completed or is no longer in queue."
        echo "Checking final logs and output..."
        show_logs

        # Check if output files were created
        echo ""
        echo "=== Output Files ==="
        if [ -d "data/wandb_confusion_analysis" ]; then
            ls -la data/wandb_confusion_analysis/
        else
            echo "No output directory found"
        fi

        break
    fi

    sleep 30  # Check every 30 seconds
done

echo ""
echo "Monitoring complete."