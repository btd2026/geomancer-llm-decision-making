#!/bin/bash
# Run Claude Code in an interactive Slurm session with adequate memory
# This avoids the OOM killer on the login node

# Request an interactive session on the devel partition
# --mem=32G gives enough headroom for Claude Code (~27GB virtual memory)
# --time=4:00:00 gives you 4 hours (adjust as needed)
# --pty allocates a pseudo-terminal for interactive use

salloc --partition=devel --mem=32G --cpus-per-task=4 --time=4:00:00 --pty bash -c '
    echo "=== Interactive session allocated ==="
    echo "Node: $(hostname)"
    echo "Memory limit: $(cat /sys/fs/cgroup/memory/memory.limit_in_bytes 2>/dev/null | numfmt --to=iec 2>/dev/null || echo "N/A")"
    echo ""
    echo "Starting Claude Code..."
    claude
'
