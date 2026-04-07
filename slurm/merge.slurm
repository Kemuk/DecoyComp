#!/bin/bash
#SBATCH --job-name=decoy-merge
#SBATCH --output=logs/merge_%j.out
#SBATCH --error=logs/merge_%j.err

# SLURM merge job - combines per-chunk results
# Submitted by submit_slurm.py with dependency on worker array job
# Environment vars set by submit_slurm.py

set -e

# Go to repo root
cd "$(dirname "$0")/.."

# Create logs directory
mkdir -p logs

# Configuration (from environment)
INDIR="${INDIR:-results}"
OUTDIR="${OUTDIR:-merged_results}"

echo "=========================================="
echo "SLURM Merge Job"
echo "=========================================="
echo "Input:  $INDIR"
echo "Output: $OUTDIR"
echo "Start Time: $(date)"
echo "=========================================="

# Activate environment
if [ -d .venv ]; then
    source .venv/bin/activate
fi

# Run merge
python3 slurm/merge.py --indir "$INDIR" --outdir "$OUTDIR"

echo "=========================================="
echo "Merge completed at $(date)"
echo "=========================================="
