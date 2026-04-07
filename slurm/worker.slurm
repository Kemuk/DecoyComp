#!/bin/bash
#SBATCH --job-name=decoy-analysis
#SBATCH --array=1-4                    # Number of parallel chunks
#SBATCH --partition=short              # Partition: short, devel, etc.
#SBATCH --time=30                      # Time in minutes
#SBATCH --cpus-per-task=4              # CPUs per chunk
#SBATCH --mem-per-cpu=4G               # Memory per CPU
#SBATCH --output=logs/chunk_%a.out
#SBATCH --error=logs/chunk_%a.err

# SLURM array job worker - processes one chunk
# Submit from repo root: sbatch slurm/worker.slurm

set -e

# Go to repo root
cd "$(dirname "$0")/.."

# Configuration (from environment or defaults)
CHUNK_ID=$SLURM_ARRAY_TASK_ID
TOTAL_CHUNKS=$SLURM_ARRAY_TASK_COUNT
OUTDIR="${OUTDIR:-results}"
WORKERS="${WORKERS:--1}"
DATASETS=("LIT-PCBA" "DEKOIS2" "DUDE-Z")
MAX_LIGANDS="${MAX_LIGANDS:-}"

mkdir -p logs "$OUTDIR"

echo "=========================================="
echo "SLURM Array Job Worker"
echo "=========================================="
echo "Chunk: $CHUNK_ID / $TOTAL_CHUNKS"
echo "Node: $(hostname)"
echo "CPUs Available: $SLURM_CPUS_PER_TASK"
echo "Allocated Memory: $SLURM_MEM_PER_NODE MB"
echo "Start Time: $(date)"
echo "=========================================="

# Activate environment
if [ -d .venv ]; then
    source .venv/bin/activate
fi

# Run analysis
ANALYSIS_ARGS=(
    "--roots" "${DATASETS[@]}"
    "--outdir" "${OUTDIR}/chunk_${CHUNK_ID}"
    "--workers" "$WORKERS"
)

if [ -n "$MAX_LIGANDS" ]; then
    ANALYSIS_ARGS+=("--max-ligands-per-dataset" "$MAX_LIGANDS")
fi

python3 analyse_datasets.py "${ANALYSIS_ARGS[@]}"

echo "=========================================="
echo "Chunk $CHUNK_ID completed at $(date)"
echo "=========================================="
