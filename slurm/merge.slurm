#!/bin/bash
#SBATCH --job-name=decoy-merge
#SBATCH --partition=short              # Partition: short, devel, etc.
#SBATCH --time=00:30:00                      # Time in minutes
#SBATCH --cpus-per-task=8              # CPUs per chunk
#SBATCH --mem-per-cpu=8G     
#SBATCH --output=logs/merge_%j.out
#SBATCH --error=logs/merge_%j.err

# SLURM merge job - combines per-chunk results
# Submitted by submit_slurm.py with dependency on worker array job
# Environment vars set by submit_slurm.py

set -euo pipefail
export PYTHONUNBUFFERED=1

# Set PROJECT_DIR from submit_slurm.py export, SLURM_SUBMIT_DIR, or pwd
if [[ -z "${PROJECT_DIR:-}" ]]; then
    PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
fi

CONPREFIX="$(readlink -f "${PROJECT_DIR}/../envs/aev-plig")"

source activate $CONPREFIX

cd "${PROJECT_DIR}"

# Create logs directory
mkdir -p "${PROJECT_DIR}/logs"

# Validate required environment variables
if [[ -z "${INDIR:-}" ]]; then
    echo "ERROR: INDIR environment variable not set by submit_slurm.py"
    exit 1
fi

if [[ -z "${OUTDIR:-}" ]]; then
    echo "ERROR: OUTDIR environment variable not set by submit_slurm.py"
    exit 1
fi

# Configuration (from environment)
INDIR="${PROJECT_DIR}/${INDIR}"
OUTDIR="${PROJECT_DIR}/${OUTDIR}"
MERGE_SCRIPT="${PROJECT_DIR}/slurm/merge.py"

# Validate merge script exists
if [[ ! -f "${MERGE_SCRIPT}" ]]; then
    echo "ERROR: Merge script not found: ${MERGE_SCRIPT}"
    exit 1
fi

# Validate input directory has chunk results
if [[ ! -d "${INDIR}" ]]; then
    echo "ERROR: Input directory not found: ${INDIR}"
    echo "Worker job may not have completed successfully"
    exit 1
fi

CHUNK_COUNT=$(find "${INDIR}" -maxdepth 1 -type d -name "chunk_*" | wc -l)
if [[ ${CHUNK_COUNT} -eq 0 ]]; then
    echo "ERROR: No chunk results found in ${INDIR}"
    echo "Worker job may not have completed successfully"
    exit 1
fi

echo "=========================================="
echo "SLURM Merge Job"
echo "=========================================="
echo "Task:         ${SLURM_ARRAY_JOB_ID}"
echo "Node:         $(hostname)"
echo "Project Dir:  ${PROJECT_DIR}"
echo "Input Dir:    ${INDIR}"
echo "Output Dir:   ${OUTDIR}"
echo "Chunks Found: ${CHUNK_COUNT}"
echo "Merge Script: ${MERGE_SCRIPT}"
echo "Start Time:   $(date)"
echo "=========================================="

# Activate environment
if [[ -d "${PROJECT_DIR}/.venv" ]]; then
    source "${PROJECT_DIR}/.venv/bin/activate"
    echo "Activated venv"
fi

# Run merge
echo ""
echo "Running: python3 ${MERGE_SCRIPT} --indir ${INDIR} --outdir ${OUTDIR}"
echo ""

python3 "${MERGE_SCRIPT}" --indir "${INDIR}" --outdir "${OUTDIR}"

EXIT_CODE=$?
if [[ ${EXIT_CODE} -ne 0 ]]; then
    echo "ERROR: Merge failed with exit code ${EXIT_CODE}"
    exit ${EXIT_CODE}
fi

echo ""
echo "=========================================="
echo "Merge completed successfully at $(date)"
echo "Results saved to: ${OUTDIR}"
echo "=========================================="
