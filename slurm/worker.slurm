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
# Submitted by submit_slurm.py with environment variables

set -euo pipefail
export PYTHONUNBUFFERED=1

# Go to repo root (submission directory)
REPO_ROOT="${REPO_ROOT:-$(pwd)}"
cd "${REPO_ROOT}"

# Create logs directory
mkdir -p logs

# Validate required environment variables
if [[ -z "${OUTDIR:-}" ]]; then
    echo "ERROR: OUTDIR environment variable not set by submit_slurm.py"
    exit 1
fi

# Configuration (from environment or defaults)
CHUNK_ID=${SLURM_ARRAY_TASK_ID}
TOTAL_CHUNKS=${SLURM_ARRAY_TASK_COUNT}
WORKERS="${WORKERS:--1}"
DATASETS=("LIT-PCBA" "DEKOIS2" "DUDE-Z")
MAX_LIGANDS="${MAX_LIGANDS:-}"

# Resolve paths
OUTDIR="${REPO_ROOT}/${OUTDIR}"
CONFIG_FILE="${REPO_ROOT}/slurm/config.yaml"

# Validate config exists
if [[ ! -f "${CONFIG_FILE}" ]]; then
    echo "ERROR: Config file not found: ${CONFIG_FILE}"
    echo "Run setup first to create config"
    exit 1
fi

# Validate analyze script exists
ANALYZE_SCRIPT="${REPO_ROOT}/analyse_datasets.py"
if [[ ! -f "${ANALYZE_SCRIPT}" ]]; then
    echo "ERROR: Analysis script not found: ${ANALYZE_SCRIPT}"
    exit 1
fi

echo "=========================================="
echo "SLURM Array Job Worker"
echo "=========================================="
echo "Chunk ID:     ${CHUNK_ID} / ${TOTAL_CHUNKS}"
echo "Task:         ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "Node:         $(hostname)"
echo "CPUs:         ${SLURM_CPUS_PER_TASK}"
echo "Memory Avail: ${SLURM_MEM_PER_NODE} MB"
echo "Repo Root:    ${REPO_ROOT}"
echo "Output Dir:   ${OUTDIR}"
echo "Config File:  ${CONFIG_FILE}"
echo "Start Time:   $(date)"
echo "=========================================="

# Create output directory
mkdir -p "${OUTDIR}"

# Activate environment
if [[ -d "${REPO_ROOT}/.venv" ]]; then
    source "${REPO_ROOT}/.venv/bin/activate"
    echo "Activated venv"
fi

# Run analysis
ANALYSIS_ARGS=(
    "--roots" "${DATASETS[@]}"
    "--outdir" "${OUTDIR}/chunk_${CHUNK_ID}"
    "--workers" "$WORKERS"
)

if [[ -n "${MAX_LIGANDS}" ]]; then
    ANALYSIS_ARGS+=("--max-ligands-per-dataset" "${MAX_LIGANDS}")
fi

echo ""
echo "Running: python3 ${ANALYZE_SCRIPT} ${ANALYSIS_ARGS[*]}"
echo ""

python3 "${ANALYZE_SCRIPT}" "${ANALYSIS_ARGS[@]}"

EXIT_CODE=$?
if [[ ${EXIT_CODE} -ne 0 ]]; then
    echo "ERROR: Analysis failed with exit code ${EXIT_CODE}"
    exit ${EXIT_CODE}
fi

echo ""
echo "=========================================="
echo "Chunk $CHUNK_ID completed successfully at $(date)"
echo "=========================================="
