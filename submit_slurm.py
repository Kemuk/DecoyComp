#!/usr/bin/env python3
"""
Submit SLURM array jobs for dataset analysis.

Usage:
    python3 submit_slurm.py --mode devel
    python3 submit_slurm.py --mode prod
    python3 submit_slurm.py --merge
"""
import argparse
import subprocess
import sys
from pathlib import Path

import yaml


def load_config():
    """Load configuration from slurm/config.yaml."""
    config_path = Path(__file__).parent / "slurm" / "config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def submit_worker(mode, config):
    """Submit worker array job."""
    mode_config = config[mode]

    # Build sbatch command
    cmd = [
        "sbatch",
        f"--partition={mode_config['partition']}",
        f"--time={mode_config['time']}",
        f"--array=1-{mode_config['chunks']}",
        f"--cpus-per-task={mode_config['cpus']}",
        f"--mem-per-cpu={mode_config['mem_per_cpu']}G",
    ]

    # Build environment variables
    env_vars = {
        "OUTDIR": mode_config['results_dir'],
        "WORKERS": "-1",
    }

    if mode_config.get('max_ligands'):
        env_vars["MAX_LIGANDS"] = str(mode_config['max_ligands'])

    # Add env vars to command
    env_str = ",".join([f"{k}={v}" for k, v in env_vars.items()])
    cmd.append(f"--export={env_str}")

    cmd.append("slurm/worker.slurm")

    print(f"[SUBMIT] Worker array job ({mode})...")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"[ERROR] Failed to submit worker: {result.stderr}")
        sys.exit(1)

    # Extract job ID from output (e.g., "Submitted batch job 12345")
    job_id = result.stdout.strip().split()[-1]
    print(f"[SUCCESS] Worker job submitted: {job_id}")
    return job_id


def submit_merge(job_id, mode, config):
    """Submit merge job with dependency on worker job."""
    mode_config = config[mode]
    merge_config = mode_config['merge']

    # Build sbatch command
    cmd = [
        "sbatch",
        f"--partition={mode_config['partition']}",
        f"--time={merge_config['time']}",
        f"--cpus-per-task={merge_config['cpus']}",
        f"--mem-per-cpu={merge_config['mem_per_cpu']}G",
        f"--dependency=afterok:{job_id}",
    ]

    # Build environment variables
    env_vars = {
        "INDIR": merge_config['indir'],
        "OUTDIR": merge_config['outdir'],
    }

    # Add env vars to command
    env_str = ",".join([f"{k}={v}" for k, v in env_vars.items()])
    cmd.append(f"--export={env_str}")

    cmd.append("slurm/merge.slurm")

    print(f"[SUBMIT] Merge job (depends on {job_id})...")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"[ERROR] Failed to submit merge: {result.stderr}")
        sys.exit(1)

    # Extract job ID
    merge_job_id = result.stdout.strip().split()[-1]
    print(f"[SUCCESS] Merge job submitted: {merge_job_id}")
    return merge_job_id


def run_merge_standalone(config):
    """Run merge.py directly (already computed, just merge)."""
    # Use prod merge config by default for standalone merge
    merge_config = config['prod']['merge']
    indir = merge_config.get('indir', 'results')
    outdir = merge_config.get('outdir', 'merged_results')

    cmd = ["python3", "slurm/merge.py", "--indir", indir, "--outdir", outdir]

    print(f"[MERGE] Running merge directly...")
    result = subprocess.run(cmd)
    sys.exit(result.returncode)


def main():
    parser = argparse.ArgumentParser(description="Submit SLURM array jobs for dataset analysis")
    parser.add_argument("--mode", choices=["devel", "prod"], help="Submission mode")
    parser.add_argument("--merge", action="store_true", help="Run merge.py directly")
    args = parser.parse_args()

    # Create logs directory before submitting jobs (SBATCH output paths evaluated at submission time)
    (Path(__file__).parent / "logs").mkdir(exist_ok=True)

    config = load_config()

    if args.merge:
        # Just run merge.py
        run_merge_standalone(config)
    elif args.mode:
        # Submit both worker and merge jobs
        job_id = submit_worker(args.mode, config)
        merge_job_id = submit_merge(job_id, args.mode, config)

        print(f"\n{'='*60}")
        print(f"Submitted {args.mode} analysis with automatic merge")
        print(f"{'='*60}")
        print(f"Worker job:  {job_id}")
        print(f"Merge job:   {merge_job_id} (depends on {job_id})")
        print(f"\nMonitor with:")
        print(f"  squeue -j {job_id}")
        print(f"  watch -n 5 'squeue -j {job_id}'")
        print(f"\nView logs:")
        print(f"  tail -f logs/chunk_*.out")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
