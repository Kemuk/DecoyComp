# SLURM Array Job Submission

Submit dataset analysis jobs as SLURM array jobs using the **Map-Reduce pattern**:
- **Worker** (`worker.slurm`) - processes independent chunks in parallel
- **Merger** (`merge.slurm`) - combines per-chunk results automatically

## Quick Start

```bash
# Test on devel partition (10 min, 5 small chunks)
python3 submit_slurm.py --mode devel

# Production on short partition (240 min, 4 chunks)
python3 submit_slurm.py --mode prod

# Just merge existing results
python3 submit_slurm.py --merge
```

## Configuration

Edit `slurm/config.yaml` to customize:
- `partition`: SLURM partition (devel, short, gpu, etc.)
- `time`: Time limit in minutes
- `chunks`: Number of parallel array tasks
- `cpus`: CPUs per task
- `mem_per_cpu`: Memory per CPU (GB)
- `max_ligands`: Limit molecules per dataset (null = no limit)
- `results_dir`: Output directory for chunks
- `merge`: Merge job configuration (indir, outdir, resources)

**Devel mode** uses small resources and limits datasets (`max_ligands: 1000`) for quick testing.

**Prod mode** uses full datasets and larger merge resources.

## How It Works

1. `python3 submit_slurm.py --mode devel` submits:
   - `worker.slurm` array job (5 chunks with limited data)
   - `merge.slurm` job (depends on worker completion)

2. Worker job runs:
   - Each chunk processes independent dataset subset
   - Saves to `results_devel/chunk_1/`, `chunk_2/`, etc.

3. Merge job runs automatically when worker completes:
   - Combines chunks: `chunk_1/`, `chunk_2/`, etc.
   - Produces: `merged_results_devel/`

## Output Structure

```
results_devel/                  (worker results)
├── chunk_1/
│   ├── dataset_summary.parquet
│   ├── dataset_unique_summary.parquet
│   └── ...
├── chunk_2/
└── ...

merged_results_devel/           (merger output)
├── dataset_summary.parquet
├── dataset_unique_summary.parquet
└── ...

logs/
├── chunk_1.out
├── chunk_2.out
└── ...
```

## Monitoring

```bash
# Check job status
squeue -j <JOB_ID>
watch -n 5 'squeue -j <JOB_ID>'

# View worker output
tail -f logs/chunk_1.out
tail -f logs/chunk_*.out

# Cancel job
scancel <JOB_ID>
```

## Performance

**Chunks:**
- 1-2: Quick test
- 4-8: Standard production
- 8+: Very large datasets

**CPU & Memory:**
- Default: 4 CPUs, 4GB per CPU per chunk
- Merge job: 8 CPUs, 8GB (prod), smaller (devel)

## Troubleshooting

**Job stuck in queue:**
```bash
squeue -j <JOB_ID> --format=long
```
Check `REASON` column for clues.

**Out of memory:**
Edit `config.yaml` to increase `mem_per_cpu` or reduce `chunks`.

**Start fresh:**
```bash
rm -rf results_devel results merged_results_devel merged_results logs/chunk_*
```

**Edit default values:**
Edit `slurm/config.yaml` directly.
