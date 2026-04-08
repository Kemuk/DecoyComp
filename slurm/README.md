# SLURM Array Job Submission

Submit dataset analysis jobs as SLURM array jobs using the **Map-Reduce pattern** with manifest-based chunk assignment:
- **Manifest Build** - pre-builds canonical parquet with all ligand data once
- **Worker** (`worker.slurm`) - each chunk fetches manifest, filters deterministically, processes results
- **Merger** (`merge.slurm`) - combines per-chunk parquets automatically

## Quick Start

```bash
# Test on devel partition (10 min, 5 chunks, 1000 ligands/dataset)
python3 submit_slurm.py --mode devel

# Production on short partition (240 min, 200 chunks, full data)
python3 submit_slurm.py --mode prod

# Just merge existing chunk results
python3 submit_slurm.py --merge
```

## Configuration

Edit `slurm/config.yaml` to customize:
- `partition`: SLURM partition (devel, short, gpu, etc.)
- `time`: Time limit in minutes
- `chunks`: Number of parallel array tasks
- `cpus`: CPUs per task
- `mem_per_cpu`: Memory per CPU (GB)
- `max_ligands_per_dataset`: Limit molecules per dataset (null = no limit)
- `datasets`: List of dataset names to include
- `results_dir`: Output directory for chunks
- `merge`: Merge job configuration (indir, outdir, resources)

**Devel mode** uses small resources and limits datasets (`max_ligands_per_dataset: 1000`) for quick testing.

**Prod mode** uses full datasets (200 chunks) and larger merge resources.

## How It Works

1. `python3 submit_slurm.py --mode devel` executes:
   - Pre-builds canonical manifest: `{results_dir}/canonical_manifest.parquet`
   - Submits `worker.slurm` array job (5 chunks, manifest passed to each)
   - Submits `merge.slurm` job (depends on worker completion)

2. Worker job (each chunk) runs:
   - Load manifest from shared path (SLURM passes via env var)
   - Filter manifest to assigned rows: `(manifest_id - 1) % total_chunks == (chunk_id - 1)`
   - Process targets and calculate descriptors for this chunk only
   - Save to `{results_dir}/chunk_XXXX.parquet`

3. Merge job runs automatically when worker completes:
   - Find all `chunk_*.parquet` files
   - Concatenate into single output: `{merge_outdir}/merged_results.parquet`

## Output Structure

```
results/                        (worker results)
├── canonical_manifest.parquet  (pre-built, shared)
├── chunk_0001.parquet
├── chunk_0002.parquet
├── ...
└── chunk_0200.parquet          (for prod mode with 200 chunks)

merged_results/                 (merger output)
└── merged_results.parquet      (concatenated chunk results)

logs/
├── chunk_1.out
├── chunk_2.out
└── ...
```

## Manifest-Based Workflow

**Key Benefit:** Manifest-based chunk assignment eliminates duplicate work:

- **Old approach:** Each chunk ran full pipeline identically → 4N duplicate work
- **New approach:** One manifest → each chunk processes deterministic subset → N total work

**Manifest Contents:**
- All ligands from all datasets (subject to `max_ligands_per_dataset` cap)
- Compound keys enforce uniqueness (dataset|protein_id|ligand_id)
- Manifest ID (row number) used for chunk assignment

**Chunk Assignment Example (5 chunks, 100 ligands):**
- Chunk 1: manifest_ids 1, 6, 11, 16, 21, 26, 31, 36, 41, 46, 51, ...
- Chunk 2: manifest_ids 2, 7, 12, 17, 22, 27, 32, 37, 42, 47, 52, ...
- ... (deterministic, reproducible)
- Chunk 5: manifest_ids 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, ...

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
- 1-2: Quick test runs
- 5: Standard devel testing
- 200: Production (default)
- 8+: Very large datasets

**CPU & Memory:**
- Worker: 4 CPUs, 16GB total (4GB × 4) per chunk
- Merge: 8 CPUs, 32GB total (4GB × 8)

## Troubleshooting

**Job stuck in queue:**
```bash
squeue -j <JOB_ID> --format=long
```
Check `REASON` column for clues.

**Out of memory:**
Edit `config.yaml` to increase `mem_per_cpu` or reduce `chunks`.

**Manifest build failed:**
Run manifest build standalone:
```bash
python3 analyse_datasets.py --build-manifest-only --roots LIT-PCBA DUDE-Z DEKOIS2 D-COID --output-manifest results/canonical_manifest.parquet
```

**Start fresh:**
```bash
rm -rf results results_devel merged_results merged_results_devel logs/chunk_*
```

**Edit default values:**
Edit `slurm/config.yaml` directly.
