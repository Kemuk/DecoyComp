#!/usr/bin/env python3
"""
Merge results from SLURM array job chunks.

Usage:
    python3 slurm/merge.py --indir results --outdir merged_results
"""
import argparse
from pathlib import Path
from typing import List
import polars as pl


def find_chunk_dirs(indir: Path) -> List[Path]:
    """Find all chunk_N directories."""
    chunk_dirs = sorted(
        indir.glob("chunk_*"),
        key=lambda p: int(p.name.split("_")[1]) if "_" in p.name else 0
    )
    return [d for d in chunk_dirs if d.is_dir()]


def merge_parquet_files(chunk_dirs: List[Path], filename: str) -> pl.DataFrame:
    """Merge a parquet file across all chunks."""
    dfs = []
    for chunk_dir in chunk_dirs:
        filepath = chunk_dir / filename
        if filepath.exists():
            print(f"  + Found: {filepath}")
            dfs.append(pl.read_parquet(filepath))
        else:
            print(f"  - Missing: {filepath}")

    if not dfs:
        raise FileNotFoundError(f"No {filename} files found in any chunks")

    return pl.concat(dfs)


def main():
    parser = argparse.ArgumentParser(
        description="Merge per-chunk results from SLURM array jobs"
    )
    parser.add_argument(
        "--indir",
        type=Path,
        default=Path("results"),
        help="Input directory containing chunk_N subdirectories"
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("merged_results"),
        help="Output directory for merged results"
    )
    args = parser.parse_args()

    indir = args.indir.resolve()
    outdir = args.outdir.resolve()

    if not indir.exists():
        print(f"[ERROR] Input directory not found: {indir}")
        return

    # Find chunks
    chunk_dirs = find_chunk_dirs(indir)
    if not chunk_dirs:
        print(f"[ERROR] No chunk_* directories found in {indir}")
        return

    print(f"[INFO] Found {len(chunk_dirs)} chunks to merge")
    for d in chunk_dirs:
        print(f"  + {d.name}")

    outdir.mkdir(parents=True, exist_ok=True)

    # Merge each file type
    files_to_merge = [
        "dataset_summary.parquet",
        "per_target_summary.parquet",
        "dataset_unique_summary.parquet",
        "dataset_unique_summary_split.parquet",
    ]

    for filename in files_to_merge:
        try:
            print(f"\n[MERGE] Merging {filename}...")
            merged_df = merge_parquet_files(chunk_dirs, filename)

            outfile = outdir / filename
            merged_df.write_parquet(outfile)
            print(f"  ✓ Saved: {outfile} ({len(merged_df)} rows)")
        except FileNotFoundError:
            print(f"  [SKIP] {filename} not found in any chunks")
        except Exception as e:
            print(f"  [ERROR] Failed to merge {filename}: {e}")

    print(f"\n[OK] Merged results saved to: {outdir}")


if __name__ == "__main__":
    main()
