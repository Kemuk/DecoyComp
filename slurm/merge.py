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


def find_chunk_files(indir: Path) -> List[Path]:
    """Find all chunk_*.parquet files."""
    chunk_files = sorted(
        indir.glob("chunk_*.parquet"),
        key=lambda p: int(p.stem.split("_")[1]) if "_" in p.stem else 0
    )
    return chunk_files


def merge_chunk_parquets(chunk_files: List[Path]) -> pl.DataFrame:
    """Merge chunk parquet files."""
    dfs = []
    for filepath in chunk_files:
        if filepath.exists():
            print(f"  + Found: {filepath}")
            dfs.append(pl.read_parquet(filepath))
        else:
            print(f"  - Missing: {filepath}")

    if not dfs:
        raise FileNotFoundError(f"No chunk_*.parquet files found")

    print(f"[MERGE] Concatenating {len(dfs)} chunk files...")
    merged = pl.concat(dfs)
    print(f"[MERGE] Total rows: {len(merged)}")
    return merged


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

    # Find chunk files
    chunk_files = find_chunk_files(indir)
    if not chunk_files:
        print(f"[ERROR] No chunk_*.parquet files found in {indir}")
        return

    print(f"[INFO] Found {len(chunk_files)} chunk files to merge")
    for f in chunk_files:
        print(f"  + {f.name}")

    outdir.mkdir(parents=True, exist_ok=True)

    # Merge all chunk parquets
    try:
        merged_df = merge_chunk_parquets(chunk_files)
        
        outfile = outdir / "merged_results.parquet"
        merged_df.write_parquet(outfile)
        print(f"\n[OK] Merged results saved to: {outfile} ({len(merged_df)} rows)")
    except Exception as e:
        print(f"[ERROR] Failed to merge chunks: {e}")
        return


if __name__ == "__main__":
    main()
