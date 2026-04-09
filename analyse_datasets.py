#!/usr/bin/env python3
"""
Dataset analysis pipeline for benchmark datasets.
(LIT-PCBA, DUDE-Z, DEKOIS2, D-COID, MUV)

Uses Polars and Joblib for fast processing.
Outputs are written as Parquet files.

Manifest Mode: Builds a canonical manifest parquet with all ligands, then processes
chunks via deterministic filtering based on manifest_id % total_chunks.
"""
import argparse
import logging
import subprocess
from pathlib import Path

import polars as pl
from rdkit import RDLogger

from molecular_utils import DescriptorCalculator
from datasets import LitPCBADataset, DudeZDataset, Dekois2Dataset, MUVDataset, DCOIDDataset
from datasets import dataset_to_manifest_frame
from analyser import Analyser

# Suppress RDKit warnings
RDLogger.DisableLog('rdApp.*')

logger = logging.getLogger(__name__)


def _build_full_manifest(
    datasets: list,
    max_ligands_per_dataset: int | None,
    workers: int
) -> pl.DataFrame:
    """
    Build canonical manifest from all datasets.
    
    Args:
        datasets: List of BaseDataset objects
        max_ligands_per_dataset: Cap per dataset (or None)
        workers: Number of parallel workers
    
    Returns:
        Full manifest DataFrame with manifest_id column added
    
    Raises:
        RuntimeError: if no datasets or manifest is empty
    """
    if not datasets:
        raise RuntimeError("No datasets provided")
    
    print("[MANIFEST] Building canonical manifest from datasets...")
    
    # Convert each dataset to normalized manifest frame
    frames = []
    for dataset_obj in datasets:
        print(f"  - Converting {dataset_obj.name}...")
        try:
            df = dataset_to_manifest_frame(dataset_obj, max_ligands_per_dataset)
            frames.append(df)
            print(f"    + {len(df)} rows")
        except Exception as e:
            raise RuntimeError(f"Failed to convert {dataset_obj.name}: {e}")
    
    if not frames:
        raise RuntimeError("No frames generated from datasets")
    
    # Concatenate all frames
    full_manifest = pl.concat(frames)
    
    if full_manifest.is_empty():
        raise RuntimeError("Manifest is empty after concatenation")
    
    # Add compound_key (dataset|protein_id|ligand_id)
    full_manifest = full_manifest.with_columns([
        (pl.col('dataset') + '|' + pl.col('protein_id') + '|' + pl.col('ligand_id'))
        .alias('compound_key')
    ])
    
    # Enforce compound_key uniqueness
    unique_keys = full_manifest.select('compound_key').unique()
    if len(unique_keys) < len(full_manifest):
        dups = len(full_manifest) - len(unique_keys)
        duplicate_key_counts = (
            full_manifest
            .group_by('compound_key')
            .agg(pl.len().alias('count'))
            .filter(pl.col('count') > 1)
            .sort('count', descending=True)
            .head(10)
        )
        print("[ERROR] Top duplicate compound keys:")
        for row in duplicate_key_counts.iter_rows(named=True):
            print(f"  - {row['compound_key']}: {row['count']} rows")
        raise RuntimeError(f"Compound key not unique: {dups} duplicates found")
    
    # Add manifest_id (1-indexed row number)
    full_manifest = full_manifest.with_row_count('manifest_id', offset=1)
    
    # Reorder columns to match schema
    full_manifest = full_manifest.select([
        'manifest_id', 'dataset', 'target_id', 'protein_id', 'label', 'smiles',
        'ligand_id', 'compound_key', 'file_path', 'ligand_file_path',
        'protein_file_path', 'cache_location', 'source_split'
    ])
    
    print(f"[MANIFEST] Full manifest: {len(full_manifest)} rows, {full_manifest.width} columns")
    
    return full_manifest


def _load_and_filter_manifest(
    manifest_path: Path,
    chunk_id: int,
    total_chunks: int
) -> pl.DataFrame:
    """
    Load manifest and filter to chunk's assigned rows.
    
    Filtering rule: (manifest_id - 1) % total_chunks == (chunk_id - 1)
    
    Args:
        manifest_path: Path to manifest parquet
        chunk_id: This chunk's ID (1-indexed)
        total_chunks: Total number of chunks
    
    Returns:
        Filtered manifest for this chunk
    
    Raises:
        FileNotFoundError: if manifest not found
        ValueError: if chunk_id or total_chunks invalid
    """
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    
    if chunk_id < 1 or chunk_id > total_chunks:
        raise ValueError(f"Invalid chunk_id {chunk_id}: must be 1 <= id <= {total_chunks}")
    
    if total_chunks < 1:
        raise ValueError(f"Invalid total_chunks {total_chunks}: must be >= 1")
    
    print(f"[MANIFEST] Loading manifest from {manifest_path}...")
    full_manifest = pl.read_parquet(manifest_path)
    
    # Filter to this chunk's rows
    chunk_manifest = full_manifest.filter(
        (pl.col('manifest_id') - 1) % total_chunks == (chunk_id - 1)
    )
    
    print(f"[MANIFEST] Chunk {chunk_id}/{total_chunks}: {len(chunk_manifest)} rows assigned")
    
    return chunk_manifest


def main():
    parser = argparse.ArgumentParser(description="Analyse benchmark datasets")
    
    # Legacy mode args (for backwards compatibility during transition)
    parser.add_argument(
        "--roots", nargs="*", default=None,
        help="[LEGACY] Dataset names/paths to analyse. Use --manifest mode for new pipeline."
    )
    
    # Manifest mode args
    parser.add_argument("--manifest", type=Path, default=None,
                        help="Path to canonical manifest parquet (manifest mode)")
    parser.add_argument("--chunk-id", type=int, default=None,
                        help="Chunk ID (1-indexed) for this worker (manifest mode)")
    parser.add_argument("--total-chunks", type=int, default=None,
                        help="Total number of chunks (manifest mode)")
    parser.add_argument("--build-manifest-only", action="store_true",
                        help="Build manifest and exit (manifest mode)")
    parser.add_argument("--output-manifest", type=Path, default=Path("canonical_manifest.parquet"),
                        help="Path to write canonical manifest (default: canonical_manifest.parquet)")
    
    # Common args
    parser.add_argument("--workers", type=int, default=-1,
                        help="Number of parallel workers (-1 = all CPUs)")
    parser.add_argument("--outdir", type=Path, default=Path("results"),
                        help="Output directory")
    parser.add_argument("--smiles-dir", type=Path, default=Path("smiles"),
                        help="Directory for SMILES files (default: smiles)")
    parser.add_argument("--muv-data-dir", type=Path, default=None,
                        help="DeepChem data directory for MUV (default: ~/.deepchem/datasets)")
    parser.add_argument("--max-ligands-per-dataset", type=int, default=None,
                        help="Limit unique ligands collected per dataset")
    parser.add_argument("--no-cache", action="store_true",
                        help="Disable joblib caching for descriptors")
    args = parser.parse_args()
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # ========== MANIFEST MODE: Build or Process ==========
    if args.build_manifest_only or args.manifest:
        print("=" * 70)
        print("MANIFEST MODE")
        print("=" * 70)
        
        if args.build_manifest_only:
            # Build manifest once
            if args.roots is None:
                args.roots = ["LIT-PCBA", "DEKOIS2", "DUDE-Z"]
            
            print(f"\n[INIT] Initialising datasets: {', '.join(args.roots)}")
            datasets = _init_datasets(args.roots, args.muv_data_dir)
            
            if not datasets:
                print("[ERROR] No valid datasets found!")
                return
            
            try:
                full_manifest = _build_full_manifest(
                    datasets,
                    args.max_ligands_per_dataset,
                    args.workers
                )
                full_manifest.write_parquet(args.output_manifest)
                print(f"[OK] Manifest written to {args.output_manifest}")
            except Exception as e:
                print(f"[ERROR] Failed to build manifest: {e}")
                raise
            return
        
        # Process from manifest as a chunk worker
        if args.manifest is None:
            raise ValueError("--manifest required for chunk processing")
        if args.chunk_id is None or args.total_chunks is None:
            raise ValueError("--chunk-id and --total-chunks required for chunk processing")
        
        chunk_manifest = _load_and_filter_manifest(args.manifest, args.chunk_id, args.total_chunks)
        
        if chunk_manifest.is_empty():
            print(f"[WARNING] Chunk {args.chunk_id} has no rows. Exiting.")
            return
        
        # Create analyser in manifest mode
        print(f"\n[ANALYSER] Creating manifest-mode analyser...")
        analyser = Analyser(manifest_df=chunk_manifest)
        
        # Process targets for this chunk
        print(f"\n[STEP 1] Collecting unique SMILES from chunk...")
        ds_smiles_frame = analyser.collect_smiles()
        all_smiles = set(ds_smiles_frame.get_column('smiles').to_list())
        print(f"[INFO] Found {len(all_smiles):,} unique SMILES")
        
        # Calculate descriptors
        use_cache = not args.no_cache
        descriptor_cache = DescriptorCalculator.calculate_all_parallel(
            all_smiles,
            workers=args.workers,
            use_cache=use_cache
        )
        print(f"[INFO] Descriptor cache built with {len(descriptor_cache):,} entries")
        
        # Process targets and save output
        print(f"\n[STEP 2] Processing chunk targets...")
        analyser.cache = descriptor_cache
        df = analyser.process_targets()
        
        # Write chunk output
        chunk_output = args.outdir / f"chunk_{args.chunk_id:04d}.parquet"
        args.outdir.mkdir(exist_ok=True, parents=True)
        df.write_parquet(chunk_output)
        print(f"[OK] Chunk {args.chunk_id} output: {chunk_output} ({len(df)} targets)")
        return
    
    # ========== LEGACY MODE (fallback for backwards compatibility) ==========
    print("=" * 70)
    print("LEGACY MODE (Dataset Reader)")
    print("=" * 70)
    
    if args.roots is None:
        args.roots = ["LIT-PCBA", "DEKOIS2", "DUDE-Z"]
    
    print(f"Output directory: {args.outdir}")
    print(f"Workers: {args.workers}")
    print(f"Datasets to process: {', '.join(args.roots)}")
    print("=" * 70)

    # Initialise datasets
    print("\n[INIT] Initialising datasets...")
    datasets = _init_datasets(args.roots, args.muv_data_dir)

    if not datasets:
        print("[ERROR] No valid datasets found!")
        return

    # Collect capped unique SMILES via analyser so the same subset drives all outputs.
    print("\n[STEP 1] Collecting unique SMILES across datasets...")
    from analyser import DatasetAnalyser
    analyser = DatasetAnalyser(datasets, {}, max_ligands_per_dataset=args.max_ligands_per_dataset)
    ds_smiles = analyser.collect_smiles()
    all_smiles = set()
    for buckets in ds_smiles.values():
        all_smiles.update(buckets["active"])
        all_smiles.update(buckets["inactive"])

    print(f"[INFO] Found {len(all_smiles):,} unique SMILES total")

    # Calculate all descriptors once (with optional caching)
    use_cache = not args.no_cache
    descriptor_cache = DescriptorCalculator.calculate_all_parallel(
        all_smiles,
        workers=args.workers,
        use_cache=use_cache
    )
    print(f"[INFO] Descriptor cache built with {len(descriptor_cache):,} entries")

    # Reuse the same analyser and cached capped SMILES set for downstream processing.
    analyser.cache = descriptor_cache

    # Process all targets
    print("\n[STEP 2] Processing targets...")
    df = analyser.process_targets()

    print("\n[OUTPUT] Saving per-target summary...")
    args.outdir.mkdir(exist_ok=True)
    output_file = args.outdir / "dataset_summary.parquet"
    internal_cols = tuple(filter(lambda c: c.startswith("_"), df.columns))
    df.drop(internal_cols).write_parquet(output_file)
    print(f"  + Saved: {output_file} ({len(df)} targets)")

    # LIT-PCBA per-target summary (subset of target summary)
    lit_pcba_df = df.filter(pl.col("Dataset") == "LIT-PCBA")
    if not lit_pcba_df.is_empty():
        print("\n[OUTPUT] Saving LIT-PCBA per-target summary...")
        output_file = args.outdir / "per_target_summary.parquet"
        internal_cols = tuple(filter(lambda c: c.startswith("_"), lit_pcba_df.columns))
        lit_pcba_df.drop(internal_cols).write_parquet(output_file)
        print(f"  + Saved: {output_file} ({len(lit_pcba_df)} targets)")

    print("\n[OUTPUT] Creating dataset-level unique summary...")
    dataset_unique_df = analyser.create_dataset_summary()
    output_file = args.outdir / "dataset_unique_summary.parquet"
    dataset_unique_df.write_parquet(output_file)
    print(f"  + Saved: {output_file}")

    print("\n[OUTPUT] Creating split summary (actives/inactives)...")
    dataset_unique_split_df = analyser.create_split_summary()
    output_file = args.outdir / "dataset_unique_summary_split.parquet"
    dataset_unique_split_df.write_parquet(output_file)
    print(f"  + Saved: {output_file}")

    print("\n[OUTPUT] Writing SMILES files...")
    analyser.write_smiles_files(args.smiles_dir)

    print("\n" + "=" * 70)
    print("[OK] Analysis complete!")
    print("=" * 70)


def _init_datasets(roots: list, muv_data_dir: Path | None) -> list:
    """Initialize datasets from root list."""
    datasets = []

    for root_str in roots:
        root = Path(root_str)
        name = root.name
        if name == "MUV":
            try:
                muv_dataset = MUVDataset(data_dir=muv_data_dir)
                muv_dataset.load()
                datasets.append(muv_dataset)
                print("  + MUV: DeepChem (17 targets)")
            except ImportError:
                print("  - MUV: DeepChem not installed (skipping)")
                print("    Install with: pip install 'decoycomp[muv]'")
            except Exception as e:
                print(f"  - MUV: Failed to load ({e})")
        elif name in ("LIT-PCBA", "LIT_PCBA"):
            datasets.append(LitPCBADataset("LIT-PCBA", root))
            print(f"  + LIT-PCBA: {root}")
        elif name == "DUDE-Z":
            datasets.append(DudeZDataset("DUDE-Z", root))
            print(f"  + DUDE-Z: {root}")
        elif name == "DEKOIS2":
            datasets.append(Dekois2Dataset("DEKOIS2", root))
            print(f"  + DEKOIS2: {root}")
        elif name == "D-COID":
            datasets.append(DCOIDDataset("D-COID", root))
            print(f"  + D-COID: {root}")
        else:
            print(f"  ? Unknown dataset: {root_str} (skipping)")

    return datasets


if __name__ == "__main__":
    main()
