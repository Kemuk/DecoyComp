#!/usr/bin/env python3
"""
Dataset analysis pipeline for benchmark datasets.
(LIT-PCBA, DUDE-Z, DEKOIS2, D-COID, MUV)

Uses Polars and Joblib for fast processing.
Outputs are written as Parquet files.
"""
import argparse
from pathlib import Path

import polars as pl

from molecular_utils import DescriptorCalculator
from datasets import LitPCBADataset, DudeZDataset, Dekois2Dataset, MUVDataset, DCOIDDataset
from analyser import DatasetAnalyser


def main():
    parser = argparse.ArgumentParser(description="Analyse benchmark datasets")
    parser.add_argument(
        "--roots", nargs="*", default=["LIT-PCBA", "DEKOIS2", "DUDE-Z"],
        help="Dataset names/paths to analyse. File-based datasets need a path; "
             "pass 'MUV' by name to load via DeepChem. Example: --roots LIT-PCBA DUDE-Z MUV"
    )
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
    parser.add_argument("--write-smiles-only", action="store_true",
                        help="Only write SMILES files, skip analysis")
    parser.add_argument("--no-cache", action="store_true",
                        help="Disable joblib caching for descriptors")
    args = parser.parse_args()

    args.outdir.mkdir(exist_ok=True)

    print("=" * 70)
    print("DATASET ANALYSIS PIPELINE")
    print("=" * 70)
    print(f"Output directory: {args.outdir}")
    print(f"Workers: {args.workers}")
    print(f"Datasets to process: {', '.join(args.roots)}")
    print("=" * 70)

    # Initialise datasets
    print("\n[INIT] Initialising datasets...")
    datasets = []

    for root_str in args.roots:
        root = Path(root_str)
        name = root.name
        if name == "MUV":
            try:
                muv_dataset = MUVDataset(data_dir=args.muv_data_dir)
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

    if not datasets:
        print("[ERROR] No valid datasets found!")
        return

    # Collect capped unique SMILES via analyser so the same subset drives all outputs.
    print("\n[STEP 1] Collecting unique SMILES across datasets...")
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

    if args.write_smiles_only:
        analyser.write_smiles_files(args.smiles_dir)
        return

    # Process all targets
    print("\n[STEP 2] Processing targets...")
    df = analyser.process_targets()

    print("\n[OUTPUT] Saving per-target summary...")
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


if __name__ == "__main__":
    main()
