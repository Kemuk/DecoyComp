#!/usr/bin/env python3
"""
Dataset analysis pipeline for benchmark datasets
(LIT-PCBA, DUDE-Z, DEKOIS2, MUV)
"""
import os
import argparse
from pathlib import Path

from tqdm.auto import tqdm

from molecular_utils import DescriptorCalculator
from datasets import LitPCBADataset, DudeZDataset, Dekois2Dataset, MUVDataset
from analyser import DatasetAnalyser


def main():
    parser = argparse.ArgumentParser(description="Analyse benchmark datasets")
    parser.add_argument("--roots", nargs="*", default=["LIT-PCBA", "DEKOIS2", "DUDE-Z"],
                        help="Root directories for file-based datasets")
    parser.add_argument("--workers", type=int, default=os.cpu_count(),
                        help="Number of parallel workers")
    parser.add_argument("--outdir", type=Path, default=Path("results"),
                        help="Output directory")
    parser.add_argument("--include-muv", action="store_true", default=True,
                        help="Include MUV dataset from DeepChem (default: True)")
    parser.add_argument("--exclude-muv", dest="include_muv", action="store_false",
                        help="Exclude MUV dataset")
    parser.add_argument("--muv-data-dir", type=Path, default=None,
                        help="DeepChem data directory for MUV (default: ~/.deepchem/datasets)")
    parser.add_argument("--write-smiles-only", action="store_true",
                        help="Only write SMILES files, skip analysis")
    args = parser.parse_args()
    
    args.outdir.mkdir(exist_ok=True)
    
    dataset_names = list(args.roots)
    if args.include_muv:
        dataset_names.append("MUV")
    
    print("="*70)
    print("DATASET ANALYSIS PIPELINE")
    print("="*70)
    print(f"Output directory: {args.outdir}")
    print(f"Workers: {args.workers}")
    print(f"Datasets to process: {', '.join(dataset_names)}")
    print("="*70)
    
    # Initialise datasets
    print("\n[INIT] Initialising datasets...")
    datasets = []
    
    # File-based datasets
    for root_str in args.roots:
        root = Path(root_str)
        name = root.name
        if name in ("LIT-PCBA", "LIT_PCBA"):
            datasets.append(LitPCBADataset("LIT-PCBA", root))
            print(f"  ✓ LIT-PCBA: {root}")
        elif name == "DUDE-Z":
            datasets.append(DudeZDataset("DUDE-Z", root))
            print(f"  ✓ DUDE-Z: {root}")
        elif name == "DEKOIS2":
            datasets.append(Dekois2Dataset("DEKOIS2", root))
            print(f"  ✓ DEKOIS2: {root}")
    
    # MUV dataset (API-based)
    if args.include_muv:
        try:
            muv_dataset = MUVDataset(data_dir=args.muv_data_dir)
            muv_dataset.load()
            datasets.append(muv_dataset)
            print(f"  ✓ MUV: DeepChem (17 targets)")
        except ImportError:
            print("  ✗ MUV: DeepChem not installed (skipping)")
            print("    Install with: pip install deepchem --break-system-packages")
        except Exception as e:
            print(f"  ✗ MUV: Failed to load ({e})")
    
    if not datasets:
        print("[ERROR] No valid datasets found!")
        return
    
    # Collect all unique SMILES
    print("\n[STEP 1] Collecting all unique SMILES across datasets...")
    all_smiles = set()
    for dataset_obj in tqdm(datasets, desc="Scanning datasets"):
        for target_name, target_path in dataset_obj.enumerate_targets():
            for smi, label in dataset_obj.read_target(target_path):
                all_smiles.add(smi)
    
    print(f"[INFO] Found {len(all_smiles)} unique SMILES total")
    
    # Calculate all descriptors once
    descriptor_cache = DescriptorCalculator.calculate_all_parallel(list(all_smiles), args.workers)
    print(f"[INFO] Descriptor cache built with {len(descriptor_cache)} entries")
    
    # Initialise analyser
    analyser = DatasetAnalyser(datasets, descriptor_cache)
    
    if args.write_smiles_only:
        analyser.write_smiles_files(args.outdir)
        return
    
    # Process all targets
    df = analyser.process_targets()
    
    print("\n[OUTPUT] Saving per-target summary...")
    output_file = args.outdir / "dataset_summary.csv"
    df.drop(columns=[c for c in df.columns if c.startswith("_")]).to_csv(output_file, index=False)
    print(f"  ✓ Saved: {output_file} ({len(df)} targets)")
    
    print("\n[OUTPUT] Saving LIT-PCBA per-target summary...")
    lit_pcba_df = df[df["Dataset"] == "LIT-PCBA"].copy()
    if not lit_pcba_df.empty:
        lit_pcba_df.rename(columns={"NumberDecoys/Inactives": "NumberInactives"}, inplace=True)
        output_file = args.outdir / "per_target_summary.csv"
        lit_pcba_df.drop(columns=[c for c in lit_pcba_df.columns if c.startswith("_")]).to_csv(output_file, index=False)
        print(f"  ✓ Saved: {output_file} ({len(lit_pcba_df)} targets)")
    else:
        print("  ! No LIT-PCBA data found")
    
    print("\n[OUTPUT] Creating dataset-level unique summary...")
    dataset_unique_df = analyser.create_dataset_summary()
    output_file = args.outdir / "dataset_unique_summary.csv"
    dataset_unique_df.to_csv(output_file, index=False)
    print(f"  ✓ Saved: {output_file}")
    
    print("\n[OUTPUT] Creating split summary (actives/inactives)...")
    dataset_unique_split_df = analyser.create_split_summary()
    output_file = args.outdir / "dataset_unique_summary_split.csv"
    dataset_unique_split_df.to_csv(output_file, index=False)
    print(f"  ✓ Saved: {output_file}")
    
    print("\n" + "="*70)
    print("[OK] Analysis complete!")
    print("="*70)


if __name__ == "__main__":
    main()