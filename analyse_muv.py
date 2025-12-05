#!/usr/bin/env python3
"""
MUV dataset analysis pipeline using DeepChem
"""
import os
import argparse
from pathlib import Path

from molecular_utils import DescriptorCalculator
from datasets import MUVDataset
from analyser import DatasetAnalyser
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 


def main():
    parser = argparse.ArgumentParser(description="Analyse MUV dataset from DeepChem")
    parser.add_argument("--workers", type=int, default=os.cpu_count(),
                        help="Number of parallel workers")
    parser.add_argument("--outdir", type=Path, default=Path("."),
                        help="Output directory")
    parser.add_argument("--data-dir", type=Path, default=None,
                        help="DeepChem data directory (default: ~/.deepchem/datasets)")
    parser.add_argument("--write-smiles", action="store_true",
                        help="Write SMILES files for each target")
    args = parser.parse_args()
    
    args.outdir.mkdir(exist_ok=True)
    
    print("="*70)
    print("MUV DATASET ANALYSIS PIPELINE")
    print("="*70)
    print(f"Output directory: {args.outdir}")
    print(f"Workers: {args.workers}")
    print("="*70)
    
    # Load MUV dataset
    muv_dataset = MUVDataset(data_dir=args.data_dir)
    muv_dataset.load()
    
    # Collect all unique SMILES
    print("\n[STEP 1] Collecting all unique SMILES...")
    all_smiles = set()
    for target_name, target_path in muv_dataset.enumerate_targets():
        for smi, label in muv_dataset.read_target(target_path):
            all_smiles.add(smi)
    
    print(f"[INFO] Found {len(all_smiles)} unique SMILES total")
    
    # Calculate all descriptors once
    descriptor_cache = DescriptorCalculator.calculate_all_parallel(list(all_smiles), args.workers)
    print(f"[INFO] Descriptor cache built with {len(descriptor_cache)} entries")
    
    # Initialise analyser
    analyser = DatasetAnalyser([muv_dataset], descriptor_cache)
    
    # Write SMILES files if requested
    if args.write_smiles:
        smiles_dir = args.outdir / "smiles_out"
        analyser.write_smiles_files(smiles_dir)
    
    # Per-target summary
    print("\n[OUTPUT] Saving per-target summary...")
    target_summary = analyser.process_targets()
    # Rename column for MUV (uses inactives not decoys)
    target_summary.rename(columns={"NumberDecoys/Inactives": "NumberInactives"}, inplace=True)
    output_file = args.outdir / "muv_target_summary.csv"
    target_summary.drop(columns=[c for c in target_summary.columns if c.startswith("_")]).to_csv(output_file, index=False)
    print(f"  ? Saved: {output_file} ({len(target_summary)} targets)")
    
    # Dataset-level summary
    print("\n[OUTPUT] Saving dataset-level summary...")
    dataset_summary = analyser.create_dataset_summary()
    output_file = args.outdir / "muv_dataset_summary.csv"
    dataset_summary.to_csv(output_file, index=False)
    print(f"  ? Saved: {output_file}")
    
    # Split summary
    print("\n[OUTPUT] Saving split summary...")
    split_summary = analyser.create_split_summary()
    output_file = args.outdir / "muv_dataset_summary_split.csv"
    split_summary.to_csv(output_file, index=False)
    print(f"  ? Saved: {output_file}")
    
    print("\n" + "="*70)
    print("[OK] MUV analysis complete!")
    print("="*70)


if __name__ == "__main__":
    main()