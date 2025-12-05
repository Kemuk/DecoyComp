#!/usr/bin/env python3
import os
import argparse
from pathlib import Path
from collections import namedtuple, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from abc import ABC, abstractmethod
from typing import Iterator

import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from tqdm.contrib.concurrent import process_map

# Matplotlib setup
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# --- RDKit Descriptors ---
METALS = {3,4,11,12,13,19,20,21,22,23,24,25,26,27,28,29,30,31,37,38,39,40,41,42,43,44,45,46,47,48,49,50,55,56,57,72,73,74,75,76,77,78,79,80,81,82,83}
MolDescriptors = namedtuple("MolDescriptors", ["mw", "clogp", "tpsa", "fsp3", "rb", "lip_pass", "veber_pass", "has_metal", "pains_hit"])

class DescriptorCalculator:
    _PAINS_CATALOG = None
    
    @classmethod
    def get_pains_catalog(cls):
        if cls._PAINS_CATALOG is None:
            print("[INFO] Initializing PAINS catalog...")
            from rdkit.Chem import FilterCatalog
            params = FilterCatalog.FilterCatalogParams()
            for catalog in (FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS_A,
                            FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS_B,
                            FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS_C):
                params.AddCatalog(catalog)
            cls._PAINS_CATALOG = FilterCatalog.FilterCatalog(params)
            print("[INFO] PAINS catalog initialized")
        return cls._PAINS_CATALOG
    
    @classmethod
    def calculate(cls, mol) -> MolDescriptors:
        from rdkit.Chem import Descriptors, Crippen, rdMolDescriptors
        mw = Descriptors.MolWt(mol)
        clogp = Crippen.MolLogP(mol)
        tpsa = rdMolDescriptors.CalcTPSA(mol)
        fsp3 = rdMolDescriptors.CalcFractionCSP3(mol)
        rb = int(Descriptors.NumRotatableBonds(mol))
        lip_pass = (mw <= 500 and clogp <= 5 and 
                    Descriptors.NumHDonors(mol) <= 5 and 
                    Descriptors.NumHAcceptors(mol) <= 10)
        veber_pass = (rb <= 10 and tpsa <= 140.0)
        has_metal = any(atom.GetAtomicNum() in METALS for atom in mol.GetAtoms())
        pains_hit = cls.get_pains_catalog().HasMatch(mol)
        return MolDescriptors(mw, clogp, tpsa, fsp3, rb, lip_pass, veber_pass, has_metal, pains_hit)
    
    @staticmethod
    def calculate_one_smiles(smi: str):
        """Returns (smiles, has_salts, MolDescriptors or None)"""
        from rdkit import Chem
        salts = "." in smi
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return smi, salts, None
        desc = DescriptorCalculator.calculate(mol)
        return smi, salts, desc
    
    @staticmethod
    def calculate_all_parallel(smiles_list: list[str], workers: int) -> dict[str, tuple[bool, MolDescriptors]]:
        """Calculate descriptors for all SMILES and return cache dict: {smiles: (has_salts, desc)}"""
        print(f"[INFO] Calculating descriptors for {len(smiles_list)} unique SMILES with {workers} workers...")
        
        results = process_map(
            DescriptorCalculator.calculate_one_smiles,
            smiles_list,
            max_workers=workers,
            chunksize=100,
            desc="Computing descriptors"
        )
        
        cache = {}
        invalid_count = 0
        for smi, salts, desc in results:
            if desc is None:
                invalid_count += 1
                cache[smi] = (salts, None)
            else:
                cache[smi] = (salts, desc)
        
        valid_count = len(smiles_list) - invalid_count
        print(f"[INFO] Computed {valid_count} valid molecules ({invalid_count} invalid)")
        return cache

# --- Aggregation from cache ---
def aggregate_from_cache(smiles_list: list[str], cache: dict) -> dict:
    """Aggregate statistics from cached descriptors"""
    results = {
        "NumberLigands": 0,
        "NumberInvalidSMILES": 0,
        "NumberWithSalts": 0,
        "NumberWithMetals": 0,
        "NumberPAINSMatches": 0,
        "NumberLipinskiCompliant": 0,
        "NumberVeberCompliant": 0,
        "_Sum_MW": 0.0,
        "_Sum_cLogP": 0.0,
        "_Sum_TPSA": 0.0,
        "_Sum_Fsp3": 0.0,
        "_Sum_RotB": 0.0,
        "_Count_Desc": 0
    }
    
    for smi in smiles_list:
        if smi not in cache:
            continue
        
        salts, desc = cache[smi]
        
        if desc is None:
            results["NumberInvalidSMILES"] += 1
            continue
        
        results["NumberLigands"] += 1
        if salts:
            results["NumberWithSalts"] += 1
        if desc.has_metal:
            results["NumberWithMetals"] += 1
        if desc.pains_hit:
            results["NumberPAINSMatches"] += 1
        if desc.lip_pass:
            results["NumberLipinskiCompliant"] += 1
        if desc.veber_pass:
            results["NumberVeberCompliant"] += 1
        
        results["_Sum_MW"] += desc.mw
        results["_Sum_cLogP"] += desc.clogp
        results["_Sum_TPSA"] += desc.tpsa
        results["_Sum_Fsp3"] += desc.fsp3
        results["_Sum_RotB"] += desc.rb
        results["_Count_Desc"] += 1
    
    return results

# --- Dataset Classes ---
def read_smi_file(filepath: Path, label: str):
    if not filepath.is_file():
        return
    with open(filepath, "r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            parts = line.strip().split()
            if parts:
                yield parts[0], label

class BaseDataset(ABC):
    def __init__(self, name: str, root_path: Path):
        self.name = name
        self.root_path = root_path
    
    @abstractmethod
    def enumerate_targets(self) -> Iterator[tuple[str, Path]]:
        pass
    
    @abstractmethod
    def read_target(self, target_path: Path) -> Iterator[tuple[str, str]]:
        pass

class LitPCBADataset(BaseDataset):
    def enumerate_targets(self):
        if not self.root_path.is_dir():
            return
        for entry in sorted(self.root_path.iterdir()):
            if entry.is_dir():
                yield entry.name, entry
    
    def read_target(self, target_path: Path):
        yield from read_smi_file(target_path / "actives.smi", "active")
        yield from read_smi_file(target_path / "inactives.smi", "inactive")

class DudeZDataset(BaseDataset):
    def enumerate_targets(self):
        if not self.root_path.is_dir():
            return
        for entry in sorted(self.root_path.iterdir()):
            if entry.is_dir():
                yield entry.name, entry
    
    def read_target(self, target_path: Path):
        yield from read_smi_file(target_path / "actives_final.ism", "active")
        yield from read_smi_file(target_path / "decoys_final.ism", "decoy")

class Dekois2Dataset(BaseDataset):
    def enumerate_targets(self):
        if not self.root_path.is_dir():
            return
        for entry in sorted(self.root_path.iterdir()):
            if entry.is_dir():
                yield entry.name, entry
    
    def read_target(self, target_path: Path):
        smi_path = target_path / "active_decoys.smi"
        if not smi_path.is_file():
            return
        with open(smi_path, "r", encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                parts = line.strip().split()
                if len(parts) >= 2:
                    smi, lig_id = parts[0], parts[1]
                    if lig_id.startswith("BDB"):
                        yield smi, "active"
                    elif lig_id.startswith("ZINC"):
                        yield smi, "decoy"

# --- Target Stats (using cache) ---
class TargetStats:
    def __init__(self, cache: dict):
        self.cache = cache
        self.counts = {k: 0 for k in ["actives", "decoys", "invalid", "salts", "metal", "pains", "lipinski", "veber"]}
        self.sums = {k: 0.0 for k in ["mw", "clogp", "tpsa", "fsp3", "rb"]}
        self.rbs = {"actives": [], "decoys": []}

    def update(self, smi: str, label: str):
        if smi not in self.cache:
            self.counts["invalid"] += 1
            return
        
        salts, desc = self.cache[smi]
        
        if salts:
            self.counts["salts"] += 1
        
        if desc is None:
            self.counts["invalid"] += 1
            return
        
        if label == "active":
            self.counts["actives"] += 1
            self.rbs["actives"].append(desc.rb)
        else:
            self.counts["decoys"] += 1
            self.rbs["decoys"].append(desc.rb)
        
        if desc.has_metal:
            self.counts["metal"] += 1
        if desc.pains_hit:
            self.counts["pains"] += 1
        if desc.lip_pass:
            self.counts["lipinski"] += 1
        if desc.veber_pass:
            self.counts["veber"] += 1
        
        for key in self.sums:
            self.sums[key] += getattr(desc, key)

    def report(self, dataset: str, target: str) -> dict:
        total = self.counts["actives"] + self.counts["decoys"]
        def safe_div(n,d): return (n/d) if d>0 else 0.0
        return {
            "Dataset": dataset, "Target": target,
            "NumberActives": self.counts["actives"],
            "NumberDecoys/Inactives": self.counts["decoys"],
            "NumberLigandsTotal": total,
            "NumberInvalidSMILES": self.counts["invalid"],
            "NumberWithSalts": self.counts["salts"],
            "NumberWithMetals": self.counts["metal"],
            "NumberPAINSMatches": self.counts["pains"],
            "NumberLipinskiCompliant": self.counts["lipinski"],
            "LipinskiComplianceRate": safe_div(self.counts["lipinski"], total),
            "NumberVeberCompliant": self.counts["veber"],
            "VeberComplianceRate": safe_div(self.counts["veber"], total),
            "ActivesFraction": safe_div(self.counts["actives"], total),
            "Mean_MW": safe_div(self.sums["mw"], total),
            "Mean_cLogP": safe_div(self.sums["clogp"], total),
            "Mean_TPSA": safe_div(self.sums["tpsa"], total),
            "Mean_Fsp3": safe_div(self.sums["fsp3"], total),
            "Mean_RotatableBonds": safe_div(self.sums["rb"], total),
            "_Sum_MW": self.sums["mw"], "_Sum_cLogP": self.sums["clogp"],
            "_Sum_TPSA": self.sums["tpsa"], "_Sum_Fsp3": self.sums["fsp3"],
            "_Sum_RotB": self.sums["rb"], "_Count_Desc": total,
            "_RBs_Actives": self.rbs["actives"],
            "_RBs_DecoysOrInactives": self.rbs["decoys"],
        }

# --- Target Processor ---
class TargetProcessor:
    def __init__(self, cache: dict):
        self.cache = cache
    
    def _process_one_target(self, dataset_obj: BaseDataset, target_name: str, target_path: Path) -> dict:
        stats = TargetStats(self.cache)
        for smi, label in dataset_obj.read_target(target_path):
            stats.update(smi, label)
        return stats.report(dataset_obj.name, target_name)
    
    def process_all(self, datasets: list[BaseDataset]) -> pd.DataFrame:
        print("\n[STEP 2] Processing individual targets (from cache)...")
        tasks = []
        for dataset_obj in datasets:
            for target_name, target_path in dataset_obj.enumerate_targets():
                tasks.append((dataset_obj, target_name, target_path))
        
        print(f"[INFO] Found {len(tasks)} targets across {len(datasets)} datasets")
        
        rows = []
        for dataset_obj, target_name, target_path in tqdm(tasks, desc="Processing targets"):
            res = self._process_one_target(dataset_obj, target_name, target_path)
            if res:
                rows.append(res)
        
        print(f"[INFO] Completed processing {len(rows)} targets")
        return pd.DataFrame(rows).sort_values(["Dataset", "Target"]).reset_index(drop=True)

# --- Unique Dataset Analyzer ---
class UniqueDatasetAnalyzer:
    def __init__(self, datasets: list[BaseDataset], cache: dict):
        self.datasets = datasets
        self.cache = cache
        self._smiles_cache = None
    
    def collect_smiles(self) -> dict[str, dict[str, set[str]]]:
        if self._smiles_cache is not None:
            print("[INFO] Using cached SMILES data")
            return self._smiles_cache
        
        print("\n[STEP 3] Collecting unique SMILES from all datasets...")
        dataset_to_smiles = defaultdict(lambda: {"active": set(), "inactive": set()})
        
        for dataset_obj in tqdm(self.datasets, desc="Scanning datasets"):
            target_count = 0
            for target_name, target_path in dataset_obj.enumerate_targets():
                target_count += 1
                for smi, label in dataset_obj.read_target(target_path):
                    if label == "active":
                        dataset_to_smiles[dataset_obj.name]["active"].add(smi)
                    else:
                        dataset_to_smiles[dataset_obj.name]["inactive"].add(smi)
            
            actives_count = len(dataset_to_smiles[dataset_obj.name]["active"])
            inactives_count = len(dataset_to_smiles[dataset_obj.name]["inactive"])
            print(f"  {dataset_obj.name}: {target_count} targets, {actives_count} unique actives, {inactives_count} unique inactives")
        
        self._smiles_cache = dict(dataset_to_smiles)
        return self._smiles_cache
    
    def create_summary(self) -> pd.DataFrame:
        print("\n[STEP 4] Creating dataset-level summary (from cache)...")
        ds_smiles = self.collect_smiles()
        rows = []
        
        for dataset in tqdm(list(ds_smiles.keys()), desc="Processing datasets"):
            buckets = ds_smiles[dataset]
            smiles = list(buckets["active"] | buckets["inactive"])
            
            agg = aggregate_from_cache(smiles, self.cache)
            total = agg["NumberLigands"]
            denom = total if total > 0 else np.nan
            means_denom = agg["_Count_Desc"] if agg["_Count_Desc"] > 0 else np.nan
            
            row = {
                "Dataset": dataset,
                "NumberLigandsUnique": int(total),
                "NumberActivesUnique": len(buckets["active"]),
                "NumberInactivesUnique": len(buckets["inactive"]),
                "NumberInvalidSMILES": int(agg["NumberInvalidSMILES"]),
                "NumberWithSalts": int(agg["NumberWithSalts"]),
                "NumberWithMetals": int(agg["NumberWithMetals"]),
                "NumberPAINSMatches": int(agg["NumberPAINSMatches"]),
                "NumberLipinskiCompliant": int(agg["NumberLipinskiCompliant"]),
                "LipinskiComplianceRate": (agg["NumberLipinskiCompliant"] / denom) if denom == denom else 0.0,
                "NumberVeberCompliant": int(agg["NumberVeberCompliant"]),
                "VeberComplianceRate": (agg["NumberVeberCompliant"] / denom) if denom == denom else 0.0,
                "Mean_MW": (agg["_Sum_MW"] / means_denom) if means_denom == means_denom else 0.0,
                "Mean_cLogP": (agg["_Sum_cLogP"] / means_denom) if means_denom == means_denom else 0.0,
                "Mean_TPSA": (agg["_Sum_TPSA"] / means_denom) if means_denom == means_denom else 0.0,
                "Mean_Fsp3": (agg["_Sum_Fsp3"] / means_denom) if means_denom == means_denom else 0.0,
                "Mean_RotatableBonds": (agg["_Sum_RotB"] / means_denom) if means_denom == means_denom else 0.0,
                "ActivesFraction": len(buckets["active"]) / denom if denom == denom else 0.0
            }
            rows.append(row)
        
        return pd.DataFrame(rows).sort_values("Dataset")
    
    def _summarize_bucket(self, smiles: list[str], bucket_label: str, 
                          dataset: str, actives: set[str], inactives: set[str]) -> dict:
        agg = aggregate_from_cache(smiles, self.cache)
        total = agg["NumberLigands"]
        denom = total if total > 0 else np.nan
        means_denom = agg["_Count_Desc"] if agg["_Count_Desc"] > 0 else np.nan
        actives_fraction = len(actives) / (len(actives) + len(inactives)) if (len(actives) + len(inactives)) > 0 else 0.0
        
        return {
            "Dataset": dataset, "Bucket": bucket_label,
            "NumberLigands": int(total),
            "NumberInvalidSMILES": int(agg["NumberInvalidSMILES"]),
            "NumberWithSalts": int(agg["NumberWithSalts"]),
            "NumberWithMetals": int(agg["NumberWithMetals"]),
            "NumberPAINSMatches": int(agg["NumberPAINSMatches"]),
            "NumberLipinskiCompliant": int(agg["NumberLipinskiCompliant"]),
            "LipinskiComplianceRate": (agg["NumberLipinskiCompliant"] / denom) if denom == denom else 0.0,
            "NumberVeberCompliant": int(agg["NumberVeberCompliant"]),
            "VeberComplianceRate": (agg["NumberVeberCompliant"] / denom) if denom == denom else 0.0,
            "Mean_MW": (agg["_Sum_MW"] / means_denom) if means_denom == means_denom else 0.0,
            "Mean_cLogP": (agg["_Sum_cLogP"] / means_denom) if means_denom == means_denom else 0.0,
            "Mean_TPSA": (agg["_Sum_TPSA"] / means_denom) if means_denom == means_denom else 0.0,
            "Mean_Fsp3": (agg["_Sum_Fsp3"] / means_denom) if means_denom == means_denom else 0.0,
            "Mean_RotatableBonds": (agg["_Sum_RotB"] / means_denom) if means_denom == means_denom else 0.0,
            "ActivesFraction": actives_fraction
        }
    
    def create_summary_split(self) -> pd.DataFrame:
        print("\n[STEP 5] Creating split summary (actives/inactives/all - from cache)...")
        ds_smiles = self.collect_smiles()
        rows = []
        
        for dataset in tqdm(list(ds_smiles.keys()), desc="Processing datasets"):
            buckets = ds_smiles[dataset]
            actives = list(buckets["active"])
            inactives = list(buckets["inactive"])
            
            print(f"\n  Processing {dataset}:")
            print(f"    Actives: {len(actives)} molecules")
            rows.append(self._summarize_bucket(actives, "Actives", dataset, buckets["active"], buckets["inactive"]))
            
            print(f"    Inactives: {len(inactives)} molecules")
            rows.append(self._summarize_bucket(inactives, "Inactives", dataset, buckets["active"], buckets["inactive"]))
            
            print(f"    All: {len(set(actives) | set(inactives))} unique molecules")
            all_ligs = list(set(actives) | set(inactives))
            rows.append(self._summarize_bucket(all_ligs, "All", dataset, buckets["active"], buckets["inactive"]))
        
        return pd.DataFrame(rows).sort_values(["Dataset", "Bucket"])
    
    def write_smiles_files(self, outdir: Path):
        print("\n[STEP] Writing unique SMILES files...")
        outdir.mkdir(exist_ok=True, parents=True)
        ds_smiles = self.collect_smiles()
        
        for dataset, buckets in ds_smiles.items():
            actives_file = outdir / f"{dataset}_actives.smi"
            inactives_file = outdir / f"{dataset}_inactives.smi"
            print(f"  Writing {dataset}...")
            
            with open(actives_file, "w") as fa:
                for smi in sorted(buckets["active"]):
                    fa.write(f"{smi}\n")
            
            with open(inactives_file, "w") as fi:
                for smi in sorted(buckets["inactive"]):
                    fi.write(f"{smi}\n")
            
            print(f"    - {actives_file.name}: {len(buckets['active'])} actives")
            print(f"    - {inactives_file.name}: {len(buckets['inactive'])} inactives")
        
        print(f"[OK] Wrote SMILES files to {outdir}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--roots", nargs="*", default=["LIT-PCBA", "DEKOIS2", "DUDE-Z"])
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    parser.add_argument("--outdir", type=Path, default=Path("."))
    parser.add_argument("--write-smiles-only", action="store_true")
    args = parser.parse_args()
    args.outdir.mkdir(exist_ok=True)

    print("="*70)
    print("DATASET ANALYSIS PIPELINE (OPTIMIZED)")
    print("="*70)
    print(f"Output directory: {args.outdir}")
    print(f"Workers: {args.workers}")
    print(f"Datasets to process: {', '.join(args.roots)}")
    print("="*70)

    # Initialize datasets
    print("\n[INIT] Initializing datasets...")
    datasets = []
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

    if not datasets:
        print("[ERROR] No valid datasets found!")
        return

    # STEP 1: Collect all unique SMILES
    print("\n[STEP 1] Collecting all unique SMILES across datasets...")
    all_smiles = set()
    for dataset_obj in tqdm(datasets, desc="Scanning datasets"):
        for target_name, target_path in dataset_obj.enumerate_targets():
            for smi, label in dataset_obj.read_target(target_path):
                all_smiles.add(smi)
    
    print(f"[INFO] Found {len(all_smiles)} unique SMILES total")
    
    # STEP 2: Calculate all descriptors once
    descriptor_cache = DescriptorCalculator.calculate_all_parallel(list(all_smiles), args.workers)
    print(f"[INFO] Descriptor cache built with {len(descriptor_cache)} entries")
    
    # Initialize analyzer with cache
    analyzer = UniqueDatasetAnalyzer(datasets, descriptor_cache)
    
    if args.write_smiles_only:
        analyzer.write_smiles_files(args.outdir)
        return

    # Process all targets using cache
    processor = TargetProcessor(descriptor_cache)
    df = processor.process_all(datasets)
    
    print("\n[OUTPUT] Saving per-target summary...")
    output_file = args.outdir/"dataset_summary.csv"
    df.drop(columns=[c for c in df.columns if c.startswith("_")]).to_csv(output_file, index=False)
    print(f"  ✓ Saved: {output_file} ({len(df)} targets)")
    
    print("\n[OUTPUT] Saving LIT-PCBA per-target summary...")
    lit_pcba_df = df[df["Dataset"]=="LIT-PCBA"].copy()
    if not lit_pcba_df.empty:
        lit_pcba_df.rename(columns={"NumberDecoys/Inactives":"NumberInactives"}, inplace=True)
        output_file = args.outdir/"per_target_summary.csv"
        lit_pcba_df.drop(columns=[c for c in lit_pcba_df.columns if c.startswith("_")]).to_csv(output_file, index=False)
        print(f"  ✓ Saved: {output_file} ({len(lit_pcba_df)} targets)")
    else:
        print("  ! No LIT-PCBA data found")
    
    print("\n[OUTPUT] Creating dataset-level unique summary...")
    dataset_unique_df = analyzer.create_summary()
    output_file = args.outdir/"dataset_unique_summary.csv"
    dataset_unique_df.to_csv(output_file, index=False)
    print(f"  ✓ Saved: {output_file}")
    
    print("\n[OUTPUT] Creating split summary (actives/inactives)...")
    dataset_unique_split_df = analyzer.create_summary_split()
    output_file = args.outdir/"dataset_unique_summary_split.csv"
    dataset_unique_split_df.to_csv(output_file, index=False)
    print(f"  ✓ Saved: {output_file}")
    
    print("\n" + "="*70)
    print("[OK] Analysis complete!")
    print("="*70)

if __name__ == "__main__":
    main()