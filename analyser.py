#!/usr/bin/env python3
"""
Analysis engine for dataset statistics and reporting
"""
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from molecular_utils import aggregate_from_cache


class TargetStats:
    """Accumulates statistics for a single target"""
    
    def __init__(self, cache: dict):
        self.cache = cache
        self.counts = {k: 0 for k in ["actives", "decoys", "invalid", "salts", "metal", "pains", "lipinski", "veber"]}
        self.sums = {k: 0.0 for k in ["mw", "clogp", "tpsa", "fsp3", "rb"]}
        self.rbs = {"actives": [], "decoys": []}

    def update(self, smi: str, label: str):
        """Update statistics with a single molecule"""
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
        """Generate report dictionary"""
        total = self.counts["actives"] + self.counts["decoys"]
        def safe_div(n, d):
            return (n / d) if d > 0 else 0.0
        
        return {
            "Dataset": dataset,
            "Target": target,
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
            "_Sum_MW": self.sums["mw"],
            "_Sum_cLogP": self.sums["clogp"],
            "_Sum_TPSA": self.sums["tpsa"],
            "_Sum_Fsp3": self.sums["fsp3"],
            "_Sum_RotB": self.sums["rb"],
            "_Count_Desc": total,
            "_RBs_Actives": self.rbs["actives"],
            "_RBs_DecoysOrInactives": self.rbs["decoys"],
        }


class DatasetAnalyser:
    """Main analysis engine for datasets"""
    
    def __init__(self, datasets: list, cache: dict):
        self.datasets = datasets
        self.cache = cache
        self._smiles_cache = None
    
    def collect_smiles(self) -> dict[str, dict[str, set[str]]]:
        """Collect all unique SMILES from all datasets"""
        if self._smiles_cache is not None:
            print("[INFO] Using cached SMILES data")
            return self._smiles_cache
        
        print("\n[INFO] Collecting unique SMILES from all datasets...")
        from collections import defaultdict
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
    
    def process_targets(self) -> pd.DataFrame:
        """Process all targets and generate per-target statistics"""
        print("\n[INFO] Processing individual targets (from cache)...")
        
        rows = []
        for dataset_obj in self.datasets:
            for target_name, target_path in tqdm(list(dataset_obj.enumerate_targets()), 
                                                  desc=f"Processing {dataset_obj.name}"):
                stats = TargetStats(self.cache)
                for smi, label in dataset_obj.read_target(target_path):
                    stats.update(smi, label)
                rows.append(stats.report(dataset_obj.name, target_name))
        
        print(f"[INFO] Completed processing {len(rows)} targets")
        return pd.DataFrame(rows).sort_values(["Dataset", "Target"]).reset_index(drop=True)
    
    def create_dataset_summary(self) -> pd.DataFrame:
        """Create dataset-level unique summary"""
        print("\n[INFO] Creating dataset-level summary...")
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
    
    def create_split_summary(self) -> pd.DataFrame:
        """Create split summary (actives/inactives/all)"""
        print("\n[INFO] Creating split summary (actives/inactives/all)...")
        ds_smiles = self.collect_smiles()
        rows = []
        
        for dataset in tqdm(list(ds_smiles.keys()), desc="Processing datasets"):
            buckets = ds_smiles[dataset]
            actives = list(buckets["active"])
            inactives = list(buckets["inactive"])
            
            print(f"\n  Processing {dataset}:")
            print(f"    Actives: {len(actives)} molecules")
            rows.append(self._summarise_bucket(actives, "Actives", dataset, buckets["active"], buckets["inactive"]))
            
            print(f"    Inactives: {len(inactives)} molecules")
            rows.append(self._summarise_bucket(inactives, "Inactives", dataset, buckets["active"], buckets["inactive"]))
            
            print(f"    All: {len(set(actives) | set(inactives))} unique molecules")
            all_ligs = list(set(actives) | set(inactives))
            rows.append(self._summarise_bucket(all_ligs, "All", dataset, buckets["active"], buckets["inactive"]))
        
        return pd.DataFrame(rows).sort_values(["Dataset", "Bucket"])
    
    def _summarise_bucket(self, smiles: list[str], bucket_label: str, 
                          dataset: str, actives: set[str], inactives: set[str]) -> dict:
        """Helper to summarise a bucket of SMILES"""
        agg = aggregate_from_cache(smiles, self.cache)
        total = agg["NumberLigands"]
        denom = total if total > 0 else np.nan
        means_denom = agg["_Count_Desc"] if agg["_Count_Desc"] > 0 else np.nan
        actives_fraction = len(actives) / (len(actives) + len(inactives)) if (len(actives) + len(inactives)) > 0 else 0.0
        
        return {
            "Dataset": dataset,
            "Bucket": bucket_label,
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
    
    def write_smiles_files(self, outdir: Path):
        """Write unique SMILES files for each dataset"""
        print("\n[INFO] Writing unique SMILES files...")
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