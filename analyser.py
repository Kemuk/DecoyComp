#!/usr/bin/env python3
"""
Analysis engine for dataset statistics and reporting.

Uses Polars for fast DataFrame operations.

Modes:
  - Manifest mode (Analyser): Works with canonical manifest parquet
  - Legacy mode (DatasetAnalyser): Works with dataset readers (backwards compatibility)
"""
from pathlib import Path
from collections import defaultdict

import numpy as np
import polars as pl
from tqdm.auto import tqdm

from molecular_utils import aggregate_from_cache


class TargetStats:
    """Accumulates statistics for a single target."""

    def __init__(self, cache: dict):
        self.cache = cache
        self.counts = {k: 0 for k in ["actives", "decoys", "invalid", "salts", "metal", "pains", "lipinski", "veber"]}
        self.sums = {k: 0.0 for k in ["mw", "clogp", "tpsa", "fsp3", "rb"]}
        self.rbs = {"actives": [], "decoys": []}

    def update(self, smi: str, label: str):
        """Update statistics with a single molecule."""
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
        """Generate report dictionary."""
        total = self.counts["actives"] + self.counts["decoys"]

        def safe_div(n, d):
            return (n / d) if d > 0 else 0.0

        return {
            "Dataset": dataset,
            "Target": target,
            "NumberActives": self.counts["actives"],
            "NumberInactives": self.counts["decoys"],
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


class Analyser:
    """
    Analyser for manifest-only mode (no legacy dataset reader support).
    
    Works with canonical manifest DataFrame containing all ligand data.
    """
    
    def __init__(self, manifest_df: pl.DataFrame):
        """
        Initialize analyser in manifest mode only.
        
        Args:
            manifest_df: Canonical manifest DataFrame with required columns
        
        Raises:
            TypeError: if manifest_df is None
            ValueError: if required columns missing
        """
        if manifest_df is None:
            raise TypeError("Analyser requires manifest_df (backwards compatibility removed)")
        
        required_cols = {
            'manifest_id', 'dataset', 'target_id', 'protein_id', 'label', 'smiles',
            'ligand_id', 'compound_key', 'file_path', 'ligand_file_path',
            'protein_file_path', 'cache_location', 'source_split'
        }
        if not required_cols.issubset(set(manifest_df.columns)):
            missing = required_cols - set(manifest_df.columns)
            raise ValueError(f"Missing required columns: {missing}")
        
        self.manifest_df = manifest_df
        self.cache = {}  # Descriptor cache
    
    def _compute_target_descriptor(self, fingerprints_list: list) -> pl.Series | None:
        """
        Compute aggregate target descriptor from compound fingerprints.
        
        For now, returns None. Can be extended to implement actual aggregation.
        """
        if not fingerprints_list:
            return None
        # Placeholder: aggregate fingerprints if needed
        return None
    
    def collect_smiles(self) -> pl.DataFrame:
        """
        Return unique SMILES from manifest.
        
        Returns:
            DataFrame with smiles and dataset columns
        """
        return self.manifest_df.select(['smiles', 'dataset']).unique()
    
    def process_targets(self) -> pl.DataFrame:
        """
        Process all targets from manifest with full vectorization.
        
        Returns:
            DataFrame with per-target statistics
        """
        return self._process_targets_from_manifest()
    
    def _process_targets_from_manifest(self) -> pl.DataFrame:
        """
        Process all targets from manifest with full Polars vectorization.
        
        Returns:
            DataFrame with per-target statistics
        """
        # Compute fingerprints for each unique compound_key
        fingerprints = (
            self.manifest_df
            .unique(subset=['compound_key'])
            .with_columns([
                pl.col('smiles').map_elements(
                    lambda smi: self._rdkit_fingerprint_or_none(smi),
                    return_dtype=pl.Binary
                ).alias('fingerprint')
            ])
            .select(['compound_key', 'fingerprint'])
        )
        
        # Aggregate by target
        target_stats = (
            self.manifest_df
            .join(fingerprints, on='compound_key', how='left')
            .group_by('target_id')
            .agg([
                pl.col('dataset').first().alias('dataset'),
                pl.col('protein_id').first().alias('protein_id'),
                pl.col('smiles').unique().count().alias('num_ligands'),
                pl.col('smiles').unique().alias('smiles_list'),
                pl.col('label').unique().alias('label_values'),
                pl.col('fingerprint').alias('fingerprint_list'),
            ])
        )
        
        # Build TargetInfo objects via vectorized map_rows
        rows = []
        for row in target_stats.iter_rows(named=True):
            # Create TargetStats-compatible report for each target
            report = {
                "Dataset": row['dataset'],
                "Target": row['target_id'],
                "NumberActives": 0,
                "NumberInactives": 0,
                "NumberLigandsTotal": row['num_ligands'],
                "NumberInvalidSMILES": 0,
                "NumberWithSalts": 0,
                "NumberWithMetals": 0,
                "NumberPAINSMatches": 0,
                "NumberLipinskiCompliant": 0,
                "LipinskiComplianceRate": 0.0,
                "NumberVeberCompliant": 0,
                "VeberComplianceRate": 0.0,
                "ActivesFraction": 0.0,
                "Mean_MW": 0.0,
                "Mean_cLogP": 0.0,
                "Mean_TPSA": 0.0,
                "Mean_Fsp3": 0.0,
                "Mean_RotatableBonds": 0.0,
                "_Sum_MW": 0.0,
                "_Sum_cLogP": 0.0,
                "_Sum_TPSA": 0.0,
                "_Sum_Fsp3": 0.0,
                "_Sum_RotB": 0.0,
                "_Count_Desc": row['num_ligands'],
                "_RBs_Actives": [],
                "_RBs_DecoysOrInactives": [],
            }
            
            # Accumulate stats from cache
            for smi in row['smiles_list']:
                if smi in self.cache:
                    salts, desc = self.cache[smi]
                    if desc is not None:
                        # Accumulate means
                        report['_Sum_MW'] += desc.mw
                        report['_Sum_cLogP'] += desc.clogp
                        report['_Sum_TPSA'] += desc.tpsa
                        report['_Sum_Fsp3'] += desc.fsp3
                        report['_Sum_RotB'] += desc.rb
            
            # Compute means
            denom = report['_Count_Desc'] if report['_Count_Desc'] > 0 else 1
            report['Mean_MW'] = report['_Sum_MW'] / denom
            report['Mean_cLogP'] = report['_Sum_cLogP'] / denom
            report['Mean_TPSA'] = report['_Sum_TPSA'] / denom
            report['Mean_Fsp3'] = report['_Sum_Fsp3'] / denom
            report['Mean_RotatableBonds'] = report['_Sum_RotB'] / denom
            
            rows.append(report)
        
        return pl.DataFrame(rows).sort(['Dataset', 'Target'])
    
    def _rdkit_fingerprint_or_none(self, smiles: str):
        """Placeholder for fingerprint generation."""
        return None


class DatasetAnalyser:
    """Main analysis engine for datasets."""

    def __init__(self, datasets: list, cache: dict, max_ligands_per_dataset: int | None = None):
        self.datasets = datasets
        self.cache = cache
        self.max_ligands_per_dataset = max_ligands_per_dataset
        self._smiles_cache = None

    def _collect_dataset_smiles(self, dataset_obj) -> tuple[set[str], set[str], int]:
        """Collect unique SMILES for one dataset, respecting the configured cap."""
        actives = set()
        inactives = set()
        target_count = 0
        limit = self.max_ligands_per_dataset

        for _target_name, target_path in dataset_obj.enumerate_targets():
            target_count += 1
            for smi, label in dataset_obj.read_target(target_path):
                if smi in actives or smi in inactives:
                    continue

                if limit is not None and len(actives) + len(inactives) >= limit:
                    return actives, inactives, target_count

                if label == "active":
                    actives.add(smi)
                else:
                    inactives.add(smi)

                if limit is not None and len(actives) + len(inactives) >= limit:
                    return actives, inactives, target_count

        return actives, inactives, target_count

    def collect_smiles(self) -> dict[str, dict[str, set[str]]]:
        """Collect all unique SMILES from all datasets."""
        if self._smiles_cache is not None:
            print("[INFO] Using cached SMILES data")
            return self._smiles_cache

        print("\n[INFO] Collecting unique SMILES from all datasets...")
        dataset_to_smiles = defaultdict(lambda: {"active": set(), "inactive": set()})

        for dataset_obj in tqdm(self.datasets, desc="Scanning datasets"):
            actives, inactives, target_count = self._collect_dataset_smiles(dataset_obj)
            dataset_to_smiles[dataset_obj.name]["active"] = actives
            dataset_to_smiles[dataset_obj.name]["inactive"] = inactives

            actives_count = len(actives)
            inactives_count = len(inactives)
            print(f"  {dataset_obj.name}: {target_count} targets, {actives_count:,} unique actives, {inactives_count:,} unique inactives")

        self._smiles_cache = dict(dataset_to_smiles)
        return self._smiles_cache

    def process_targets(self) -> pl.DataFrame:
        """Process all targets and generate per-target statistics."""
        print("\n[INFO] Processing individual targets (from cache)...")

        rows = []
        ds_smiles = self.collect_smiles()
        for dataset_obj in self.datasets:
            allowed = None
            if self.max_ligands_per_dataset is not None:
                buckets = ds_smiles[dataset_obj.name]
                allowed = buckets["active"] | buckets["inactive"]

            for target_name, target_path in tqdm(dataset_obj.enumerate_targets(), desc=f"Processing {dataset_obj.name}"):
                stats = TargetStats(self.cache)
                for smi, label in dataset_obj.read_target(target_path):
                    if allowed is None or smi in allowed:
                        stats.update(smi, label)
                report = stats.report(dataset_obj.name, target_name)
                if self.max_ligands_per_dataset is None or report["NumberLigandsTotal"] > 0:
                    rows.append(report)

        print(f"[INFO] Completed processing {len(rows)} targets")

        # Create Polars DataFrame and sort
        df = pl.DataFrame(rows)
        return df.sort(["Dataset", "Target"])

    def create_dataset_summary(self) -> pl.DataFrame:
        """Create dataset-level unique summary."""
        print("\n[INFO] Creating dataset-level summary...")
        ds_smiles = self.collect_smiles()
        rows = []

        for dataset in tqdm(ds_smiles.keys(), desc="Processing datasets"):
            buckets = ds_smiles[dataset]
            smiles = buckets["active"] | buckets["inactive"]

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

        return pl.DataFrame(rows).sort("Dataset")

    def create_split_summary(self) -> pl.DataFrame:
        """Create split summary (actives/inactives/all)."""
        print("\n[INFO] Creating split summary (actives/inactives/all)...")
        ds_smiles = self.collect_smiles()
        rows = []

        for dataset in tqdm(ds_smiles.keys(), desc="Processing datasets"):
            buckets = ds_smiles[dataset]
            actives = buckets["active"]
            inactives = buckets["inactive"]

            print(f"\n  Processing {dataset}:")
            print(f"    Actives: {len(actives):,} molecules")
            rows.append(self._summarise_bucket(actives, "Actives", dataset, buckets["active"], buckets["inactive"]))

            print(f"    Inactives: {len(inactives):,} molecules")
            rows.append(self._summarise_bucket(inactives, "Inactives", dataset, buckets["active"], buckets["inactive"]))

            print(f"    All: {len(actives | inactives):,} unique molecules")
            all_ligs = actives | inactives
            rows.append(self._summarise_bucket(all_ligs, "All", dataset, buckets["active"], buckets["inactive"]))

        return pl.DataFrame(rows).sort(["Dataset", "Bucket"])

    def _summarise_bucket(
        self,
        smiles,
        bucket_label: str,
        dataset: str,
        actives: set[str],
        inactives: set[str]
    ) -> dict:
        """Helper to summarise a bucket of SMILES."""
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
        """Write unique SMILES files for each dataset (skip if already exists)."""
        print("\n[INFO] Writing unique SMILES files...")
        outdir.mkdir(exist_ok=True, parents=True)
        ds_smiles = self.collect_smiles()

        files_written = 0
        files_skipped = 0

        for dataset, buckets in ds_smiles.items():
            actives_file = outdir / f"{dataset}_actives.smi"
            inactives_file = outdir / f"{dataset}_inactives.smi"

            # Check if both files already exist
            if actives_file.exists() and inactives_file.exists():
                print(f"  Skipping {dataset} (files already exist)")
                files_skipped += 2
                continue

            print(f"  Writing {dataset}...")

            if not actives_file.exists():
                with open(actives_file, "w") as fa:
                    for smi in sorted(buckets["active"]):
                        fa.write(f"{smi}\n")
                print(f"    - {actives_file.name}: {len(buckets['active']):,} actives")
                files_written += 1
            else:
                print(f"    - {actives_file.name}: skipped (already exists)")
                files_skipped += 1

            if not inactives_file.exists():
                with open(inactives_file, "w") as fi:
                    for smi in sorted(buckets["inactive"]):
                        fi.write(f"{smi}\n")
                print(f"    - {inactives_file.name}: {len(buckets['inactive']):,} inactives")
                files_written += 1
            else:
                print(f"    - {inactives_file.name}: skipped (already exists)")
                files_skipped += 1

        print(f"[OK] SMILES files in {outdir}: {files_written} written, {files_skipped} skipped")
