#!/usr/bin/env python3
import os
import argparse
from pathlib import Path
from collections import namedtuple, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

# Matplotlib setup
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# Avoid BLAS/OpenMP oversubscription in workers
for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(var, "1")

# --- RDKit Helpers ---
METALS = {3,4,11,12,13,19,20,21,22,23,24,25,26,27,28,29,30,31,37,38,39,40,41,42,43,44,45,46,47,48,49,50,55,56,57,72,73,74,75,76,77,78,79,80,81,82,83}
MolDescriptors = namedtuple("MolDescriptors", ["mw", "clogp", "tpsa", "fsp3", "rb", "lip_pass", "veber_pass", "has_metal", "pains_hit"])
_PAINS_CATALOG = None

def get_pains_catalog():
    global _PAINS_CATALOG
    if _PAINS_CATALOG is None:
        from rdkit.Chem import FilterCatalog
        params = FilterCatalog.FilterCatalogParams()
        for catalog in (FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS_A,
                        FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS_B,
                        FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS_C):
            params.AddCatalog(catalog)
        _PAINS_CATALOG = FilterCatalog.FilterCatalog(params)
    return _PAINS_CATALOG

def calculate_descriptors(mol) -> MolDescriptors:
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
    pains_hit = get_pains_catalog().HasMatch(mol)
    return MolDescriptors(mw, clogp, tpsa, fsp3, rb, lip_pass, veber_pass, has_metal, pains_hit)

# --- Dataset Readers ---
def read_smi_file(filepath: Path, label: str):
    if not filepath.is_file():
        return
    with open(filepath, "r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            parts = line.strip().split()
            if parts:
                yield parts[0], label

def read_lit_pcba_target(target_dir: Path):
    yield from read_smi_file(target_dir / "actives.smi", "active")
    yield from read_smi_file(target_dir / "inactives.smi", "inactive")

def read_dudez_target(target_dir: Path):
    yield from read_smi_file(target_dir / "actives_final.ism", "active")
    yield from read_smi_file(target_dir / "decoys_final.ism", "decoy")

def read_dekois2_target(target_dir: Path):
    smi_path = target_dir / "active_decoys.smi"
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

def enumerate_targets(roots: list[Path]):
    for root in roots:
        if not root.is_dir():
            continue
        dataset = "LIT-PCBA" if root.name in ("LIT-PCBA", "LIT_PCBA") else root.name
        for entry in sorted(root.iterdir()):
            if entry.is_dir():
                yield dataset, entry.name, entry

# --- Target Worker ---
class TargetStats:
    def __init__(self):
        self.counts = {k: 0 for k in ["actives", "decoys", "invalid", "salts", "metal", "pains", "lipinski", "veber"]}
        self.sums = {k: 0.0 for k in ["mw", "clogp", "tpsa", "fsp3", "rb"]}
        self.rbs = {"actives": [], "decoys": []}

    def update(self, smi: str, label: str):
        from rdkit import Chem
        if "." in smi:
            self.counts["salts"] += 1
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            self.counts["invalid"] += 1
            return
        desc = calculate_descriptors(mol)
        if label == "active":
            self.counts["actives"] += 1
            self.rbs["actives"].append(desc.rb)
        else:
            self.counts["decoys"] += 1
            self.rbs["decoys"].append(desc.rb)
        if desc.has_metal: self.counts["metal"] += 1
        if desc.pains_hit: self.counts["pains"] += 1
        if desc.lip_pass: self.counts["lipinski"] += 1
        if desc.veber_pass: self.counts["veber"] += 1
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

def process_one_target(task: tuple[str,str,Path]) -> dict:
    dataset, target, target_dir = task
    if dataset == "LIT-PCBA":
        stream = read_lit_pcba_target(target_dir)
    elif dataset == "DEKOIS2":
        stream = read_dekois2_target(target_dir)
    elif dataset == "DUDE-Z":
        stream = read_dudez_target(target_dir)
    else:
        return None
    stats = TargetStats()
    for smi, label in stream:
        stats.update(smi,label)
    return stats.report(dataset,target)

# --- Helpers for unique ligand summaries ---
def _calc_one_smiles(smi: str):
    from rdkit import Chem
    salts = "." in smi
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return False, salts, None
    desc = calculate_descriptors(mol)
    return True, salts, desc

def _calc_desc_for_smiles_parallel(smiles: list[str], workers: int) -> dict:
    results = {"NumberLigands":0,"NumberInvalidSMILES":0,"NumberWithSalts":0,
               "NumberWithMetals":0,"NumberPAINSMatches":0,
               "NumberLipinskiCompliant":0,"NumberVeberCompliant":0,
               "_Sum_MW":0.0,"_Sum_cLogP":0.0,"_Sum_TPSA":0.0,
               "_Sum_Fsp3":0.0,"_Sum_RotB":0.0,"_Count_Desc":0}
    with ProcessPoolExecutor(max_workers=workers) as exe:
        futs=[exe.submit(_calc_one_smiles,s) for s in smiles]
        for fut in as_completed(futs):
            ok,salts,desc=fut.result()
            if not ok: 
                results["NumberInvalidSMILES"]+=1; continue
            results["NumberLigands"]+=1
            if salts: results["NumberWithSalts"]+=1
            if desc.has_metal: results["NumberWithMetals"]+=1
            if desc.pains_hit: results["NumberPAINSMatches"]+=1
            if desc.lip_pass: results["NumberLipinskiCompliant"]+=1
            if desc.veber_pass: results["NumberVeberCompliant"]+=1
            results["_Sum_MW"]+=desc.mw; results["_Sum_cLogP"]+=desc.clogp
            results["_Sum_TPSA"]+=desc.tpsa; results["_Sum_Fsp3"]+=desc.fsp3
            results["_Sum_RotB"]+=desc.rb; results["_Count_Desc"]+=1
    return results

def _collect_dataset_unique_smiles(roots: list[Path]) -> dict[str,dict[str,set[str]]]:
    dataset_to_smiles=defaultdict(lambda:{"active":set(),"inactive":set()})
    for dataset,_,tdir in enumerate_targets(roots):
        if dataset=="LIT-PCBA": gen=read_lit_pcba_target(tdir)
        elif dataset=="DEKOIS2": gen=read_dekois2_target(tdir)
        elif dataset=="DUDE-Z": gen=read_dudez_target(tdir)
        else: continue
        for smi,label in gen:
            if label=="active": dataset_to_smiles[dataset]["active"].add(smi)
            else: dataset_to_smiles[dataset]["inactive"].add(smi)
    return dataset_to_smiles

def create_dataset_unique_summary(roots: list[Path], workers:int)->pd.DataFrame:
    ds_smiles=_collect_dataset_unique_smiles(roots)
    rows=[]
    for dataset,buckets in ds_smiles.items():
        smiles=list(buckets["active"]|buckets["inactive"])
        agg=_calc_desc_for_smiles_parallel(smiles,workers)
        total=agg["NumberLigands"]; denom=total if total>0 else np.nan
        means_denom=agg["_Count_Desc"] if agg["_Count_Desc"]>0 else np.nan
        row={"Dataset":dataset,
             "NumberLigandsUnique":int(total),
             "NumberActivesUnique":len(buckets["active"]),
             "NumberInactivesUnique":len(buckets["inactive"]),
             "NumberInvalidSMILES":int(agg["NumberInvalidSMILES"]),
             "NumberWithSalts":int(agg["NumberWithSalts"]),
             "NumberWithMetals":int(agg["NumberWithMetals"]),
             "NumberPAINSMatches":int(agg["NumberPAINSMatches"]),
             "NumberLipinskiCompliant":int(agg["NumberLipinskiCompliant"]),
             "LipinskiComplianceRate":(agg["NumberLipinskiCompliant"]/denom) if denom==denom else 0.0,
             "NumberVeberCompliant":int(agg["NumberVeberCompliant"]),
             "VeberComplianceRate":(agg["NumberVeberCompliant"]/denom) if denom==denom else 0.0,
             "Mean_MW":(agg["_Sum_MW"]/means_denom) if means_denom==means_denom else 0.0,
             "Mean_cLogP":(agg["_Sum_cLogP"]/means_denom) if means_denom==means_denom else 0.0,
             "Mean_TPSA":(agg["_Sum_TPSA"]/means_denom) if means_denom==means_denom else 0.0,
             "Mean_Fsp3":(agg["_Sum_Fsp3"]/means_denom) if means_denom==means_denom else 0.0,
             "Mean_RotatableBonds":(agg["_Sum_RotB"]/means_denom) if means_denom==means_denom else 0.0,
             "ActivesFraction":len(buckets["active"])/denom if denom==denom else 0.0}
        rows.append(row)
    return pd.DataFrame(rows).sort_values("Dataset")



def _summarize_bucket(smiles:list[str],workers:int,bucket_label:str,dataset:str,actives:set[str],inactives:set[str])->dict:
    agg=_calc_desc_for_smiles_parallel(smiles,workers)
    total=agg["NumberLigands"]; denom=total if total>0 else np.nan
    means_denom=agg["_Count_Desc"] if agg["_Count_Desc"]>0 else np.nan
    actives_fraction=len(actives)/(len(actives)+len(inactives)) if (len(actives)+len(inactives))>0 else 0.0
    return {"Dataset":dataset,"Bucket":bucket_label,
            "NumberLigands":int(total),
            "NumberInvalidSMILES":int(agg["NumberInvalidSMILES"]),
            "NumberWithSalts":int(agg["NumberWithSalts"]),
            "NumberWithMetals":int(agg["NumberWithMetals"]),
            "NumberPAINSMatches":int(agg["NumberPAINSMatches"]),
            "NumberLipinskiCompliant":int(agg["NumberLipinskiCompliant"]),
            "LipinskiComplianceRate":(agg["NumberLipinskiCompliant"]/denom) if denom==denom else 0.0,
            "NumberVeberCompliant":int(agg["NumberVeberCompliant"]),
            "VeberComplianceRate":(agg["NumberVeberCompliant"]/denom) if denom==denom else 0.0,
            "Mean_MW":(agg["_Sum_MW"]/means_denom) if means_denom==means_denom else 0.0,
            "Mean_cLogP":(agg["_Sum_cLogP"]/means_denom) if means_denom==means_denom else 0.0,
            "Mean_TPSA":(agg["_Sum_TPSA"]/means_denom) if means_denom==means_denom else 0.0,
            "Mean_Fsp3":(agg["_Sum_Fsp3"]/means_denom) if means_denom==means_denom else 0.0,
            "Mean_RotatableBonds":(agg["_Sum_RotB"]/means_denom) if means_denom==means_denom else 0.0,
            "ActivesFraction":actives_fraction}

def create_dataset_unique_summary_split(roots:list[Path],workers:int)->pd.DataFrame:
    ds_smiles=_collect_dataset_unique_smiles(roots)
    rows=[]
    for dataset,buckets in ds_smiles.items():
        actives=list(buckets["active"]); inactives=list(buckets["inactive"]); all_ligs=list(set(actives)|set(inactives))
        rows.append(_summarize_bucket(actives,workers,"Actives",dataset,buckets["active"],buckets["inactive"]))
        rows.append(_summarize_bucket(inactives,workers,"Inactives",dataset,buckets["active"],buckets["inactive"]))
        rows.append(_summarize_bucket(all_ligs,workers,"All",dataset,buckets["active"],buckets["inactive"]))
    return pd.DataFrame(rows).sort_values(["Dataset","Bucket"])

# --- (plotting + protein_summary unchanged; omitted for brevity, keep from your version) ---

def write_smiles_files(roots: list[Path], outdir: Path):
    outdir.mkdir(exist_ok=True, parents=True)
    ds_smiles = _collect_dataset_unique_smiles(roots)
    for dataset, buckets in ds_smiles.items():
        actives_file = outdir / f"{dataset}_actives.smi"
        inactives_file = outdir / f"{dataset}_inactives.smi"

        # Write actives
        with open(actives_file, "w") as fa:
            for smi in sorted(buckets["active"]):
                fa.write(f"{smi}\n")

        # Write inactives
        with open(inactives_file, "w") as fi:
            for smi in sorted(buckets["inactive"]):
                fi.write(f"{smi}\n")

    print(f"[OK] Wrote SMILES files for each dataset to {outdir}")



# --- Main ---
def get_default_workers()->int:
    for var in ("SLURM_CPUS_PER_TASK","SLURM_JOB_CPUS_PER_NODE","PBS_NP"):
        if var in os.environ:
            try:
                return int(str(os.environ[var]).split("(")[0].split(",")[0])
            except: pass
    return os.cpu_count() or 1

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--roots", nargs="*", default=["LIT-PCBA", "DEKOIS2", "DUDE-Z"])
    parser.add_argument("--workers", type=int, default=get_default_workers())
    parser.add_argument("--outdir", type=Path, default=Path("."))
    parser.add_argument("--write-smiles-only", action="store_true",
                        help="Only write actives/inactives SMILES files and exit.")
    args = parser.parse_args()
    args.outdir.mkdir(exist_ok=True)

    roots = [Path(r) for r in args.roots]

    # ✅ If only SMILES requested
    if args.write_smiles_only:
        write_smiles_files(roots, args.outdir)
        return

    # --- otherwise run your full pipeline as before ---
    tasks = list(enumerate_targets(roots))
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as exe:
        futs = [exe.submit(process_one_target, t) for t in tasks]
        for fut in tqdm(as_completed(futs), total=len(tasks), desc="Processing targets"):
            res = fut.result()
            if res:
                rows.append(res)
    df = pd.DataFrame(rows).sort_values(["Dataset", "Target"]).reset_index(drop=True)
    df.drop(columns=[c for c in df.columns if c.startswith("_")]).to_csv(args.outdir/"dataset_summary.csv", index=False)
    lit_pcba_df = df[df["Dataset"]=="LIT-PCBA"].copy()
    lit_pcba_df.rename(columns={"NumberDecoys/Inactives":"NumberInactives"}, inplace=True)
    lit_pcba_df.drop(columns=[c for c in lit_pcba_df.columns if c.startswith("_")]).to_csv(args.outdir/"per_target_summary.csv", index=False)
    dataset_unique_df = create_dataset_unique_summary(roots, args.workers)
    dataset_unique_df.to_csv(args.outdir/"dataset_unique_summary.csv", index=False)
    dataset_unique_split_df = create_dataset_unique_summary_split(roots, args.workers)
    dataset_unique_split_df.to_csv(args.outdir/"dataset_unique_summary_split.csv", index=False)
    print("[OK] Done.")

if __name__=="__main__":
    main()
