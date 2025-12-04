#!/usr/bin/env python3
import os
import argparse
from pathlib import Path
from collections import namedtuple
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

# --- RDKit Helpers (lazy import inside functions for workers) ---

# Define constants at the module level for clarity and efficiency
METALS = {3,4,11,12,13,19,20,21,22,23,24,25,26,27,28,29,30,31,37,38,39,40,41,42,43,44,45,46,47,48,49,50,55,56,57,72,73,74,75,76,77,78,79,80,81,82,83}
MolDescriptors = namedtuple("MolDescriptors", ["mw", "clogp", "tpsa", "fsp3", "rb", "lip_pass", "veber_pass", "has_metal", "pains_hit"])
_PAINS_CATALOG = None

def get_pains_catalog():
    """Initializes and returns the PAINS filter catalog (singleton per process)."""
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
    """Calculates a standard set of molecular descriptors."""
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
    """A generic reader for simple SMILES files, yielding (smiles, label)."""
    if not filepath.is_file():
        return
    with open(filepath, "r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            parts = line.strip().split()
            if parts:
                yield parts[0], label

def read_lit_pcba_target(target_dir: Path):
    """Yields (smiles, class_label) for LIT-PCBA actives/inactives."""
    yield from read_smi_file(target_dir / "actives.smi", "active")
    yield from read_smi_file(target_dir / "inactives.smi", "inactive")

def read_dekois2_target(target_dir: Path):
    """Yields (smiles, class_label) for DEKOIS2 actives/decoys."""
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
    """Yields (dataset_name, target_name, target_directory_path)."""
    for root in roots:
        if not root.is_dir():
            continue
        dataset = "LIT-PCBA" if root.name in ("LIT-PCBA", "LIT_PCBA") else root.name
        for entry in sorted(root.iterdir()):
            if entry.is_dir():
                yield dataset, entry.name, entry

# --- Worker Logic ---

class TargetStats:
    """A helper class to accumulate statistics for a single target."""
    def __init__(self):
        self.counts = {k: 0 for k in ["actives", "decoys", "invalid", "salts", "metal", "pains", "lipinski", "veber"]}
        self.sums = {k: 0.0 for k in ["mw", "clogp", "tpsa", "fsp3", "rb"]}
        self.rbs = {"actives": [], "decoys": []}

    def update(self, smi: str, label: str):
        """Processes one molecule and updates the running totals."""
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
            
        # Update boolean flags
        if desc.has_metal: self.counts["metal"] += 1
        if desc.pains_hit: self.counts["pains"] += 1
        if desc.lip_pass: self.counts["lipinski"] += 1
        if desc.veber_pass: self.counts["veber"] += 1

        # Update property sums for calculating means later
        for key in self.sums:
            self.sums[key] += getattr(desc, key)

    def report(self, dataset: str, target: str) -> dict:
        """Computes final metrics and returns a dictionary report."""
        total = self.counts["actives"] + self.counts["decoys"]
        
        def safe_div(numerator, denominator):
            return (numerator / denominator) if denominator > 0 else 0.0

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
            # Keep sums for global aggregation
            "_Sum_MW": self.sums["mw"], "_Sum_cLogP": self.sums["clogp"],
            "_Sum_TPSA": self.sums["tpsa"], "_Sum_Fsp3": self.sums["fsp3"],
            "_Sum_RotB": self.sums["rb"], "_Count_Desc": total,
            # Data for plotting
            "_RBs_Actives": self.rbs["actives"],
            "_RBs_DecoysOrInactives": self.rbs["decoys"],
        }

def process_one_target(task: tuple[str, str, Path]) -> dict:
    """Worker function: processes a single target directory and returns a stats dictionary."""
    dataset, target, target_dir = task
    
    stream = read_lit_pcba_target(target_dir) if dataset == "LIT-PCBA" else read_dekois2_target(target_dir)

    stats = TargetStats()
    for smi, label in stream:
        stats.update(smi, label)
        
    return stats.report(dataset, target)

# --- Plotting ---

OKABE_ITO_COLORS = { "act": "#0072B2", "decoy": "#D55E00", "all": "#999999" }

def save_violin_plot(data_lists, labels, title, outfile, colors, ylim_max=None):
    """Generates and saves a single violin plot."""
    fig, ax = plt.subplots(figsize=(6, 4))
    
    valid_data = [(d, l, c) for d, l, c in zip(data_lists, labels, colors) if d]
    if not valid_data:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")
    else:
        data, labs, cols = zip(*valid_data)
        v = ax.violinplot(data, showmeans=False, showmedians=True, showextrema=True)
        
        for body, color in zip(v['bodies'], cols):
            body.set_facecolor(color)
            body.set_alpha(0.85)
            body.set_edgecolor("black")
            body.set_linewidth(0.8)

        for part in ('cmedians', 'cbars', 'cmins', 'cmaxes'):
            if part in v and v[part] is not None:
                v[part].set_edgecolor("black")
                v[part].set_linewidth(0.8)
                
        ax.set_xticks(range(1, len(labs) + 1))
        ax.set_xticklabels(labs)
        ax.set_ylabel("Number of Rotatable Bonds")
        ax.set_title(title)
        if ylim_max is not None:
            ax.set_ylim(0, max(ylim_max, 1))

        handles = [Patch(facecolor=c, edgecolor="black", label=l, alpha=0.85) for l, c in zip(labs, cols)]
        ax.legend(handles=handles, loc="best", frameon=False)
        
    fig.tight_layout()
    fig.savefig(outfile, dpi=150)
    plt.close(fig)

def sanitize_filename(name: str) -> str:
    """Removes characters from a string that are unsuitable for a filename."""
    return "".join(c for c in name if c.isalnum() or c in "-._")

# --- Main Logic ---

def get_default_workers() -> int:
    """Determines a sensible default for worker count from environment or CPU count."""
    for var in ("SLURM_CPUS_PER_TASK", "SLURM_JOB_CPUS_PER_NODE", "PBS_NP"):
        if var in os.environ:
            try:
                # Extract the first number from variables like "16(x2)"
                val_str = str(os.environ[var]).split("(")[0].split(",")[0]
                return int(val_str)
            except (ValueError, IndexError):
                pass
    return os.cpu_count() or 1

def create_protein_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregates per-target data to a per-protein summary."""
    sum_cols = [
        "NumberActives", "NumberDecoys/Inactives", "NumberLigandsTotal",
        "NumberInvalidSMILES", "NumberWithSalts", "NumberWithMetals",
        "NumberPAINSMatches", "NumberLipinskiCompliant", "NumberVeberCompliant",
        "_Sum_MW", "_Sum_cLogP", "_Sum_TPSA", "_Sum_Fsp3", "_Sum_RotB", "_Count_Desc"
    ]
    protein_df = df.groupby("Target", as_index=False)[sum_cols].sum()

    total_ligands = protein_df["NumberLigandsTotal"].replace(0, np.nan)
    total_descs = protein_df["_Count_Desc"].replace(0, np.nan)

    protein_df["LipinskiComplianceRate"] = (protein_df["NumberLipinskiCompliant"] / total_ligands).fillna(0)
    protein_df["VeberComplianceRate"] = (protein_df["NumberVeberCompliant"] / total_ligands).fillna(0)
    protein_df["ActivesFraction"] = (protein_df["NumberActives"] / total_ligands).fillna(0)
    
    protein_df["Mean_MW"] = protein_df["_Sum_MW"] / total_descs
    protein_df["Mean_cLogP"] = protein_df["_Sum_cLogP"] / total_descs
    protein_df["Mean_TPSA"] = protein_df["_Sum_TPSA"] / total_descs
    protein_df["Mean_Fsp3"] = protein_df["_Sum_Fsp3"] / total_descs
    protein_df["Mean_RotatableBonds"] = protein_df["_Sum_RotB"] / total_descs
    
    output_cols = [
        "Target", "NumberActives", "NumberDecoys/Inactives", "NumberLigandsTotal",
        "NumberInvalidSMILES", "NumberWithSalts", "NumberWithMetals", "NumberPAINSMatches",
        "NumberLipinskiCompliant", "LipinskiComplianceRate", "NumberVeberCompliant",
        "VeberComplianceRate", "Mean_MW", "Mean_cLogP", "Mean_TPSA", "Mean_Fsp3",
        "Mean_RotatableBonds", "ActivesFraction"
    ]
    return protein_df[output_cols].sort_values("Target")

def generate_global_plots(df: pd.DataFrame, out_dir: Path):
    """Generates and saves the global violin plots for each dataset."""
    for dataset, decoy_label in [("DEKOIS2", "Decoys"), ("LIT-PCBA", "Inactives")]:
        mask = df["Dataset"] == dataset
        if not mask.any(): continue
        
        actives = [rb for sublist in df.loc[mask, "_RBs_Actives"] for rb in sublist]
        decoys = [rb for sublist in df.loc[mask, "_RBs_DecoysOrInactives"] for rb in sublist]
        
        save_violin_plot(
            data_lists=[actives, decoys, actives + decoys],
            labels=["Actives", decoy_label, "All"],
            title=f"{dataset}: Rotatable Bonds",
            outfile=out_dir / f"violin_{dataset}_rotatable_bonds.png",
            colors=[OKABE_ITO_COLORS["act"], OKABE_ITO_COLORS["decoy"], OKABE_ITO_COLORS["all"]],
            ylim_max=max([0] + actives + decoys)
        )

def generate_per_target_lit_pcba_plots(df: pd.DataFrame, out_dir: Path) -> Path | None:
    """Generates violin plots for each LIT-PCBA target and returns the output directory path."""
    lit_pcba_df = df[df["Dataset"] == "LIT-PCBA"]
    if lit_pcba_df.empty: return None
        
    plot_dir = out_dir / "violin_LIT-PCBA_targets"
    plot_dir.mkdir(exist_ok=True)
    
    all_rbs = [rb for series in (lit_pcba_df["_RBs_Actives"], lit_pcba_df["_RBs_DecoysOrInactives"])
               for sublist in series for rb in sublist]
    global_max_rb = max(all_rbs) if all_rbs else 0

    desc = "Plotting LIT-PCBA targets"
    for _, row in tqdm(lit_pcba_df.iterrows(), total=len(lit_pcba_df), desc=desc):
        a, d = row["_RBs_Actives"], row["_RBs_DecoysOrInactives"]
        fname = f"violin_LIT-PCBA_{sanitize_filename(row['Target'])}_rotatable_bonds.png"
        
        save_violin_plot(
            data_lists=[a, d, a + d], labels=["Actives", "Inactives", "All"],
            title=f"LIT-PCBA {row['Target']}: Rotatable Bonds",
            outfile=plot_dir / fname,
            colors=[OKABE_ITO_COLORS["act"], OKABE_ITO_COLORS["decoy"], OKABE_ITO_COLORS["all"]],
            ylim_max=global_max_rb
        )
    return plot_dir

def main():
    """Main execution function to parse arguments and run the analysis."""
    parser = argparse.ArgumentParser(description="DEKOIS2/LIT-PCBA dataset summarizer and plotter.")
    parser.add_argument("--roots", nargs="*", default=["LIT-PCBA", "DEKOIS2"],
                        help="Dataset root directories to scan.")
    parser.add_argument("--workers", type=int, default=get_default_workers(),
                        help="Number of worker processes (default: auto-detected).")
    parser.add_argument("--outdir", type=Path, default=Path("."),
                        help="Output directory for CSVs and plots.")
    args = parser.parse_args()

    args.outdir.mkdir(exist_ok=True)
    
    tasks = list(enumerate_targets([Path(r) for r in args.roots]))
    if not tasks:
        raise SystemExit("Error: No targets found under the specified root directories.")

    # --- Parallel Processing ---
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(process_one_target, task) for task in tasks]
        desc = "Processing targets"
        for future in tqdm(as_completed(futures), total=len(tasks), desc=desc, unit="target"):
            rows.append(future.result())

    if not rows:
        raise SystemExit("Processing finished, but no data was generated.")

    # --- Data Handling and CSV Output ---
    df = pd.DataFrame(rows).sort_values(["Dataset", "Target"]).reset_index(drop=True)

    # Save general summary
    df.drop(columns=[c for c in df.columns if c.startswith("_")]).to_csv(
        args.outdir / "dataset_summary.csv", index=False)

    # Save LIT-PCBA specific summary
    lit_pcba_df = df[df["Dataset"] == "LIT-PCBA"].copy()
    lit_pcba_df.rename(columns={"NumberDecoys/Inactives": "NumberInactives"}, inplace=True)
    lit_pcba_df.drop(columns=[c for c in lit_pcba_df.columns if c.startswith("_")]).to_csv(
        args.outdir / "per_target_summary.csv", index=False)

    # Save protein-level aggregated summary
    protein_summary_df = create_protein_summary(df)
    protein_summary_df.to_csv(args.outdir / "protein_summary.csv", index=False)

    # --- Plotting ---
    generate_global_plots(df, args.outdir)
    per_target_dir = generate_per_target_lit_pcba_plots(df, args.outdir)

    # --- Final Report ---
    print(f"\n[OK] Analysis complete. {len(df)} targets processed.")
    print("Generated files:")
    print(f" - {args.outdir / 'dataset_summary.csv'}")
    print(f" - {args.outdir / 'per_target_summary.csv'}")
    print(f" - {args.outdir / 'protein_summary.csv'}")
    print(f" - {args.outdir / 'violin_DEKOIS2_rotatable_bonds.png'}")
    print(f" - {args.outdir / 'violin_LIT-PCBA_rotatable_bonds.png'}")
    if per_target_dir:
        print(f" - {per_target_dir}/ (containing per-target plots)")

if __name__ == "__main__":
    main()