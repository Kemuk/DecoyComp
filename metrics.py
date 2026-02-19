#!/usr/bin/env python3
"""
Visualization and descriptor analysis with Polars and joblib.

Uses Polars for fast data loading and processing, converts to pandas
only for matplotlib/seaborn visualization compatibility.
"""
import argparse
from pathlib import Path

import polars as pl
import pandas as pd
from joblib import Parallel, delayed
from tqdm.auto import tqdm
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
import matplotlib.pyplot as plt

# Colours (colourblind safe)
COLOR_ACTIVE = "#0072B2"   # blue
COLOR_INACTIVE = "#D55E00"  # orange


def compute_desc(args: tuple[str, str, str]) -> dict | None:
    """Worker function for parallel processing. Takes (smi, dataset, bucket) tuple."""
    smi, dataset, bucket = args
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    return {
        "dataset": dataset,
        "bucket": bucket,
        "mw": Descriptors.MolWt(mol),
        "tpsa": rdMolDescriptors.CalcTPSA(mol),
        "rotbonds": Descriptors.NumRotatableBonds(mol),
    }


def read_smiles(smiles_dir: Path) -> list[tuple[str, str, str]]:
    """Read all SMILES files and return list of (smiles, dataset, bucket) tuples."""
    files = list(smiles_dir.glob("*_actives.smi")) + list(smiles_dir.glob("*_inactives.smi"))

    smiles_list = []
    print("[INFO] Reading SMILES files...")
    for f in tqdm(files, desc="Reading files"):
        dataset = f.stem.replace("_actives", "").replace("_inactives", "")
        bucket = "active" if "_actives" in f.stem else "inactive"
        with open(f) as fh:
            for line in fh:
                smi = line.strip()
                if smi:
                    smiles_list.append((smi, dataset, bucket))

    print(f"[INFO] Read {len(smiles_list):,} SMILES strings")
    return smiles_list


def process_smiles_parallel(
    smiles_dir: Path,
    out_csv: Path,
    batch_size: int = 1000,
    max_workers: int = -1
) -> pl.DataFrame:
    """Process SMILES in parallel and write results in batches using joblib."""

    # Read all SMILES
    smiles_list = read_smiles(smiles_dir)

    if out_csv.exists():
        out_csv.unlink()

    # Parallel compute with joblib
    print(f"[INFO] Computing descriptors in parallel with {max_workers} workers...")
    results = Parallel(n_jobs=max_workers, backend="loky", return_as="generator")(
        delayed(compute_desc)(args) for args in smiles_list
    )

    # Collect results with progress bar, filtering None
    valid_results = []
    for result in tqdm(results, total=len(smiles_list), desc="Processing SMILES"):
        if result is not None:
            valid_results.append(result)

    print(f"[INFO] Writing {len(valid_results):,} results to CSV...")

    # Create Polars DataFrame for efficient processing
    df = pl.DataFrame(valid_results)

    # Write to CSV (Polars is much faster than pandas for this)
    df.write_csv(out_csv)

    print(f"[OK] Processed {len(valid_results):,} molecules and wrote to {out_csv}")
    return df


def plot_and_save_single_violin(df_single: pd.DataFrame, ds_name: str, column: str,
                                 ylabel: str, title: str, filename: Path):
    """Generates and saves a violin plot for a single dataset."""
    fig, ax = plt.subplots(figsize=(6, 6))

    act = df_single[df_single["bucket"] == "active"][column].values
    inact = df_single[df_single["bucket"] == "inactive"][column].values

    data = []
    labels = []
    if len(act) > 0:
        data.append(act)
        labels.append("Actives")
    if len(inact) > 0:
        data.append(inact)
        labels.append("Inactives")

    if not data:
        plt.close(fig)
        return

    parts = ax.violinplot(data, showmeans=False, showmedians=True)

    for j, pc in enumerate(parts['bodies']):
        color = COLOR_ACTIVE if labels[j] == "Actives" else COLOR_INACTIVE
        pc.set_facecolor(color)
        pc.set_alpha(0.6)

    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels)
    ax.set_title(f"{title}\n({ds_name})")
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    fig.savefig(filename, dpi=300)
    plt.close(fig)


def violin_plot(df: pd.DataFrame, column: str, ylabel: str, title: str, filename: Path):
    """Create violin plots (both combined and individual per dataset)."""
    # Part 1: Loop through datasets to create and save individual plots
    for ds in df["dataset"].unique():
        df_single = df[df["dataset"] == ds]
        p = Path(filename)
        single_filename = p.parent / f"{p.stem}_{ds}{p.suffix}"
        plot_and_save_single_violin(df_single, ds, column, ylabel, title, single_filename)

    # Part 2: Create and save the combined plot
    fig, ax = plt.subplots(figsize=(10, 6))
    data_combined = []
    positions_combined = []
    labels_combined = []
    i = 1

    for ds in sorted(df["dataset"].unique()):
        act = df[(df["dataset"] == ds) & (df["bucket"] == "active")][column].values
        inact = df[(df["dataset"] == ds) & (df["bucket"] == "inactive")][column].values

        if len(act) > 0:
            data_combined.append(act)
            positions_combined.append(i)
            labels_combined.append(f"{ds}\nActives")
            i += 1
        if len(inact) > 0:
            data_combined.append(inact)
            positions_combined.append(i)
            labels_combined.append(f"{ds}\nInactives")
            i += 2

    if not data_combined:
        plt.close(fig)
        return

    parts = ax.violinplot(data_combined, positions=positions_combined, showmeans=False, showmedians=True)

    for j, pc in enumerate(parts['bodies']):
        color = COLOR_ACTIVE if "Actives" in labels_combined[j] else COLOR_INACTIVE
        pc.set_facecolor(color)
        pc.set_alpha(0.6)

    ax.set_xticks(positions_combined)
    ax.set_xticklabels(labels_combined, rotation=45, ha="right")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    fig.savefig(filename, dpi=300)
    plt.close(fig)


def compliance_bar_from_summary(summary_csv: Path, outdir: Path):
    """Create compliance bar charts from summary CSV."""
    # Use Polars for fast loading, convert to pandas for plotting
    df_pl = pl.read_csv(summary_csv)
    df = df_pl.filter(pl.col("Bucket").is_in(["Actives", "Inactives"])).to_pandas()

    comp = df.melt(
        id_vars=["Dataset", "Bucket"],
        value_vars=["LipinskiComplianceRate", "VeberComplianceRate"],
        var_name="rule", value_name="rate"
    )
    comp["rate"] = comp["rate"] * 100  # %

    buckets = ["Actives", "Inactives"]

    for rule, title in [("LipinskiComplianceRate", "Lipinski Compliance Rates"),
                        ("VeberComplianceRate", "Veber Compliance Rates")]:
        fig, ax = plt.subplots(figsize=(10, 6))
        datasets = comp["Dataset"].unique()
        x = range(len(datasets))
        width = 0.35

        for j, bucket in enumerate(buckets):
            sub = comp[(comp["rule"] == rule) & (comp["Bucket"] == bucket)]
            color = COLOR_ACTIVE if bucket == "Actives" else COLOR_INACTIVE
            ax.bar([xi + j * width for xi in x],
                   sub["rate"], width=width, label=bucket, color=color)

        ax.set_xticks([xi + width / 2 for xi in x])
        ax.set_xticklabels(datasets, rotation=45, ha="right")
        ax.set_ylabel("Compliance rate (%)")
        ax.set_title(title)
        ax.set_ylim(0, 100)
        ax.legend()
        fig.tight_layout()
        fname = outdir / f"{rule.replace('ComplianceRate', '').lower()}_compliance.png"
        fig.savefig(fname, dpi=300)
        plt.close(fig)


def scatter_plot_grouped(df: pd.DataFrame, xcol: str, ycol: str,
                          xlabel: str, ylabel: str, title: str, filename: Path):
    """Create grouped scatter plots."""
    datasets = df["dataset"].unique()
    fig, axes = plt.subplots(1, len(datasets), figsize=(6 * len(datasets), 5), sharey=True)

    if len(datasets) == 1:
        axes = [axes]

    for ax, ds in zip(axes, datasets):
        sub = df[df["dataset"] == ds]
        colors = sub["bucket"].map({"active": COLOR_ACTIVE, "inactive": COLOR_INACTIVE})
        ax.scatter(sub[xcol], sub[ycol], c=colors, alpha=0.3, s=10)
        ax.set_title(ds)
        ax.set_xlabel(xlabel)
        if ax is axes[0]:
            ax.set_ylabel(ylabel)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(filename, dpi=300)
    plt.close(fig)


def hist_overlay_grouped(df: pd.DataFrame, column: str, xlabel: str,
                          title: str, filename: Path, bins: int = 50):
    """Create grouped histogram overlays."""
    datasets = df["dataset"].unique()
    fig, axes = plt.subplots(1, len(datasets), figsize=(6 * len(datasets), 5), sharey=True)

    if len(datasets) == 1:
        axes = [axes]

    for ax, ds in zip(axes, datasets):
        sub = df[df["dataset"] == ds]
        ax.hist(sub[sub["bucket"] == "active"][column], bins=bins, alpha=0.6,
                color=COLOR_ACTIVE, label="Actives", density=True)
        ax.hist(sub[sub["bucket"] == "inactive"][column], bins=bins, alpha=0.6,
                color=COLOR_INACTIVE, label="Inactives", density=True)
        ax.set_title(ds)
        ax.set_xlabel(xlabel)
        if ax is axes[0]:
            ax.set_ylabel("Density")
        ax.legend()

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(filename, dpi=300)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Generate molecular descriptor visualizations")
    parser.add_argument("--smiles-dir", type=Path, default=Path("smiles_out"),
                        help="Directory with *_actives.smi and *_inactives.smi")
    parser.add_argument("--outdir", type=Path, default=Path("."),
                        help="Output directory for plots/CSV")
    parser.add_argument("--skip-csv", action="store_true",
                        help="Skip recomputing descriptors, read ligand_descriptors.csv instead")
    parser.add_argument("--summary-csv", type=Path, default=Path("dataset_unique_summary_split.csv"),
                        help="Path to dataset_unique_summary_split.csv for compliance plotting")
    parser.add_argument("--batch-size", type=int, default=1000,
                        help="Batch size for CSV writes (default: 1000)")
    parser.add_argument("--max-workers", type=int, default=-1,
                        help="Max parallel workers (-1 = all CPUs)")
    args = parser.parse_args()

    args.outdir.mkdir(exist_ok=True)
    out_csv = args.outdir / "ligand_descriptors.csv"

    if args.skip_csv:
        print(f"[INFO] Reading existing {out_csv}")
        df_pl = pl.read_csv(out_csv)
    else:
        df_pl = process_smiles_parallel(args.smiles_dir, out_csv, args.batch_size, args.max_workers)

    # Convert to pandas for matplotlib/seaborn visualization
    df = df_pl.to_pandas()

    print("[INFO] Making violin plots (combined and for each dataset)...")
    violin_plot(df, "mw", "Molecular weight (Da)", "Molecular Weight Distribution",
                args.outdir / "violin_mw.png")
    violin_plot(df, "tpsa", "Topological polar surface area (A^2)", "TPSA Distribution",
                args.outdir / "violin_tpsa.png")
    violin_plot(df, "rotbonds", "Rotatable bonds (count)", "Rotatable Bonds Distribution",
                args.outdir / "violin_rotbonds.png")

    print("[INFO] Making compliance bar charts...")
    compliance_bar_from_summary(args.summary_csv, args.outdir)

    print("[INFO] Making extra visualisations...")
    scatter_plot_grouped(df, "mw", "tpsa", "Molecular weight (Da)", "TPSA (A^2)",
                         "MW vs TPSA", args.outdir / "scatter_mw_tpsa.png")
    scatter_plot_grouped(df, "mw", "rotbonds", "Molecular weight (Da)", "Rotatable bonds (count)",
                         "MW vs Rotatable bonds", args.outdir / "scatter_mw_rotbonds.png")

    hist_overlay_grouped(df, "mw", "Molecular weight (Da)", "MW distribution overlay",
                         args.outdir / "hist_mw.png")
    hist_overlay_grouped(df, "tpsa", "TPSA (A^2)", "TPSA distribution overlay",
                         args.outdir / "hist_tpsa.png")
    hist_overlay_grouped(df, "rotbonds", "Rotatable bonds (count)",
                         "Rotatable bonds distribution overlay", args.outdir / "hist_rotbonds.png")

    print("[OK] All plots saved to", args.outdir)


if __name__ == "__main__":
    main()
