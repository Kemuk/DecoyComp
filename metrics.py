#!/usr/bin/env python3
"""
Visualization and descriptor analysis using Polars and joblib.

Uses Polars for all data manipulation; numpy arrays are passed directly
to matplotlib — no pandas dependency.
"""
import argparse
from pathlib import Path

import polars as pl
import numpy as np
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
    out_parquet: Path,
    max_workers: int = -1
) -> pl.DataFrame:
    """Process SMILES in parallel and write results as Parquet using joblib."""

    smiles_list = read_smiles(smiles_dir)

    print(f"[INFO] Computing descriptors in parallel with {max_workers} workers...")
    results = Parallel(n_jobs=max_workers, backend="loky", return_as="generator")(
        delayed(compute_desc)(args) for args in smiles_list
    )

    valid_results = []
    for result in tqdm(results, total=len(smiles_list), desc="Processing SMILES"):
        if result is not None:
            valid_results.append(result)

    print(f"[INFO] Writing {len(valid_results):,} results to {out_parquet}...")
    df = pl.DataFrame(valid_results)
    df.write_parquet(out_parquet)

    print(f"[OK] Processed {len(valid_results):,} molecules")
    return df


def plot_and_save_single_violin(df_single: pl.DataFrame, ds_name: str, column: str,
                                 ylabel: str, title: str, filename: Path):
    """Generates and saves a violin plot for a single dataset."""
    fig, ax = plt.subplots(figsize=(6, 6))

    act = df_single.filter(pl.col("bucket") == "active").get_column(column).to_numpy()
    inact = df_single.filter(pl.col("bucket") == "inactive").get_column(column).to_numpy()

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


def violin_plot(df: pl.DataFrame, column: str, ylabel: str, title: str, filename: Path):
    """Create violin plots (both combined and individual per dataset)."""
    datasets = df.get_column("dataset").unique().to_list()

    for ds in datasets:
        df_single = df.filter(pl.col("dataset") == ds)
        p = Path(filename)
        single_filename = p.parent / f"{p.stem}_{ds}{p.suffix}"
        plot_and_save_single_violin(df_single, ds, column, ylabel, title, single_filename)

    # Combined plot across all datasets
    fig, ax = plt.subplots(figsize=(10, 6))
    data_combined = []
    positions_combined = []
    labels_combined = []
    i = 1

    for ds in sorted(datasets):
        act = df.filter((pl.col("dataset") == ds) & (pl.col("bucket") == "active")).get_column(column).to_numpy()
        inact = df.filter((pl.col("dataset") == ds) & (pl.col("bucket") == "inactive")).get_column(column).to_numpy()

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


def compliance_bar_from_summary(summary_parquet: Path, outdir: Path):
    """Create compliance bar charts from split summary Parquet."""
    df = pl.read_parquet(summary_parquet).filter(pl.col("Bucket").is_in(["Actives", "Inactives"]))

    comp = df.unpivot(
        on=["LipinskiComplianceRate", "VeberComplianceRate"],
        index=["Dataset", "Bucket"],
        variable_name="rule",
        value_name="rate",
    ).with_columns((pl.col("rate") * 100).alias("rate"))

    buckets = ["Actives", "Inactives"]
    datasets = sorted(comp.get_column("Dataset").unique().to_list())
    x = np.arange(len(datasets))
    width = 0.35

    for rule, title in [("LipinskiComplianceRate", "Lipinski Compliance Rates"),
                        ("VeberComplianceRate", "Veber Compliance Rates")]:
        fig, ax = plt.subplots(figsize=(10, 6))

        for j, bucket in enumerate(buckets):
            sub = comp.filter((pl.col("rule") == rule) & (pl.col("Bucket") == bucket))
            rates = [
                sub.filter(pl.col("Dataset") == ds).get_column("rate").to_list()[0]
                if sub.filter(pl.col("Dataset") == ds).height > 0 else 0.0
                for ds in datasets
            ]
            color = COLOR_ACTIVE if bucket == "Actives" else COLOR_INACTIVE
            ax.bar(x + j * width, rates, width=width, label=bucket, color=color)

        ax.set_xticks(x + width / 2)
        ax.set_xticklabels(datasets, rotation=45, ha="right")
        ax.set_ylabel("Compliance rate (%)")
        ax.set_title(title)
        ax.set_ylim(0, 100)
        ax.legend()
        fig.tight_layout()
        fname = outdir / f"{rule.replace('ComplianceRate', '').lower()}_compliance.png"
        fig.savefig(fname, dpi=300)
        plt.close(fig)


def scatter_plot_grouped(df: pl.DataFrame, xcol: str, ycol: str,
                          xlabel: str, ylabel: str, title: str, filename: Path):
    """Create grouped scatter plots."""
    datasets = df.get_column("dataset").unique().to_list()
    fig, axes = plt.subplots(1, len(datasets), figsize=(6 * len(datasets), 5), sharey=True)

    if len(datasets) == 1:
        axes = [axes]

    for ax, ds in zip(axes, datasets):
        sub = df.filter(pl.col("dataset") == ds)
        colors = sub.get_column("bucket").replace({"active": COLOR_ACTIVE, "inactive": COLOR_INACTIVE}).to_list()
        ax.scatter(sub.get_column(xcol).to_numpy(), sub.get_column(ycol).to_numpy(),
                   c=colors, alpha=0.3, s=10)
        ax.set_title(ds)
        ax.set_xlabel(xlabel)
        if ax is axes[0]:
            ax.set_ylabel(ylabel)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(filename, dpi=300)
    plt.close(fig)


def hist_overlay_grouped(df: pl.DataFrame, column: str, xlabel: str,
                          title: str, filename: Path, bins: int = 50):
    """Create grouped histogram overlays."""
    datasets = df.get_column("dataset").unique().to_list()
    fig, axes = plt.subplots(1, len(datasets), figsize=(6 * len(datasets), 5), sharey=True)

    if len(datasets) == 1:
        axes = [axes]

    for ax, ds in zip(axes, datasets):
        sub = df.filter(pl.col("dataset") == ds)
        ax.hist(sub.filter(pl.col("bucket") == "active").get_column(column).to_numpy(),
                bins=bins, alpha=0.6, color=COLOR_ACTIVE, label="Actives", density=True)
        ax.hist(sub.filter(pl.col("bucket") == "inactive").get_column(column).to_numpy(),
                bins=bins, alpha=0.6, color=COLOR_INACTIVE, label="Inactives", density=True)
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
    parser.add_argument("--smiles-dir", type=Path, default=Path("smiles"),
                        help="Directory with *_actives.smi and *_inactives.smi")
    parser.add_argument("--outdir", type=Path, default=Path("."),
                        help="Output directory for plots/Parquet")
    parser.add_argument("--skip-descriptors", action="store_true",
                        help="Skip recomputing descriptors, read ligand_descriptors.parquet instead")
    parser.add_argument("--summary-parquet", type=Path,
                        default=Path("dataset_unique_summary_split.parquet"),
                        help="Path to dataset_unique_summary_split.parquet for compliance plotting")
    parser.add_argument("--max-workers", type=int, default=-1,
                        help="Max parallel workers (-1 = all CPUs)")
    args = parser.parse_args()

    args.outdir.mkdir(exist_ok=True)
    out_parquet = args.outdir / "ligand_descriptors.parquet"

    if args.skip_descriptors:
        print(f"[INFO] Reading existing {out_parquet}")
        df = pl.read_parquet(out_parquet)
    else:
        df = process_smiles_parallel(args.smiles_dir, out_parquet, args.max_workers)

    print("[INFO] Making violin plots (combined and for each dataset)...")
    violin_plot(df, "mw", "Molecular weight (Da)", "Molecular Weight Distribution",
                args.outdir / "violin_mw.png")
    violin_plot(df, "tpsa", "Topological polar surface area (A^2)", "TPSA Distribution",
                args.outdir / "violin_tpsa.png")
    violin_plot(df, "rotbonds", "Rotatable bonds (count)", "Rotatable Bonds Distribution",
                args.outdir / "violin_rotbonds.png")

    print("[INFO] Making compliance bar charts...")
    compliance_bar_from_summary(args.summary_parquet, args.outdir)

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
