import marimo

__generated_with = "0.10.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import polars as pl
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from pathlib import Path
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    import warnings

    warnings.filterwarnings("ignore")

    from metrics import COLOR_ACTIVE, COLOR_INACTIVE, violin_plot, compliance_bar_from_summary

    sns.set_style("whitegrid")
    plt.rcParams["figure.dpi"] = 100
    plt.rcParams["savefig.dpi"] = 300
    plt.rcParams["figure.figsize"] = (10, 6)

    RESULTS_DIR = Path("results")
    OUTPUT_DIR = Path("notebook_figures")
    OUTPUT_DIR.mkdir(exist_ok=True)

    DATASET_COLORS = {
        "LIT-PCBA": "#1f77b4",
        "DUDE-Z": "#ff7f0e",
        "DEKOIS2": "#2ca02c",
        "MUV": "#d62728",
        "D-COID": "#9467bd",
    }
    return (
        COLOR_ACTIVE,
        COLOR_INACTIVE,
        DATASET_COLORS,
        OUTPUT_DIR,
        PCA,
        RESULTS_DIR,
        StandardScaler,
        compliance_bar_from_summary,
        mo,
        np,
        pl,
        plt,
        sns,
        violin_plot,
        warnings,
    )


@app.cell
def _(mo):
    return mo.md("""
    # DecoyComp: Dataset Comparison Executive Summary

    **Purpose**: Quick comparison of molecular benchmark datasets for virtual screening

    **Datasets**: LIT-PCBA, DUDE-Z, DEKOIS2, MUV, D-COID

    ---

    Before running, generate the dataset summaries:

    ```bash
    analyse-datasets --roots LIT-PCBA DUDE-Z DEKOIS2 D-COID MUV --outdir results
    decoycomp-metrics --smiles-dir results/smiles --outdir results \\
        --summary-parquet results/dataset_unique_summary_split.parquet
    ```
    """),


@app.cell
def _(RESULTS_DIR, mo, pl):
    summary_file = RESULTS_DIR / "dataset_unique_summary.parquet"
    split_summary_file = RESULTS_DIR / "dataset_unique_summary_split.parquet"

    mo.stop(
        not summary_file.exists(),
        mo.callout(
            mo.md(
                f"**Summary file not found**: `{summary_file}`\n\n"
                "Run `analyse-datasets --roots LIT-PCBA DUDE-Z DEKOIS2 D-COID MUV --outdir results` first."
            ),
            kind="warn",
        ),
    )

    df_summary_pl = pl.read_parquet(summary_file)
    df_split_pl = pl.read_parquet(split_summary_file) if split_summary_file.exists() else None
    return df_split_pl, df_summary_pl, split_summary_file, summary_file


@app.cell
def _(mo):
    return mo.md("## 1. Dataset Comparison Table"),


@app.cell
def _(df_summary_pl, pl):
    def _build_comparison(df: "pl.DataFrame") -> "pl.DataFrame":
        rows = []
        for dataset in df.get_column("Dataset").unique().to_list():
            row = df.filter(pl.col("Dataset") == dataset).row(0, named=True)
            n_actives = int(row.get("NumberActivesUnique", 0))
            n_inactives = int(row.get("NumberInactivesUnique", 0))
            total = row["NumberLigandsUnique"]
            invalid_pct = (row.get("NumberInvalidSMILES", 0) / total * 100) if total > 0 else 0
            rows.append({
                "Dataset": dataset,
                "Total Molecules": int(total),
                "Actives": n_actives,
                "Inactives": n_inactives,
                "Active:Inactive Ratio": f"1:{n_inactives/n_actives:.1f}" if n_actives > 0 else "N/A",
                "Quality Score": f"{100 - invalid_pct:.1f}%",
                "Lipinski %": f"{row.get('LipinskiComplianceRate', 0) * 100:.1f}%",
                "Veber %": f"{row.get('VeberComplianceRate', 0) * 100:.1f}%",
            })
        return pl.DataFrame(rows).sort("Dataset")

    comparison_pl = _build_comparison(df_summary_pl)
    return comparison_pl,


@app.cell
def _(comparison_pl):
    return comparison_pl,


@app.cell
def _(mo):
    return mo.md("## 2. Dataset Sizes"),


@app.cell
def _(COLOR_ACTIVE, COLOR_INACTIVE, DATASET_COLORS, OUTPUT_DIR, df_summary_pl, np, plt):
    _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(14, 5))

    _datasets = df_summary_pl.get_column("Dataset").to_list()
    _total_mols = df_summary_pl.get_column("NumberLigandsUnique").to_list()
    _colors = [DATASET_COLORS.get(ds, "gray") for ds in _datasets]

    _ax1.bar(_datasets, _total_mols, color=_colors, alpha=0.7)
    _ax1.set_ylabel("Number of Unique Molecules")
    _ax1.set_title("Dataset Sizes", fontweight="bold", fontsize=12)
    _ax1.tick_params(axis="x", rotation=45)
    for _i, (_ds, _val) in enumerate(zip(_datasets, _total_mols)):
        _ax1.text(_i, _val, f"{int(_val):,}", ha="center", va="bottom", fontsize=9)

    _actives = df_summary_pl.get_column("NumberActivesUnique").to_list()
    _inactives = df_summary_pl.get_column("NumberInactivesUnique").to_list()
    _x = np.arange(len(_datasets))
    _w = 0.35

    _ax2.bar(_x - _w / 2, _actives, _w, label="Actives", color=COLOR_ACTIVE, alpha=0.7)
    _ax2.bar(_x + _w / 2, _inactives, _w, label="Inactives", color=COLOR_INACTIVE, alpha=0.7)
    _ax2.set_ylabel("Number of Molecules")
    _ax2.set_title("Active vs Inactive Distribution", fontweight="bold", fontsize=12)
    _ax2.set_xticks(_x)
    _ax2.set_xticklabels(_datasets, rotation=45, ha="right")
    _ax2.legend()

    _fig.tight_layout()
    _fig.savefig(OUTPUT_DIR / "dataset_sizes.png", dpi=300, bbox_inches="tight")
    return _fig,


@app.cell
def _(mo):
    return mo.md("## 3. Molecular Property Distributions"),


@app.cell
def _(RESULTS_DIR, mo, pl):
    _desc_file = RESULTS_DIR / "ligand_descriptors.parquet"
    if _desc_file.exists():
        df_desc_pl = pl.read_parquet(_desc_file)
        _status = mo.md(f"Loaded **{len(df_desc_pl):,}** molecular descriptors from `{_desc_file}`.")
    else:
        df_desc_pl = None
        _status = mo.callout(
            mo.md(
                f"Descriptor file not found: `{_desc_file}`\n\n"
                "Run `decoycomp-metrics --smiles-dir results/smiles --outdir results` to generate it."
            ),
            kind="warn",
        )
    return df_desc_pl, _status


@app.cell
def _(df_desc_pl, mo, _status):
    return mo.vstack([_status]) if df_desc_pl is None else _status,


@app.cell
def _(OUTPUT_DIR, df_desc_pl, mo, violin_plot):
    if df_desc_pl is not None:
        violin_plot(df_desc_pl, "mw", "Molecular Weight (Da)",
                    "Molecular Weight Distribution", OUTPUT_DIR / "violin_mw.png")
        violin_plot(df_desc_pl, "tpsa", "TPSA (Å²)",
                    "Topological Polar Surface Area", OUTPUT_DIR / "violin_tpsa.png")
        violin_plot(df_desc_pl, "rotbonds", "Rotatable Bonds",
                    "Rotatable Bonds Distribution", OUTPUT_DIR / "violin_rotbonds.png")
        _out = mo.md("Violin plots saved to `notebook_figures/`.")
    else:
        _out = mo.md("Skipping violin plots — descriptor file not available.")
    return _out,


@app.cell
def _(mo):
    return mo.md("## 4. Drug-likeness Compliance"),


@app.cell
def _(OUTPUT_DIR, df_summary_pl, np, plt):
    _fig, _ax = plt.subplots(figsize=(10, 6))

    _datasets = df_summary_pl.get_column("Dataset").to_list()
    _lip = [r * 100 for r in df_summary_pl.get_column("LipinskiComplianceRate").to_list()]
    _veb = [r * 100 for r in df_summary_pl.get_column("VeberComplianceRate").to_list()]
    _x = np.arange(len(_datasets))
    _w = 0.35

    _b1 = _ax.bar(_x - _w / 2, _lip, _w, label="Lipinski's Rule of Five", color="#3498db", alpha=0.8)
    _b2 = _ax.bar(_x + _w / 2, _veb, _w, label="Veber's Rule", color="#e74c3c", alpha=0.8)

    for _bars in [_b1, _b2]:
        for _bar in _bars:
            _h = _bar.get_height()
            _ax.text(_bar.get_x() + _bar.get_width() / 2, _h,
                     f"{_h:.1f}%", ha="center", va="bottom", fontsize=9)

    _ax.set_ylabel("Compliance Rate (%)", fontsize=11)
    _ax.set_title("Drug-likeness Rule Compliance", fontweight="bold", fontsize=13)
    _ax.set_xticks(_x)
    _ax.set_xticklabels(_datasets, rotation=45, ha="right")
    _ax.set_ylim(0, 105)
    _ax.legend(fontsize=10)
    _ax.grid(axis="y", alpha=0.3)

    _fig.tight_layout()
    _fig.savefig(OUTPUT_DIR / "druglikeness_compliance.png", dpi=300, bbox_inches="tight")
    return _fig,


@app.cell
def _(mo):
    return mo.md("## 5. Data Quality Scorecard"),


@app.cell
def _(OUTPUT_DIR, df_summary_pl, pl, plt, sns):
    import pandas as pd  # local import: seaborn heatmap requires a pandas DataFrame

    _quality_data = []
    for _ds in df_summary_pl.get_column("Dataset").to_list():
        _row = df_summary_pl.filter(pl.col("Dataset") == _ds).row(0, named=True)
        _total = _row["NumberLigandsUnique"]
        _invalid_pct = (_row.get("NumberInvalidSMILES", 0) / _total * 100) if _total > 0 else 0
        _quality_data.append({
            "Dataset": _ds,
            "Valid SMILES": 100 - _invalid_pct,
            "Lipinski Compliant": _row.get("LipinskiComplianceRate", 0) * 100,
            "Veber Compliant": _row.get("VeberComplianceRate", 0) * 100,
        })

    _heatmap_data = pd.DataFrame(_quality_data).set_index("Dataset").T

    _fig, _ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(_heatmap_data, annot=True, fmt=".1f", cmap="RdYlGn",
                vmin=0, vmax=100, cbar_kws={"label": "Percentage (%)"},
                linewidths=0.5, ax=_ax)
    _ax.set_title("Data Quality Scorecard", fontweight="bold", fontsize=13, pad=15)
    _fig.tight_layout()
    _fig.savefig(OUTPUT_DIR / "quality_scorecard.png", dpi=300, bbox_inches="tight")
    return _fig,


@app.cell
def _(mo):
    return mo.md("## 6. Chemical Space (PCA)"),


@app.cell
def _(DATASET_COLORS, OUTPUT_DIR, PCA, StandardScaler, df_desc_pl, mo, np, plt):
    if df_desc_pl is not None:
        _max_per_ds = 5000
        _sampled = (
            df_desc_pl
            .group_by("dataset")
            .map_groups(lambda grp: grp.sample(n=min(len(grp), _max_per_ds), seed=42))
        )

        _X = _sampled.select(["mw", "tpsa", "rotbonds"]).to_numpy()
        _labels = _sampled.get_column("dataset").to_list()

        _X_scaled = StandardScaler().fit_transform(_X)
        _pca = PCA(n_components=2)
        _X_pca = _pca.fit_transform(_X_scaled)

        _fig, _ax = plt.subplots(figsize=(10, 8))
        for _ds in _sampled.get_column("dataset").unique().to_list():
            _mask = np.array([d == _ds for d in _labels])
            _ax.scatter(_X_pca[_mask, 0], _X_pca[_mask, 1],
                        label=_ds, alpha=0.4, s=10, color=DATASET_COLORS.get(_ds, "gray"))

        _ax.set_xlabel(f"PC1 ({_pca.explained_variance_ratio_[0]*100:.1f}% variance)", fontsize=11)
        _ax.set_ylabel(f"PC2 ({_pca.explained_variance_ratio_[1]*100:.1f}% variance)", fontsize=11)
        _ax.set_title("Chemical Space Distribution (PCA)", fontweight="bold", fontsize=13)
        _ax.legend(loc="best", framealpha=0.9)
        _ax.grid(alpha=0.3)
        _fig.tight_layout()
        _fig.savefig(OUTPUT_DIR / "chemical_space_pca.png", dpi=300, bbox_inches="tight")
        _result = _fig
    else:
        _result = mo.md("Skipping PCA — descriptor file not available.")
    return _result,


@app.cell
def _(mo):
    return mo.md("## 7. Key Findings"),


@app.cell
def _(df_summary_pl, mo, pl):
    def _findings(df: "pl.DataFrame") -> list[str]:
        out = []
        _largest = df.sort("NumberLigandsUnique", descending=True).row(0, named=True)
        _smallest = df.sort("NumberLigandsUnique", descending=False).row(0, named=True)
        out.append(
            f"**Dataset Size**: {_largest['Dataset']} is the largest "
            f"({int(_largest['NumberLigandsUnique']):,} molecules); "
            f"{_smallest['Dataset']} is the smallest ({int(_smallest['NumberLigandsUnique']):,})."
        )
        _best_lip = df.sort("LipinskiComplianceRate", descending=True).row(0, named=True)
        out.append(
            f"**Drug-likeness**: {_best_lip['Dataset']} has the highest Lipinski compliance "
            f"({_best_lip['LipinskiComplianceRate']*100:.1f}%)."
        )
        _clean = (
            df.with_columns(
                (pl.col("NumberInvalidSMILES") / pl.col("NumberLigandsUnique")).alias("inv_rate")
            )
            .sort("inv_rate")
            .row(0, named=True)
        )
        out.append(
            f"**Data Quality**: {_clean['Dataset']} has the fewest invalid SMILES "
            f"({_clean['inv_rate']*100:.3f}%)."
        )
        _ratios = [
            (r["Dataset"], r["NumberInactivesUnique"] / r["NumberActivesUnique"])
            for r in df.iter_rows(named=True)
            if r["NumberActivesUnique"] > 0
        ]
        if _ratios:
            _balanced = min(_ratios, key=lambda t: abs(t[1] - 1))
            out.append(
                f"**Balance**: {_balanced[0]} has the most balanced active:inactive ratio "
                f"(1:{_balanced[1]:.1f})."
            )
        return out

    _items = "\n".join(f"- {f}" for f in _findings(df_summary_pl))
    return mo.md(_items),


@app.cell
def _(df_summary_pl, mo):
    _best_lip = df_summary_pl.sort("LipinskiComplianceRate", descending=True).row(0, named=True)["Dataset"]
    return mo.md(f"""
    ## 8. Use Case Recommendations

    | Use Case | Recommendation |
    |----------|---------------|
    | Method Development | Dataset with highest data quality score |
    | Realistic Benchmarking | LIT-PCBA (literature-derived actives/inactives) |
    | Challenging Evaluation | DUDE-Z (property-matched decoys) |
    | Drug-like Screening | {_best_lip} (highest Lipinski compliance) |
    | Structure-based Studies | D-COID (PDB-derived structures) |
    | Bioactivity Prediction | MUV (experimental bioactivity data) |
    """),


@app.cell
def _(mo):
    return mo.md("## 9. Descriptor Statistics"),


@app.cell
def _(df_desc_pl, mo, pl):
    if df_desc_pl is not None:
        _stats = (
            df_desc_pl.group_by("dataset").agg([
                pl.col("mw").mean().round(2).alias("MW mean"),
                pl.col("mw").std().round(2).alias("MW std"),
                pl.col("mw").median().round(2).alias("MW median"),
                pl.col("tpsa").mean().round(2).alias("TPSA mean"),
                pl.col("tpsa").std().round(2).alias("TPSA std"),
                pl.col("tpsa").median().round(2).alias("TPSA median"),
                pl.col("rotbonds").mean().round(2).alias("RotBonds mean"),
                pl.col("rotbonds").std().round(2).alias("RotBonds std"),
                pl.col("rotbonds").median().round(2).alias("RotBonds median"),
            ])
            .sort("dataset")
            .rename({"dataset": "Dataset"})
        )
        _result = _stats
    else:
        _result = mo.md("Descriptor statistics not available — run `decoycomp-metrics` first.")
    return _result,


if __name__ == "__main__":
    app.run()
