# DecoyComp

Automated comparison and analysis of molecular benchmark datasets for virtual screening.

Analyses five major datasets — **LIT-PCBA**, **DUDE-Z**, **DEKOIS2**, **D-COID**, and **MUV** — computing molecular descriptors (MW, cLogP, TPSA, Fsp3, rotatable bonds), drug-likeness metrics (Lipinski, Veber), and data quality statistics. Results are written as Parquet files for fast downstream use.

---

## Prerequisites

- Python ≥ 3.10
- [RDKit](https://www.rdkit.org/) ≥ 2023.9
- [BioPython](https://biopython.org/) ≥ 1.80 (D-COID PDB parsing)
- [DeepChem](https://deepchem.io/) ≥ 2.7 — optional, needed only for MUV

---

## Installation

```bash
# Core (LIT-PCBA, DUDE-Z, DEKOIS2, D-COID)
pip install -e .

# Include MUV support
pip install -e ".[muv]"

# Include Marimo notebook extras (seaborn heatmap)
pip install -e ".[notebook]"
```

---

## Dataset Acquisition

Download each dataset and place it in a directory matching its name:

| Dataset | Source | Expected layout |
|---------|--------|----------------|
| LIT-PCBA | https://drugdesign.unistra.fr/LIT-PCBA/ | `LIT-PCBA/<target>/actives.smi`, `inactives.smi` |
| DUDE-Z | https://dude.docking.org/ | `DUDE-Z/<target>/actives_final.ism`, `decoys_final.ism` |
| DEKOIS2 | http://www.dekois.com/ | `DEKOIS2/<target>/active_decoys.smi` |
| D-COID | https://github.com/rinikerlab/D-COID | `D-COID/<target>/*.pdb` |
| MUV | Loaded automatically via DeepChem | — |

---

## Running the Analysis Pipeline

```bash
# Analyse file-based datasets
analyse-datasets --roots LIT-PCBA DUDE-Z DEKOIS2 D-COID --outdir results

# Include MUV (requires DeepChem)
analyse-datasets --roots LIT-PCBA DUDE-Z DEKOIS2 D-COID MUV --outdir results

# MUV only
analyse-datasets --roots MUV --outdir results

# Disable caching (force recompute descriptors)
analyse-datasets --roots LIT-PCBA DUDE-Z --outdir results --no-cache

# Write SMILES files only
analyse-datasets --roots LIT-PCBA DUDE-Z --write-smiles-only
```

### Output files (`results/`)

| File | Contents |
|------|----------|
| `dataset_summary.parquet` | Per-target statistics for all datasets |
| `per_target_summary.parquet` | LIT-PCBA per-target breakdown (when present) |
| `dataset_unique_summary.parquet` | Dataset-level aggregated statistics |
| `dataset_unique_summary_split.parquet` | Split by actives / inactives / all |

All Parquet files can be opened with:
```python
import polars as pl
df = pl.read_parquet("results/dataset_unique_summary.parquet")
```

---

## Generating Visualizations

```bash
# Compute per-molecule descriptors and generate plots
decoycomp-metrics \
    --smiles-dir results/smiles \
    --outdir results \
    --summary-parquet results/dataset_unique_summary_split.parquet

# Skip recomputing descriptors (use existing parquet)
decoycomp-metrics \
    --smiles-dir results/smiles \
    --outdir results \
    --summary-parquet results/dataset_unique_summary_split.parquet \
    --skip-descriptors
```

---

## Running the Marimo Notebook

The interactive executive summary is a [Marimo](https://marimo.io/) notebook. Run the
analysis pipeline first to generate the required Parquet files, then:

```bash
# Interactive reactive notebook
marimo edit dataset_comparison_executive_summary.py

# Run as a script (non-interactive)
python dataset_comparison_executive_summary.py
```

The notebook loads `results/dataset_unique_summary.parquet` and
`results/ligand_descriptors.parquet` and produces figures in `notebook_figures/`.

---

## Running Tests

```bash
pip install -e ".[test]"
pytest tests/
```

---

## Caching

Molecular descriptor calculations are cached with [joblib](https://joblib.readthedocs.io/)
in `.cache/descriptors/` to avoid redundant computation across runs.

```python
from cache_manager import clear_descriptor_cache, get_cache_info
print(get_cache_info())   # show cache size
clear_descriptor_cache()  # wipe cache
```

Dataset-level metadata (SMILES counts, validity) is cached as Parquet files alongside
each dataset directory.
