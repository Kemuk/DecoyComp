#!/usr/bin/env python3
"""
Molecular utilities for descriptor calculation and aggregation.

Uses joblib for persistent caching and parallel processing.
"""
from collections import namedtuple
from typing import Callable

from joblib import Parallel, delayed
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, rdMolDescriptors, FilterCatalog
from tqdm.auto import tqdm

from cache_manager import get_descriptor_memory

# Constants - atomic numbers of metals
METALS = {
    3, 4, 11, 12, 13, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
    37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
    55, 56, 57, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83
}

MolDescriptors = namedtuple(
    "MolDescriptors",
    ["mw", "clogp", "tpsa", "fsp3", "rb", "lip_pass", "veber_pass", "has_metal", "pains_hit"]
)


class DescriptorCalculator:
    """Calculates molecular descriptors using RDKit with joblib caching."""

    _PAINS_CATALOG = None

    @classmethod
    def get_pains_catalog(cls):
        """Lazy initialisation of PAINS catalog."""
        if cls._PAINS_CATALOG is None:
            print("[INFO] Initialising PAINS catalog...")
            params = FilterCatalog.FilterCatalogParams()
            for catalog in (
                FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS_A,
                FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS_B,
                FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS_C
            ):
                params.AddCatalog(catalog)
            cls._PAINS_CATALOG = FilterCatalog.FilterCatalog(params)
            print("[INFO] PAINS catalog initialised")
        return cls._PAINS_CATALOG

    @classmethod
    def calculate(cls, mol) -> MolDescriptors:
        """Calculate descriptors for a single RDKit molecule object."""
        mw = Descriptors.MolWt(mol)
        clogp = Crippen.MolLogP(mol)
        tpsa = rdMolDescriptors.CalcTPSA(mol)
        fsp3 = rdMolDescriptors.CalcFractionCSP3(mol)
        rb = int(Descriptors.NumRotatableBonds(mol))
        lip_pass = (
            mw <= 500 and clogp <= 5 and
            Descriptors.NumHDonors(mol) <= 5 and
            Descriptors.NumHAcceptors(mol) <= 10
        )
        veber_pass = rb <= 10 and tpsa <= 140.0
        has_metal = any(atom.GetAtomicNum() in METALS for atom in mol.GetAtoms())
        pains_hit = cls.get_pains_catalog().HasMatch(mol)
        return MolDescriptors(mw, clogp, tpsa, fsp3, rb, lip_pass, veber_pass, has_metal, pains_hit)

    @staticmethod
    def calculate_one_smiles(smi: str) -> tuple[str, bool, MolDescriptors | None]:
        """
        Calculate descriptors for a single SMILES string.

        Returns:
            Tuple of (smiles, has_salts, MolDescriptors or None if invalid)
        """
        salts = "." in smi
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return smi, salts, None
        desc = DescriptorCalculator.calculate(mol)
        return smi, salts, desc

    @classmethod
    def calculate_all_parallel(
        cls,
        smiles_list: list[str],
        workers: int = -1,
        use_cache: bool = True,
        show_progress: bool = True
    ) -> dict[str, tuple[bool, MolDescriptors | None]]:
        """
        Calculate descriptors for all SMILES in parallel with optional caching.

        Args:
            smiles_list: List of SMILES strings to process
            workers: Number of parallel workers (-1 = all CPUs)
            use_cache: Whether to use joblib persistent cache
            show_progress: Whether to show progress bar

        Returns:
            Dictionary mapping SMILES to (has_salts, MolDescriptors or None)
        """
        n_smiles = len(smiles_list)
        print(f"[INFO] Calculating descriptors for {n_smiles:,} unique SMILES with {workers} workers...")

        # Get the cached version of the calculation function
        if use_cache:
            memory = get_descriptor_memory()
            cached_calculate = memory.cache(cls.calculate_one_smiles)
            calc_func = cached_calculate
        else:
            calc_func = cls.calculate_one_smiles

        # Process in parallel with joblib
        if show_progress:
            results = Parallel(n_jobs=workers, backend="loky", return_as="generator")(
                delayed(calc_func)(smi) for smi in smiles_list
            )
            # Wrap with tqdm for progress
            results = list(tqdm(results, total=n_smiles, desc="Computing descriptors"))
        else:
            results = Parallel(n_jobs=workers, backend="loky")(
                delayed(calc_func)(smi) for smi in smiles_list
            )

        # Build cache dictionary
        cache = {}
        invalid_count = 0
        for smi, salts, desc in results:
            if desc is None:
                invalid_count += 1
            cache[smi] = (salts, desc)

        valid_count = n_smiles - invalid_count
        print(f"[INFO] Computed {valid_count:,} valid molecules ({invalid_count:,} invalid)")

        if use_cache:
            from cache_manager import get_cache_info
            info = get_cache_info()
            print(f"[INFO] Cache location: {info['path']} ({info['size_mb']:.1f} MB)")

        return cache


def aggregate_from_cache(smiles_list: list[str], cache: dict) -> dict:
    """
    Aggregate statistics from cached descriptors.

    Args:
        smiles_list: List of SMILES to aggregate
        cache: Dictionary mapping SMILES to (has_salts, MolDescriptors)

    Returns:
        Dictionary with aggregated statistics
    """
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
