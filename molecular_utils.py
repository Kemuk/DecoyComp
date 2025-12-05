#!/usr/bin/env python3
"""
Molecular utilities for descriptor calculation and aggregation
"""
from collections import namedtuple

from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, rdMolDescriptors, FilterCatalog
from tqdm.contrib.concurrent import process_map

# Constants
METALS = {3,4,11,12,13,19,20,21,22,23,24,25,26,27,28,29,30,31,37,38,39,40,41,42,43,44,45,46,47,48,49,50,55,56,57,72,73,74,75,76,77,78,79,80,81,82,83}

MolDescriptors = namedtuple("MolDescriptors", ["mw", "clogp", "tpsa", "fsp3", "rb", "lip_pass", "veber_pass", "has_metal", "pains_hit"])


class DescriptorCalculator:
    """Calculates molecular descriptors using RDKit"""
    
    _PAINS_CATALOG = None
    
    @classmethod
    def get_pains_catalog(cls):
        """Lazy initialisation of PAINS catalog"""
        if cls._PAINS_CATALOG is None:
            print("[INFO] Initialising PAINS catalog...")
            params = FilterCatalog.FilterCatalogParams()
            for catalog in (FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS_A,
                            FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS_B,
                            FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS_C):
                params.AddCatalog(catalog)
            cls._PAINS_CATALOG = FilterCatalog.FilterCatalog(params)
            print("[INFO] PAINS catalog initialised")
        return cls._PAINS_CATALOG
    
    @classmethod
    def calculate(cls, mol) -> MolDescriptors:
        """Calculate descriptors for a single molecule"""
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
        """Worker function for parallel processing. Returns (smiles, has_salts, MolDescriptors or None)"""
        salts = "." in smi
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return smi, salts, None
        desc = DescriptorCalculator.calculate(mol)
        return smi, salts, desc
    
    @staticmethod
    def calculate_all_parallel(smiles_list: list[str], workers: int) -> dict[str, tuple[bool, MolDescriptors]]:
        """Calculate descriptors for all SMILES in parallel. Returns cache dict: {smiles: (has_salts, desc)}"""
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