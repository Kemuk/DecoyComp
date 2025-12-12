#!/usr/bin/env python3
"""
Dataset abstractions for different benchmark datasets
"""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterator
from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from Bio.PDB import PDBParser
from rdkit import Chem


def read_smi_file(filepath: Path, label: str):
    """Helper function to read SMILES files"""
    if not filepath.is_file():
        return
    with open(filepath, "r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            parts = line.strip().split()
            if parts:
                yield parts[0], label


class BaseDataset(ABC):
    """Abstract base class for all datasets"""
    
    def __init__(self, name: str, root_path: Path):
        self.name = name
        self.root_path = root_path
    
    @abstractmethod
    def enumerate_targets(self) -> Iterator[tuple[str, Path]]:
        """Enumerate all targets in the dataset. Yields (target_name, target_path)"""
        pass
    
    @abstractmethod
    def read_target(self, target_path: Path) -> Iterator[tuple[str, str]]:
        """Read SMILES from a target. Yields (smiles, label)"""
        pass


class FileBasedDataset(BaseDataset):
    """Base class for file-based datasets"""
    pass


class LitPCBADataset(FileBasedDataset):
    """LIT-PCBA dataset"""
    
    def enumerate_targets(self):
        if not self.root_path.is_dir():
            return
        for entry in sorted(self.root_path.iterdir()):
            if entry.is_dir():
                yield entry.name, entry
    
    def read_target(self, target_path: Path):
        yield from read_smi_file(target_path / "actives.smi", "active")
        yield from read_smi_file(target_path / "inactives.smi", "inactive")


class DudeZDataset(FileBasedDataset):
    """DUDE-Z dataset"""
    
    def enumerate_targets(self):
        if not self.root_path.is_dir():
            return
        for entry in sorted(self.root_path.iterdir()):
            if entry.is_dir():
                yield entry.name, entry
    
    def read_target(self, target_path: Path):
        yield from read_smi_file(target_path / "actives_final.ism", "active")
        yield from read_smi_file(target_path / "decoys_final.ism", "decoy")


class Dekois2Dataset(FileBasedDataset):
    """DEKOIS2 dataset"""
    
    def enumerate_targets(self):
        if not self.root_path.is_dir():
            return
        for entry in sorted(self.root_path.iterdir()):
            if entry.is_dir():
                yield entry.name, entry
    
    def read_target(self, target_path: Path):
        smi_path = target_path / "active_decoys.smi"
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


class APIBasedDataset(BaseDataset):
    """Base class for API-based datasets (e.g., DeepChem)"""
    pass


class MUVDataset(APIBasedDataset):
    """MUV dataset via DeepChem"""

    TARGET_NAMES = [
        'MUV-466', 'MUV-548', 'MUV-600', 'MUV-644', 'MUV-652', 'MUV-689',
        'MUV-692', 'MUV-712', 'MUV-713', 'MUV-733', 'MUV-737', 'MUV-810',
        'MUV-832', 'MUV-846', 'MUV-852', 'MUV-858', 'MUV-859'
    ]

    def __init__(self, name: str = "MUV", data_dir: Path = None):
        super().__init__(name, data_dir or Path.home() / '.deepchem' / 'datasets')
        self.tasks = None
        self._target_data = None
        self._loaded = False

    def load(self):
        """Load MUV dataset using DeepChem"""
        if self._loaded:
            return self

        print(f"\n[INFO] Loading {self.name} dataset from DeepChem...")

        import warnings
        import deepchem as dc

        # Suppress DeepChem's deprecation warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*MorganGenerator.*")
            tasks, datasets, transformers = dc.molnet.load_muv(
                featurizer='raw',
                splitter='random',
                data_dir=str(self.root_path)
            )

        self.tasks = tasks
        train_dataset, valid_dataset, test_dataset = datasets

        print(f"[INFO] Loaded {self.name} with {len(tasks)} targets")
        print(f"[INFO] Train: {len(train_dataset)}, Valid: {len(valid_dataset)}, Test: {len(test_dataset)}")

        # Extract and organise SMILES by target
        print(f"[INFO] Extracting SMILES by target...")
        all_ids = []
        all_labels = []

        for dataset in [train_dataset, valid_dataset, test_dataset]:
            all_ids.extend(dataset.ids)
            all_labels.append(dataset.y)

        all_labels = np.vstack(all_labels)

        self._target_data = defaultdict(lambda: {"active": set(), "inactive": set()})

        for idx, smi in enumerate(tqdm(all_ids, desc="Processing molecules")):
            for task_idx, task_name in enumerate(self.tasks):
                label = all_labels[idx, task_idx]

                if np.isnan(label) or label == -1:
                    continue

                if label == 1:
                    self._target_data[task_name]["active"].add(smi)
                else:
                    self._target_data[task_name]["inactive"].add(smi)

        for task_name in self.tasks:
            n_active = len(self._target_data[task_name]["active"])
            n_inactive = len(self._target_data[task_name]["inactive"])
            print(f"  {task_name}: {n_active} actives, {n_inactive} inactives")

        self._loaded = True
        return self

    def enumerate_targets(self):
        """Enumerate all MUV targets"""
        if not self._loaded:
            self.load()

        for task_name in sorted(self._target_data.keys()):
            yield task_name, task_name  # For MUV, target_path is just the task name

    def read_target(self, target_path):
        """Read SMILES for a specific MUV target"""
        if not self._loaded:
            self.load()

        task_name = target_path  # For MUV, target_path is the task name

        for smi in self._target_data[task_name]["active"]:
            yield smi, "active"

        for smi in self._target_data[task_name]["inactive"]:
            yield smi, "inactive"


class DCOIDDataset(FileBasedDataset):
    """D-COID dataset with PDB-to-SMILES preprocessing"""

    def __init__(self, name: str, root_path: Path):
        super().__init__(name, root_path)
        self._metadata = None
        self._ensure_metadata()

    def _ensure_metadata(self):
        """Ensure metadata cache exists, create if needed"""
        cache_path = self.root_path / "dcoid_metadata.parquet"

        if cache_path.exists():
            print(f"[INFO] Loading D-COID metadata cache from {cache_path}")
            self._metadata = pd.read_parquet(cache_path)
            print(f"[INFO] Loaded {len(self._metadata)} ligands from cache")
        else:
            print(f"[INFO] D-COID metadata cache not found, preprocessing PDB files...")
            self._preprocess_pdb_files(cache_path)

    def _extract_ligand_smiles(self, pdb_path: Path, ligand_code: str):
        """Extract ligand from PDB file and convert to SMILES"""
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure('complex', str(pdb_path))

        # Find the ligand residue
        ligand_residue = None
        ligand_chain = None
        for model in structure:
            for chain in model:
                for residue in chain:
                    resname = residue.get_resname().strip()
                    if resname == ligand_code:
                        ligand_residue = residue
                        ligand_chain = chain
                        break
                if ligand_residue:
                    break
            if ligand_residue:
                break

        if not ligand_residue:
            return None, f"Ligand {ligand_code} not found in structure"

        # Create PDB block for just the ligand
        pdb_lines = []
        for atom in ligand_residue:
            coord = atom.get_coord()
            element = atom.element.strip() if hasattr(atom, 'element') and atom.element else atom.get_name()[0]
            line = (
                f"HETATM{atom.get_serial_number():5d} "
                f"{atom.get_name():^4s} "
                f"{ligand_residue.get_resname():3s} "
                f"{ligand_chain.get_id():1s}"
                f"{ligand_residue.get_id()[1]:4d}    "
                f"{coord[0]:8.3f}{coord[1]:8.3f}{coord[2]:8.3f}"
                f"  1.00  0.00          "
                f"{element:>2s}\n"
            )
            pdb_lines.append(line)

        pdb_lines.append("END\n")
        pdb_block = "".join(pdb_lines)

        # Convert to SMILES using RDKit
        mol = Chem.MolFromPDBBlock(pdb_block, sanitize=False, removeHs=False)
        if mol is None:
            return None, "Failed to parse ligand PDB block"

        # Try to sanitize and remove hydrogens
        try:
            Chem.SanitizeMol(mol)
            mol = Chem.RemoveHs(mol)
        except Exception as e:
            # Continue even if sanitization fails
            print(f"[WARNING] Sanitization failed for {pdb_path}: {e}")

        # Convert to SMILES
        smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
        if not smiles:
            return None, "Failed to generate SMILES"

        return smiles, None

    def _preprocess_pdb_files(self, cache_path: Path):
        """Extract ligands from PDB files and create metadata cache"""
        # Read ligand IDs from txt files
        actives_txt = self.root_path / "actives.txt"
        decoys_txt = self.root_path / "decoys.txt"

        if not actives_txt.exists() or not decoys_txt.exists():
            raise FileNotFoundError(
                f"Missing required txt files in {self.root_path}. "
                f"Need both actives.txt and decoys.txt"
            )

        ligand_data = []

        # Process actives
        with open(actives_txt, 'r') as f:
            for line in f:
                ligand_ids = line.strip().split()
                for ligand_id in ligand_ids:
                    if ligand_id:  # Skip empty strings
                        ligand_data.append((ligand_id, "active"))

        # Process decoys
        with open(decoys_txt, 'r') as f:
            for line in f:
                ligand_ids = line.strip().split()
                for ligand_id in ligand_ids:
                    if ligand_id:  # Skip empty strings
                        ligand_data.append((ligand_id, "decoy"))

        print(f"[INFO] Found {len(ligand_data)} ligands to process")

        if not ligand_data:
            raise ValueError("No ligand data found in txt files")

        # Extract SMILES from PDB files
        records = []
        failed_extractions = []

        for ligand_id, label in tqdm(ligand_data, desc="Extracting ligands from PDB files"):
            # Determine subdirectory
            subdir = "actives" if label == "active" else "decoys"
            pdb_filename = f"mini_complex_{ligand_id}.pdb"
            pdb_path = self.root_path / subdir / pdb_filename

            # Extract target_id (first 4 chars - PDB ID)
            target_id = ligand_id[:4]

            # Extract ligand code (between first and second underscore)
            parts = ligand_id.split('_')
            ligand_code = parts[1] if len(parts) > 1 else None

            smiles = None
            error = None

            if not pdb_path.exists():
                error = "PDB file not found"
                failed_extractions.append((ligand_id, error))
            elif not ligand_code:
                error = "Could not parse ligand code from ligand_id"
                failed_extractions.append((ligand_id, error))
            else:
                try:
                    smiles, error = self._extract_ligand_smiles(pdb_path, ligand_code)
                    if error:
                        failed_extractions.append((ligand_id, error))
                except Exception as e:
                    error = f"Exception during extraction: {str(e)}"
                    failed_extractions.append((ligand_id, error))
                    print(f"[ERROR] Failed to extract {ligand_id}: {error}")

            records.append({
                'ligand_id': ligand_id,
                'target_id': target_id,
                'label': label,
                'smiles': smiles,
                'file_path': str(pdb_path),
                'error': error
            })

        # Save to parquet
        self._metadata = pd.DataFrame(records)

        # Report statistics
        total = len(self._metadata)
        successful = self._metadata['smiles'].notna().sum()
        failed = total - successful
        print(f"[INFO] Extraction complete: {successful}/{total} successful ({failed} failed)")

        if failed > 0:
            print(f"[INFO] Failed extractions by error type:")
            error_counts = self._metadata[self._metadata['smiles'].isna()]['error'].value_counts()
            for error_type, count in error_counts.items():
                print(f"  - {error_type}: {count}")

            # Show first few failures for debugging
            print(f"[INFO] First 5 failed extractions:")
            for ligand_id, error in failed_extractions[:5]:
                print(f"  - {ligand_id}: {error}")

        if successful == 0:
            raise RuntimeError(
                f"Failed to extract any SMILES strings from {total} ligands. "
                f"Check that PDB files are valid and ligand codes are correct."
            )

        self._metadata.to_parquet(cache_path)
        print(f"[INFO] Saved metadata cache to {cache_path}")

    def enumerate_targets(self):
        """Enumerate all targets in D-COID"""
        if self._metadata is None or self._metadata.empty:
            return

        # Group by target_id and yield unique targets
        for target_id in sorted(self._metadata['target_id'].unique()):
            yield target_id, target_id

    def read_target(self, target_path):
        """Read SMILES for a specific target"""
        if self._metadata is None or self._metadata.empty:
            return

        target_id = target_path
        target_data = self._metadata[self._metadata['target_id'] == target_id]

        for _, row in target_data.iterrows():
            # Only yield if SMILES was successfully extracted
            if pd.notna(row['smiles']) and row['smiles']:
                yield row['smiles'], row['label']