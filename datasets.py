#!/usr/bin/env python3
"""
Dataset abstractions for different benchmark datasets
"""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterator
from collections import defaultdict

import numpy as np
from tqdm.auto import tqdm


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