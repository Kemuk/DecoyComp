"""
Shared pytest fixtures for DecoyComp tests.
"""
import sys
from pathlib import Path

import pytest
import polars as pl

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


# --- Sample SMILES for testing ---
VALID_SMILES = [
    ("CCO", "ethanol"),
    ("CCCO", "propanol"),
    ("c1ccccc1", "benzene"),
    ("CC(=O)OC1=CC=CC=C1C(=O)O", "aspirin"),
    ("CN1C=NC2=C1C(=O)N(C(=O)N2C)C", "caffeine"),
    ("CC(C)CC1=CC=C(C=C1)C(C)C(=O)O", "ibuprofen"),
]

INVALID_SMILES = [
    "not_a_smiles",
    "invalid[molecule",
    "C1CC1C1",  # Invalid ring
    "",
]

SALT_SMILES = [
    "[Na+].[Cl-]",  # Sodium chloride
    "CC(=O)O.[Na]",  # Sodium acetate
    "c1ccccc1.[Cl-]",  # Benzene with chloride
]


@pytest.fixture
def sample_smiles() -> list[str]:
    """Return a list of valid SMILES strings for testing."""
    return [smi for smi, _ in VALID_SMILES]


@pytest.fixture
def invalid_smiles() -> list[str]:
    """Return a list of invalid SMILES strings for testing."""
    return INVALID_SMILES


@pytest.fixture
def salt_smiles() -> list[str]:
    """Return a list of SMILES containing salts."""
    return SALT_SMILES


@pytest.fixture
def sample_metadata() -> pl.DataFrame:
    """Create sample metadata DataFrame for testing."""
    return pl.DataFrame({
        "target_id": ["target1", "target1", "target1", "target2", "target2"],
        "smiles": ["CCO", "CCCO", "c1ccccc1", "CC(=O)O", "CCN"],
        "label": ["active", "inactive", "active", "inactive", "active"],
    })


@pytest.fixture
def temp_smiles_file(tmp_path: Path) -> Path:
    """Create a temporary SMILES file for testing."""
    smi_file = tmp_path / "test_actives.smi"
    smi_file.write_text("CCO\tethanol\nCCCO\tpropanol\nc1ccccc1\tbenzene\n")
    return smi_file


@pytest.fixture
def temp_dataset_dir(tmp_path: Path) -> Path:
    """Create a temporary dataset directory structure for testing."""
    # Create a minimal LIT-PCBA-like structure
    dataset_dir = tmp_path / "test_dataset"
    target_dir = dataset_dir / "TEST001"
    target_dir.mkdir(parents=True)

    # Create actives file
    actives_file = target_dir / "actives.smi"
    actives_file.write_text("CCO\nCCCO\n")

    # Create inactives file
    inactives_file = target_dir / "inactives.smi"
    inactives_file.write_text("c1ccccc1\nCCN\nCCCC\n")

    return dataset_dir


@pytest.fixture
def mock_descriptor_cache() -> dict:
    """Create a mock descriptor cache for testing."""
    from molecular_utils import MolDescriptors

    return {
        "CCO": (False, MolDescriptors(
            mw=46.07, clogp=-0.18, tpsa=20.23, fsp3=0.5,
            rb=0, lip_pass=True, veber_pass=True, has_metal=False, pains_hit=False
        )),
        "CCCO": (False, MolDescriptors(
            mw=60.10, clogp=0.25, tpsa=20.23, fsp3=0.67,
            rb=1, lip_pass=True, veber_pass=True, has_metal=False, pains_hit=False
        )),
        "c1ccccc1": (False, MolDescriptors(
            mw=78.11, clogp=1.68, tpsa=0.0, fsp3=0.0,
            rb=0, lip_pass=True, veber_pass=True, has_metal=False, pains_hit=False
        )),
        "invalid": (False, None),
    }
