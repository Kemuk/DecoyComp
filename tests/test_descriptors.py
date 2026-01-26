"""
Unit tests for molecular descriptor calculations.
"""
import pytest
from rdkit import Chem

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from molecular_utils import (
    DescriptorCalculator,
    MolDescriptors,
    aggregate_from_cache,
    METALS,
)


class TestDescriptorCalculator:
    """Unit tests for the DescriptorCalculator class."""

    def test_calculate_returns_mol_descriptors(self):
        """Valid molecule should return MolDescriptors namedtuple."""
        mol = Chem.MolFromSmiles("CCO")
        result = DescriptorCalculator.calculate(mol)
        assert isinstance(result, MolDescriptors)

    def test_calculate_ethanol_mw(self):
        """Ethanol should have MW around 46."""
        mol = Chem.MolFromSmiles("CCO")
        result = DescriptorCalculator.calculate(mol)
        assert 45 < result.mw < 47

    def test_calculate_aspirin_lipinski(self):
        """Aspirin should be Lipinski compliant."""
        mol = Chem.MolFromSmiles("CC(=O)OC1=CC=CC=C1C(=O)O")
        result = DescriptorCalculator.calculate(mol)
        assert result.lip_pass is True

    def test_calculate_large_molecule_lipinski_fail(self):
        """Large molecule should fail Lipinski's rule."""
        # A molecule with MW > 500
        large_smiles = "CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC"
        mol = Chem.MolFromSmiles(large_smiles)
        result = DescriptorCalculator.calculate(mol)
        assert result.lip_pass is False

    def test_calculate_veber_compliance(self):
        """Small molecule should be Veber compliant."""
        mol = Chem.MolFromSmiles("CCO")
        result = DescriptorCalculator.calculate(mol)
        assert result.veber_pass is True

    def test_calculate_metal_detection(self):
        """Molecule with metal should be flagged."""
        # Ferrocene has iron
        mol = Chem.MolFromSmiles("[Fe]")
        result = DescriptorCalculator.calculate(mol)
        assert result.has_metal is True

    def test_calculate_no_metal(self):
        """Organic molecule should not have metal flag."""
        mol = Chem.MolFromSmiles("c1ccccc1")
        result = DescriptorCalculator.calculate(mol)
        assert result.has_metal is False

    def test_calculate_one_smiles_valid(self):
        """Valid SMILES should return tuple with descriptors."""
        smi, salts, desc = DescriptorCalculator.calculate_one_smiles("CCO")
        assert smi == "CCO"
        assert salts is False
        assert desc is not None
        assert isinstance(desc, MolDescriptors)

    def test_calculate_one_smiles_invalid(self):
        """Invalid SMILES should return tuple with None descriptors."""
        smi, salts, desc = DescriptorCalculator.calculate_one_smiles("not_valid")
        assert smi == "not_valid"
        assert desc is None

    def test_calculate_one_smiles_salt_detection(self):
        """SMILES with '.' should be flagged as salt."""
        smi, salts, desc = DescriptorCalculator.calculate_one_smiles("[Na+].[Cl-]")
        assert salts is True

    def test_calculate_one_smiles_no_salt(self):
        """Normal SMILES should not be flagged as salt."""
        smi, salts, desc = DescriptorCalculator.calculate_one_smiles("CCO")
        assert salts is False

    def test_pains_catalog_initialized(self):
        """PAINS catalog should be initialized on first use."""
        # Force initialization by calculating on a molecule
        mol = Chem.MolFromSmiles("c1ccccc1")
        DescriptorCalculator.calculate(mol)
        assert DescriptorCalculator._PAINS_CATALOG is not None

    def test_calculate_all_parallel_basic(self, sample_smiles):
        """Parallel calculation should work on valid SMILES list."""
        result = DescriptorCalculator.calculate_all_parallel(
            sample_smiles,
            workers=2,
            use_cache=False,
            show_progress=False
        )
        assert isinstance(result, dict)
        assert len(result) == len(sample_smiles)
        for smi in sample_smiles:
            assert smi in result

    def test_calculate_all_parallel_handles_invalid(self, sample_smiles, invalid_smiles):
        """Parallel calculation should handle mix of valid and invalid."""
        all_smiles = sample_smiles + invalid_smiles[:2]
        result = DescriptorCalculator.calculate_all_parallel(
            all_smiles,
            workers=2,
            use_cache=False,
            show_progress=False
        )
        # All SMILES should be in result
        assert len(result) == len(all_smiles)
        # Invalid ones should have None descriptors
        for smi in invalid_smiles[:2]:
            if smi:  # Skip empty string
                assert result[smi][1] is None


class TestAggregateFromCache:
    """Unit tests for the aggregate_from_cache function."""

    def test_aggregate_basic(self, mock_descriptor_cache):
        """Basic aggregation should work correctly."""
        smiles_list = ["CCO", "CCCO"]
        result = aggregate_from_cache(smiles_list, mock_descriptor_cache)

        assert result["NumberLigands"] == 2
        assert result["NumberInvalidSMILES"] == 0
        assert result["NumberWithSalts"] == 0

    def test_aggregate_with_invalid(self, mock_descriptor_cache):
        """Aggregation should count invalid SMILES."""
        smiles_list = ["CCO", "invalid"]
        result = aggregate_from_cache(smiles_list, mock_descriptor_cache)

        assert result["NumberLigands"] == 1
        assert result["NumberInvalidSMILES"] == 1

    def test_aggregate_missing_smiles(self, mock_descriptor_cache):
        """SMILES not in cache should be skipped."""
        smiles_list = ["CCO", "not_in_cache"]
        result = aggregate_from_cache(smiles_list, mock_descriptor_cache)

        assert result["NumberLigands"] == 1

    def test_aggregate_compliance_rates(self, mock_descriptor_cache):
        """Compliance counts should be correct."""
        smiles_list = ["CCO", "CCCO", "c1ccccc1"]
        result = aggregate_from_cache(smiles_list, mock_descriptor_cache)

        assert result["NumberLipinskiCompliant"] == 3
        assert result["NumberVeberCompliant"] == 3

    def test_aggregate_empty_list(self, mock_descriptor_cache):
        """Empty list should return zero counts."""
        result = aggregate_from_cache([], mock_descriptor_cache)

        assert result["NumberLigands"] == 0
        assert result["NumberInvalidSMILES"] == 0


class TestMetalConstants:
    """Tests for the METALS constant."""

    def test_metals_contains_common_metals(self):
        """METALS should contain common drug-interfering metals."""
        # Iron (26), Zinc (30), Copper (29)
        assert 26 in METALS
        assert 30 in METALS
        assert 29 in METALS

    def test_metals_excludes_carbon(self):
        """METALS should not contain carbon (6)."""
        assert 6 not in METALS

    def test_metals_excludes_nitrogen(self):
        """METALS should not contain nitrogen (7)."""
        assert 7 not in METALS


class TestDescriptorConsistency:
    """Tests for descriptor calculation consistency."""

    def test_same_smiles_same_result(self):
        """Same SMILES should produce identical results."""
        result1 = DescriptorCalculator.calculate_one_smiles("CCO")
        result2 = DescriptorCalculator.calculate_one_smiles("CCO")
        assert result1 == result2

    def test_canonical_smiles_equivalent(self):
        """Equivalent SMILES should produce same MW."""
        mol1 = Chem.MolFromSmiles("CCO")
        mol2 = Chem.MolFromSmiles("OCC")  # Same molecule, different order

        desc1 = DescriptorCalculator.calculate(mol1)
        desc2 = DescriptorCalculator.calculate(mol2)

        assert abs(desc1.mw - desc2.mw) < 0.01
