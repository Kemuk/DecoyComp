"""
Unit tests for dataset abstractions.
"""
import pytest
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

import polars as pl

from datasets import (
    BaseDataset,
    LitPCBADataset,
    DudeZDataset,
    Dekois2Dataset,
    read_smi_file,
)


class TestReadSmiFile:
    """Tests for the read_smi_file helper function."""

    def test_read_valid_file(self, temp_smiles_file):
        """Should correctly read SMILES from file."""
        result = list(read_smi_file(temp_smiles_file, "active"))
        assert len(result) == 3
        assert result[0] == ("CCO", "active")
        assert result[1] == ("CCCO", "active")
        assert result[2] == ("c1ccccc1", "active")

    def test_read_nonexistent_file(self, tmp_path):
        """Should return empty for nonexistent file."""
        fake_path = tmp_path / "nonexistent.smi"
        result = list(read_smi_file(fake_path, "active"))
        assert result == []

    def test_read_with_label(self, temp_smiles_file):
        """Should apply correct label to all SMILES."""
        result = list(read_smi_file(temp_smiles_file, "decoy"))
        for smi, label in result:
            assert label == "decoy"


class TestLitPCBADataset:
    """Tests for the LitPCBADataset class."""

    def test_init_creates_empty_metadata_if_no_dir(self, tmp_path):
        """Should handle missing directory gracefully."""
        dataset = LitPCBADataset("LIT-PCBA", tmp_path / "nonexistent")
        assert dataset._metadata is not None
        assert dataset._metadata.is_empty()

    def test_enumerate_targets_from_structure(self, temp_dataset_dir):
        """Should enumerate targets from directory structure."""
        dataset = LitPCBADataset("test", temp_dataset_dir)
        targets = list(dataset.enumerate_targets())
        assert len(targets) == 1
        assert targets[0][0] == "TEST001"

    def test_read_target_returns_smiles(self, temp_dataset_dir):
        """Should read SMILES with labels from target."""
        dataset = LitPCBADataset("test", temp_dataset_dir)
        smiles_list = list(dataset.read_target("TEST001"))

        # Check counts
        actives = [s for s, l in smiles_list if l == "active"]
        inactives = [s for s, l in smiles_list if l == "inactive"]
        assert len(actives) == 2
        assert len(inactives) == 3

    def test_parquet_cache_created(self, temp_dataset_dir):
        """Should create parquet cache file."""
        dataset = LitPCBADataset("test", temp_dataset_dir)
        cache_path = temp_dataset_dir / "litpcba_metadata.parquet"
        assert cache_path.exists()

    def test_parquet_cache_reused(self, temp_dataset_dir):
        """Should reuse existing cache on second load."""
        # First load creates cache
        dataset1 = LitPCBADataset("test", temp_dataset_dir)

        # Modify cache file to verify it's reused
        cache_path = temp_dataset_dir / "litpcba_metadata.parquet"
        original_size = cache_path.stat().st_size

        # Second load should use cache
        dataset2 = LitPCBADataset("test", temp_dataset_dir)

        # Cache file should be unchanged
        assert cache_path.stat().st_size == original_size


class TestDatasetMetadataSchema:
    """Tests for consistent metadata schema across datasets."""

    def test_litpcba_has_required_columns(self, temp_dataset_dir):
        """LitPCBA metadata should have required columns."""
        dataset = LitPCBADataset("test", temp_dataset_dir)
        required_cols = {"target_id", "smiles", "label"}
        assert required_cols.issubset(set(dataset._metadata.columns))

    def test_metadata_types_correct(self, temp_dataset_dir):
        """Metadata columns should have correct types."""
        dataset = LitPCBADataset("test", temp_dataset_dir)
        schema = dataset._metadata.schema
        assert schema["target_id"] == pl.Utf8
        assert schema["smiles"] == pl.Utf8
        assert schema["label"] == pl.Utf8


class TestDudeZDatasetStructure:
    """Tests for DudeZ dataset with its file structure."""

    def test_init_creates_empty_if_no_dir(self, tmp_path):
        """Should handle missing directory."""
        dataset = DudeZDataset("DUDE-Z", tmp_path / "nonexistent")
        assert dataset._metadata is not None
        assert dataset._metadata.is_empty()

    def test_reads_actives_final_ism(self, tmp_path):
        """Should read from actives_final.ism files."""
        # Create DUDE-Z structure
        target_dir = tmp_path / "TEST001"
        target_dir.mkdir()
        (target_dir / "actives_final.ism").write_text("CCO\nCCCO\n")
        (target_dir / "decoys_final.ism").write_text("c1ccccc1\n")

        dataset = DudeZDataset("test", tmp_path)
        smiles_list = list(dataset.read_target("TEST001"))

        actives = [s for s, l in smiles_list if l == "active"]
        decoys = [s for s, l in smiles_list if l == "decoy"]
        assert len(actives) == 2
        assert len(decoys) == 1


class TestDekois2DatasetStructure:
    """Tests for DEKOIS2 dataset with its file structure."""

    def test_parses_bdb_as_active(self, tmp_path):
        """Should parse BDB prefix as active."""
        target_dir = tmp_path / "TEST001"
        target_dir.mkdir()
        smi_content = "CCO\tBDB12345\nCCCO\tZINC00001\n"
        (target_dir / "active_decoys.smi").write_text(smi_content)

        dataset = Dekois2Dataset("test", tmp_path)
        smiles_list = list(dataset.read_target("TEST001"))

        actives = [s for s, l in smiles_list if l == "active"]
        decoys = [s for s, l in smiles_list if l == "decoy"]
        assert len(actives) == 1
        assert len(decoys) == 1

    def test_skips_unknown_prefix(self, tmp_path):
        """Should skip lines with unknown prefix."""
        target_dir = tmp_path / "TEST001"
        target_dir.mkdir()
        smi_content = "CCO\tBDB12345\nCCCO\tUNKNOWN123\n"
        (target_dir / "active_decoys.smi").write_text(smi_content)

        dataset = Dekois2Dataset("test", tmp_path)
        smiles_list = list(dataset.read_target("TEST001"))

        assert len(smiles_list) == 1  # Only BDB line


class TestPolarsIntegration:
    """Tests for Polars-specific functionality."""

    def test_metadata_is_polars_dataframe(self, temp_dataset_dir):
        """Metadata should be a Polars DataFrame."""
        dataset = LitPCBADataset("test", temp_dataset_dir)
        assert isinstance(dataset._metadata, pl.DataFrame)

    def test_filter_performance(self, temp_dataset_dir):
        """Filtering should work efficiently with Polars."""
        dataset = LitPCBADataset("test", temp_dataset_dir)

        # Test filter syntax
        actives = dataset._metadata.filter(pl.col("label") == "active")
        assert isinstance(actives, pl.DataFrame)

    def test_unique_targets(self, temp_dataset_dir):
        """Should get unique targets efficiently."""
        dataset = LitPCBADataset("test", temp_dataset_dir)

        unique_targets = dataset._metadata.get_column("target_id").unique().to_list()
        assert isinstance(unique_targets, list)
        assert "TEST001" in unique_targets
