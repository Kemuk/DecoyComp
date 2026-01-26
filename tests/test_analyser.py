"""
Unit tests for the analysis engine.
"""
import pytest
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

import polars as pl

from analyser import TargetStats, DatasetAnalyser
from molecular_utils import MolDescriptors


class TestTargetStats:
    """Tests for the TargetStats class."""

    def test_init_zeroes_counts(self, mock_descriptor_cache):
        """Should initialize all counts to zero."""
        stats = TargetStats(mock_descriptor_cache)
        assert stats.counts["actives"] == 0
        assert stats.counts["decoys"] == 0
        assert stats.counts["invalid"] == 0

    def test_update_active_increments_count(self, mock_descriptor_cache):
        """Should increment active count for active molecules."""
        stats = TargetStats(mock_descriptor_cache)
        stats.update("CCO", "active")
        assert stats.counts["actives"] == 1

    def test_update_decoy_increments_count(self, mock_descriptor_cache):
        """Should increment decoy count for non-active molecules."""
        stats = TargetStats(mock_descriptor_cache)
        stats.update("CCO", "decoy")
        assert stats.counts["decoys"] == 1

    def test_update_invalid_smiles(self, mock_descriptor_cache):
        """Should count invalid SMILES."""
        stats = TargetStats(mock_descriptor_cache)
        stats.update("invalid", "active")
        assert stats.counts["invalid"] == 1

    def test_update_missing_smiles(self, mock_descriptor_cache):
        """Should count SMILES not in cache as invalid."""
        stats = TargetStats(mock_descriptor_cache)
        stats.update("not_in_cache", "active")
        assert stats.counts["invalid"] == 1

    def test_update_accumulates_sums(self, mock_descriptor_cache):
        """Should accumulate molecular weight sums."""
        stats = TargetStats(mock_descriptor_cache)
        stats.update("CCO", "active")  # MW ~46
        stats.update("CCCO", "active")  # MW ~60
        assert stats.sums["mw"] > 100

    def test_report_returns_dict(self, mock_descriptor_cache):
        """Report should return dictionary with all fields."""
        stats = TargetStats(mock_descriptor_cache)
        stats.update("CCO", "active")
        stats.update("CCCO", "decoy")

        report = stats.report("TestDataset", "TestTarget")

        assert isinstance(report, dict)
        assert report["Dataset"] == "TestDataset"
        assert report["Target"] == "TestTarget"
        assert report["NumberActives"] == 1
        assert report["NumberDecoys/Inactives"] == 1

    def test_report_calculates_means(self, mock_descriptor_cache):
        """Report should calculate mean values."""
        stats = TargetStats(mock_descriptor_cache)
        stats.update("CCO", "active")
        stats.update("CCCO", "active")

        report = stats.report("TestDataset", "TestTarget")

        # Mean MW should be average of CCO (~46) and CCCO (~60)
        assert 50 < report["Mean_MW"] < 55

    def test_report_handles_empty_stats(self, mock_descriptor_cache):
        """Report should handle case with no molecules."""
        stats = TargetStats(mock_descriptor_cache)
        report = stats.report("TestDataset", "TestTarget")

        assert report["NumberActives"] == 0
        assert report["Mean_MW"] == 0.0
        assert report["LipinskiComplianceRate"] == 0.0

    def test_report_tracks_rotatable_bonds(self, mock_descriptor_cache):
        """Report should track rotatable bonds separately for actives/decoys."""
        stats = TargetStats(mock_descriptor_cache)
        stats.update("CCO", "active")
        stats.update("CCCO", "decoy")

        report = stats.report("TestDataset", "TestTarget")

        assert "_RBs_Actives" in report
        assert "_RBs_DecoysOrInactives" in report


class MockDataset:
    """Mock dataset for testing DatasetAnalyser."""

    def __init__(self, name: str, targets: dict):
        self.name = name
        self._targets = targets

    def enumerate_targets(self):
        for target_name in self._targets:
            yield target_name, target_name

    def read_target(self, target_path):
        for smi, label in self._targets.get(target_path, []):
            yield smi, label


class TestDatasetAnalyser:
    """Tests for the DatasetAnalyser class."""

    @pytest.fixture
    def mock_dataset(self):
        """Create a mock dataset for testing."""
        return MockDataset("TestDS", {
            "target1": [("CCO", "active"), ("CCCO", "inactive")],
            "target2": [("c1ccccc1", "active")],
        })

    def test_collect_smiles_returns_dict(self, mock_dataset, mock_descriptor_cache):
        """Should return dict of unique SMILES by dataset."""
        analyser = DatasetAnalyser([mock_dataset], mock_descriptor_cache)
        result = analyser.collect_smiles()

        assert isinstance(result, dict)
        assert "TestDS" in result
        assert "active" in result["TestDS"]
        assert "inactive" in result["TestDS"]

    def test_collect_smiles_unique(self, mock_descriptor_cache):
        """Should deduplicate SMILES across targets."""
        dataset = MockDataset("TestDS", {
            "target1": [("CCO", "active")],
            "target2": [("CCO", "active")],  # Same SMILES
        })
        analyser = DatasetAnalyser([dataset], mock_descriptor_cache)
        result = analyser.collect_smiles()

        assert len(result["TestDS"]["active"]) == 1

    def test_collect_smiles_caches_result(self, mock_dataset, mock_descriptor_cache):
        """Should cache SMILES collection result."""
        analyser = DatasetAnalyser([mock_dataset], mock_descriptor_cache)

        result1 = analyser.collect_smiles()
        result2 = analyser.collect_smiles()

        assert result1 is result2

    def test_process_targets_returns_polars_df(self, mock_dataset, mock_descriptor_cache):
        """Should return Polars DataFrame."""
        analyser = DatasetAnalyser([mock_dataset], mock_descriptor_cache)
        result = analyser.process_targets()

        assert isinstance(result, pl.DataFrame)

    def test_process_targets_has_all_targets(self, mock_dataset, mock_descriptor_cache):
        """Should include all targets in result."""
        analyser = DatasetAnalyser([mock_dataset], mock_descriptor_cache)
        result = analyser.process_targets()

        targets = result.get_column("Target").to_list()
        assert "target1" in targets
        assert "target2" in targets

    def test_create_dataset_summary_returns_df(self, mock_dataset, mock_descriptor_cache):
        """Should return Polars DataFrame."""
        analyser = DatasetAnalyser([mock_dataset], mock_descriptor_cache)
        result = analyser.create_dataset_summary()

        assert isinstance(result, pl.DataFrame)

    def test_create_split_summary_has_buckets(self, mock_dataset, mock_descriptor_cache):
        """Split summary should have Actives/Inactives/All buckets."""
        analyser = DatasetAnalyser([mock_dataset], mock_descriptor_cache)
        result = analyser.create_split_summary()

        buckets = result.get_column("Bucket").to_list()
        assert "Actives" in buckets
        assert "Inactives" in buckets
        assert "All" in buckets

    def test_write_smiles_files_creates_files(self, mock_dataset, mock_descriptor_cache, tmp_path):
        """Should create SMILES files in output directory."""
        analyser = DatasetAnalyser([mock_dataset], mock_descriptor_cache)
        analyser.write_smiles_files(tmp_path)

        actives_file = tmp_path / "TestDS_actives.smi"
        inactives_file = tmp_path / "TestDS_inactives.smi"

        assert actives_file.exists()
        assert inactives_file.exists()

    def test_write_smiles_files_skips_existing(self, mock_dataset, mock_descriptor_cache, tmp_path):
        """Should skip existing files."""
        # Create existing files
        actives_file = tmp_path / "TestDS_actives.smi"
        inactives_file = tmp_path / "TestDS_inactives.smi"
        actives_file.write_text("existing\n")
        inactives_file.write_text("existing\n")

        analyser = DatasetAnalyser([mock_dataset], mock_descriptor_cache)
        analyser.write_smiles_files(tmp_path)

        # Files should not be overwritten
        assert actives_file.read_text() == "existing\n"
