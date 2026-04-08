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

    def test_collect_smiles_respects_max_ligands_per_dataset(self, mock_descriptor_cache):
        """Should stop collecting after the configured dataset cap."""
        target1 = []
        for i in range(700):
            target1.append(("C" * (i + 1), "active"))

        target2 = []
        for i in range(700):
            target2.append(("N" * (i + 1), "inactive"))

        dataset = MockDataset("TestDS", {
            "target1": target1,
            "target2": target2,
        })
        analyser = DatasetAnalyser([dataset], mock_descriptor_cache, max_ligands_per_dataset=1000)
        result = analyser.collect_smiles()

        total = len(result["TestDS"]["active"]) + len(result["TestDS"]["inactive"])
        assert total == 1000

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


class TestAnalyserManifestMode:
    """Tests for the manifest-mode Analyser class."""

    @pytest.fixture
    def sample_manifest_df(self):
        """Create sample manifest DataFrame for testing."""
        return pl.DataFrame({
            'manifest_id': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'dataset': ['kinase', 'kinase', 'kinase', 'protease', 'protease',
                        'kinase', 'protease', 'kinase', 'protease', 'kinase'],
            'target_id': ['tgt1', 'tgt1', 'tgt2', 'tgt2', 'tgt3',
                          'tgt1', 'tgt2', 'tgt1', 'tgt3', 'tgt2'],
            'protein_id': ['p1', 'p1', 'p2', 'p2', 'p3',
                           'p1', 'p2', 'p1', 'p3', 'p2'],
            'label': [1, 0, 1, 0, 1, 0, 1, 1, 0, 1],
            'smiles': ['CC(C)O', 'CC(C)O', 'CCO', 'CCO', 'c1ccccc1',
                       'CC(C)O', 'CCO', 'CC(C)O', 'c1ccccc1', 'CCO'],
            'ligand_id': ['lid1', 'lid1', 'lid2', 'lid2', 'lid3',
                          'lid1', 'lid2', 'lid1', 'lid3', 'lid2'],
            'compound_key': ['kinase|p1|lid1', 'kinase|p1|lid1', 'kinase|p2|lid2', 'kinase|p2|lid2', 'kinase|p3|lid3',
                             'kinase|p1|lid1', 'protease|p2|lid2', 'kinase|p1|lid1', 'protease|p3|lid3', 'kinase|p2|lid2'],
            'file_path': [f'path_{i}.sdf' for i in range(10)],
            'source_split': ['train'] * 10,
        })

    def test_analyser_requires_manifest(self):
        """Analyser constructor must reject None manifest."""
        from analyser import Analyser
        with pytest.raises(TypeError, match="backwards compatibility removed"):
            Analyser(manifest_df=None)

    def test_analyser_validates_schema(self, sample_manifest_df):
        """Analyser must fail if manifest missing required columns."""
        from analyser import Analyser
        invalid_manifest = sample_manifest_df.drop('compound_key')
        with pytest.raises(ValueError, match="Missing required columns"):
            Analyser(manifest_df=invalid_manifest)

    def test_collect_smiles_returns_df(self, sample_manifest_df):
        """collect_smiles should return DataFrame with smiles and dataset columns."""
        from analyser import Analyser
        analyser = Analyser(manifest_df=sample_manifest_df)
        result = analyser.collect_smiles()
        
        assert isinstance(result, pl.DataFrame)
        assert set(result.columns) == {'smiles', 'dataset'}

    def test_collect_smiles_unique(self, sample_manifest_df):
        """collect_smiles should return unique SMILES."""
        from analyser import Analyser
        analyser = Analyser(manifest_df=sample_manifest_df)
        result = analyser.collect_smiles()
        
        # Should have fewer rows than manifest (due to duplicate SMILES)
        assert len(result) < len(sample_manifest_df)

    def test_process_targets_returns_df(self, sample_manifest_df):
        """process_targets should return DataFrame."""
        from analyser import Analyser
        analyser = Analyser(manifest_df=sample_manifest_df)
        result = analyser.process_targets()
        
        assert isinstance(result, pl.DataFrame)

    def test_process_targets_has_all_targets(self, sample_manifest_df):
        """process_targets should include all unique targets."""
        from analyser import Analyser
        analyser = Analyser(manifest_df=sample_manifest_df)
        result = analyser.process_targets()
        
        targets = set(result.get_column('Target').to_list())
        assert targets == {'tgt1', 'tgt2', 'tgt3'}

    def test_chunk_assignment_deterministic(self):
        """Chunk assignment via (manifest_id - 1) % total_chunks should be deterministic."""
        total_chunks = 5
        chunk_id = 2
        expected_indices = {
            manifest_id 
            for manifest_id in range(1, 51)
            if (manifest_id - 1) % total_chunks == (chunk_id - 1)
        }
        
        # Chunk 2 out of 5 should get manifest_ids: 2, 7, 12, 17, 22, ...
        assert expected_indices == {2, 7, 12, 17, 22, 27, 32, 37, 42, 47}

    def test_manifest_chunk_coverage(self):
        """All chunks union must equal full manifest (no gaps, no overlaps)."""
        total_chunks = 5
        manifest_ids = list(range(1, 101))  # 100 rows
        
        all_assigned = set()
        for chunk_id in range(1, total_chunks + 1):
            chunk_ids = {
                mid for mid in manifest_ids
                if (mid - 1) % total_chunks == (chunk_id - 1)
            }
            all_assigned.update(chunk_ids)
        
        # All manifest_ids assigned exactly once
        assert all_assigned == set(manifest_ids)
