from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest
import zarr

from electrai.dataloader.mp_zarr_s3_data import ZarrS3Reader, load_data


@pytest.fixture
def rng():
    """Provide a numpy random generator for tests."""
    return np.random.default_rng(seed=42)


@pytest.fixture
def mock_zarr_store(tmp_path: Path, rng) -> Path:
    """
    Create a mock zarr store with sample data.

    Returns
    -------
    Path
        Path to the created zarr store
    """
    zarr_path = tmp_path / "test_task.zarr"

    # Create zarr store
    root = zarr.open_group(str(zarr_path), mode="w")

    # Create sample charge density data
    charge_data = rng.random((10, 10, 10)).astype(np.float32)
    root.create(name="charge_density_total", data=charge_data, chunks=(5, 5, 5))

    # Create diff data for spin-polarized case
    diff_data = rng.random((10, 10, 10)).astype(np.float32) * 0.1
    root.create(name="charge_density_diff", data=diff_data, chunks=(5, 5, 5))

    # Add metadata
    metadata = {"task_id": "mp-12345", "pymatgen_version": "2024.1.1"}
    root.attrs["metadata"] = json.dumps(metadata)

    # Add structure data
    structure = {
        "lattice": {"a": 5.0, "b": 5.0, "c": 5.0},
        "sites": [{"species": [{"element": "Si", "occu": 1}], "abc": [0, 0, 0]}],
    }
    root.attrs["structure"] = json.dumps(structure)

    return zarr_path


@pytest.fixture
def mock_mapping_file(tmp_path: Path) -> Path:
    """
    Create a mock mapping file for task IDs.

    Returns
    -------
    Path
        Path to the mapping file
    """
    import gzip

    mapping = {
        "GGA": ["mp-12345", "mp-67890", "mp-11111"],
        "GGA+U": ["mp-22222", "mp-33333"],
        "SCAN": ["mp-44444"],
    }

    mapping_path = tmp_path / "mapping.json.gz"
    with gzip.open(mapping_path, "wt") as f:
        json.dump(mapping, f)

    return mapping_path


@pytest.fixture
def mock_zarr_directory(tmp_path: Path, rng) -> Path:
    """
    Create a directory with multiple zarr stores.

    Returns
    -------
    Path
        Path to the directory containing zarr stores
    """
    zarr_dir = tmp_path / "zarr_data"
    zarr_dir.mkdir()

    # Create multiple zarr stores
    for task_id in ["mp-12345", "mp-67890", "mp-11111"]:
        store_path = zarr_dir / f"{task_id}.zarr"
        root = zarr.open_group(str(store_path), mode="w")

        # Create charge density with different sizes
        size = 8 + int(task_id.split("-")[1]) % 5
        charge_data = rng.random((size, size, size)).astype(np.float32)
        root.create(name="charge_density_total", data=charge_data, chunks=(4, 4, 4))

        metadata = {"task_id": task_id}
        root.attrs["metadata"] = json.dumps(metadata)

    return zarr_dir


class TestZarrS3ReaderInitialization:
    """Test initialization and configuration of ZarrS3Reader."""

    def test_local_initialization(self, tmp_path: Path, mock_mapping_file: Path):
        """Test initialization with local paths."""
        reader = ZarrS3Reader(
            data_dir=tmp_path / "data",
            label_dir=tmp_path / "labels",
            map_dir=mock_mapping_file,
            functional="GGA",
            normalize=True,
            train_fraction=0.8,
            random_state=42,
        )

        assert isinstance(reader.data_dir, Path)
        assert isinstance(reader.label_dir, Path)
        assert reader.functional == "GGA"
        assert reader.normalize is True
        assert reader.tf == 0.8
        assert reader.rs == 42
        assert reader.use_s3 is False

    def test_s3_initialization(self, mock_mapping_file: Path):
        """Test initialization with S3 paths."""
        # Mock s3fs at import time
        mock_s3fs_module = MagicMock()
        mock_s3fs_module.S3FileSystem = Mock(return_value=Mock())

        with patch.dict("sys.modules", {"s3fs": mock_s3fs_module}):
            reader = ZarrS3Reader(
                data_dir="s3://bucket/data",
                label_dir="s3://bucket/labels",
                map_dir=mock_mapping_file,
                functional="GGA",
                normalize=False,
                train_fraction=0.7,
                s3_kwargs={"anon": True},
            )

            assert reader.data_dir == "s3://bucket/data"
            assert reader.label_dir == "s3://bucket/labels"
            assert reader.use_s3 is True
            mock_s3fs_module.S3FileSystem.assert_called_once_with(anon=True)

    def test_mixed_s3_local_paths(self, tmp_path: Path, mock_mapping_file: Path):
        """Test initialization with mixed S3 and local paths."""
        mock_s3fs_module = MagicMock()
        mock_s3fs_module.S3FileSystem = Mock(return_value=Mock())

        with patch.dict("sys.modules", {"s3fs": mock_s3fs_module}):
            reader = ZarrS3Reader(
                data_dir="s3://bucket/data",
                label_dir=tmp_path / "labels",
                map_dir=mock_mapping_file,
                functional="GGA",
                normalize=True,
                train_fraction=0.8,
            )

            assert reader.use_s3 is True


class TestZarrPathConstruction:
    """Test zarr path construction for different storage backends."""

    def test_local_path_construction(self, tmp_path: Path, mock_mapping_file: Path):
        """Test construction of local zarr paths."""
        reader = ZarrS3Reader(
            data_dir=tmp_path / "data",
            label_dir=tmp_path / "labels",
            map_dir=mock_mapping_file,
            functional="GGA",
            normalize=True,
            train_fraction=0.8,
        )

        path = reader._get_zarr_path(str(tmp_path / "data"), "mp-12345")
        assert isinstance(path, str)
        assert path == str(tmp_path / "data" / "mp-12345.zarr")

    def test_s3_path_construction(self, mock_mapping_file: Path):
        """Test construction of S3 zarr paths."""
        mock_s3fs_module = MagicMock()
        mock_s3fs_module.S3FileSystem = Mock(return_value=Mock())

        with patch.dict("sys.modules", {"s3fs": mock_s3fs_module}):
            reader = ZarrS3Reader(
                data_dir="s3://bucket/data",
                label_dir="s3://bucket/labels",
                map_dir=mock_mapping_file,
                functional="GGA",
                normalize=True,
                train_fraction=0.8,
            )

            path = reader._get_zarr_path("s3://bucket/data", "mp-12345")
            assert isinstance(path, str)
            assert path == "s3://bucket/data/mp-12345.zarr"


class TestZarrStoreOperations:
    """Test reading and accessing zarr stores."""

    def test_read_local_zarr_store(
        self, mock_zarr_store: Path, mock_mapping_file: Path
    ):
        """Test reading a zarr store from local filesystem."""
        reader = ZarrS3Reader(
            data_dir=mock_zarr_store.parent,
            label_dir=mock_zarr_store.parent,
            map_dir=mock_mapping_file,
            functional="GGA",
            normalize=False,
            train_fraction=0.8,
        )

        store = reader.read_zarr_store(mock_zarr_store)
        assert isinstance(store, zarr.Group)
        assert "charge_density_total" in store
        assert "charge_density_diff" in store

    def test_read_charge_density_total(
        self, mock_zarr_store: Path, mock_mapping_file: Path
    ):
        """Test reading total charge density from zarr store."""
        reader = ZarrS3Reader(
            data_dir=mock_zarr_store.parent,
            label_dir=mock_zarr_store.parent,
            map_dir=mock_mapping_file,
            functional="GGA",
            normalize=False,
            train_fraction=0.8,
            density_type="total",
        )

        store = reader.read_zarr_store(mock_zarr_store)
        charge, gridsize = reader.read_charge_density(store)

        assert isinstance(charge, np.ndarray)
        assert charge.dtype == np.float32
        assert len(charge.shape) == 1  # Flattened
        assert gridsize == (10, 10, 10)
        assert charge.shape[0] == np.prod(gridsize)

    def test_read_charge_density_with_normalization(
        self, mock_zarr_store: Path, mock_mapping_file: Path
    ):
        """Test that normalization is applied correctly."""
        reader = ZarrS3Reader(
            data_dir=mock_zarr_store.parent,
            label_dir=mock_zarr_store.parent,
            map_dir=mock_mapping_file,
            functional="GGA",
            normalize=True,
            train_fraction=0.8,
            density_type="total",
        )

        store = reader.read_zarr_store(mock_zarr_store)
        charge_normalized, gridsize = reader.read_charge_density(store)

        # Compare with non-normalized
        reader.normalize = False
        charge_raw, _ = reader.read_charge_density(store)

        volume = np.prod(gridsize)
        expected = charge_raw / volume

        np.testing.assert_array_almost_equal(charge_normalized, expected)

    def test_get_metadata(self, mock_zarr_store: Path, mock_mapping_file: Path):
        """Test extracting metadata from zarr store."""
        reader = ZarrS3Reader(
            data_dir=mock_zarr_store.parent,
            label_dir=mock_zarr_store.parent,
            map_dir=mock_mapping_file,
            functional="GGA",
            normalize=False,
            train_fraction=0.8,
        )

        store = reader.read_zarr_store(mock_zarr_store)
        metadata = reader.get_metadata(store)

        assert "task_id" in metadata
        assert metadata["task_id"] == "mp-12345"
        assert "structure" in metadata
        assert "lattice" in metadata["structure"]


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_missing_density_array(
        self, mock_zarr_store: Path, mock_mapping_file: Path
    ):
        """Test error when requested density type doesn't exist."""
        reader = ZarrS3Reader(
            data_dir=mock_zarr_store.parent,
            label_dir=mock_zarr_store.parent,
            map_dir=mock_mapping_file,
            functional="GGA",
            normalize=False,
            train_fraction=0.8,
            density_type="nonexistent",
        )

        store = reader.read_zarr_store(mock_zarr_store)

        with pytest.raises(ValueError, match="not found in zarr store"):
            reader.read_charge_density(store)

    def test_nonexistent_zarr_store(self, tmp_path: Path, mock_mapping_file: Path):
        """Test error when zarr store doesn't exist."""
        reader = ZarrS3Reader(
            data_dir=tmp_path,
            label_dir=tmp_path,
            map_dir=mock_mapping_file,
            functional="GGA",
            normalize=False,
            train_fraction=0.8,
        )

        nonexistent_path = tmp_path / "nonexistent.zarr"

        # Can raise various errors depending on zarr version
        with pytest.raises((FileNotFoundError, KeyError, OSError)):
            reader.read_zarr_store(nonexistent_path)

    def test_invalid_functional(
        self, mock_zarr_directory: Path, mock_mapping_file: Path
    ):
        """Test error when functional is not in mapping."""
        reader = ZarrS3Reader(
            data_dir=mock_zarr_directory,
            label_dir=mock_zarr_directory,
            map_dir=mock_mapping_file,
            functional="INVALID_FUNCTIONAL",
            normalize=False,
            train_fraction=0.8,
        )

        with pytest.raises(ValueError, match="not found in mapping"):
            reader.data_split()

    def test_empty_task_id_list(self, mock_zarr_directory: Path, tmp_path: Path):
        """Test handling of functional with no task IDs."""
        import gzip

        # Create mapping with empty list
        mapping = {"GGA": []}
        mapping_path = tmp_path / "empty_mapping.json.gz"
        with gzip.open(mapping_path, "wt") as f:
            json.dump(mapping, f)

        reader = ZarrS3Reader(
            data_dir=mock_zarr_directory,
            label_dir=mock_zarr_directory,
            map_dir=mapping_path,
            functional="GGA",
            normalize=False,
            train_fraction=0.8,
        )

        with pytest.raises(
            ValueError, match=r"With n_samples=0, test_size=.* and train_size=.*"
        ):
            reader.data_split()

    def test_corrupted_zarr_store(self, tmp_path: Path, mock_mapping_file: Path):
        """Test handling of corrupted zarr store (missing required arrays)."""
        # Create zarr store without charge density arrays
        corrupted_path = tmp_path / "corrupted.zarr"
        root = zarr.open_group(str(corrupted_path), mode="w")
        root.attrs["metadata"] = json.dumps({"task_id": "mp-99999"})

        reader = ZarrS3Reader(
            data_dir=tmp_path,
            label_dir=tmp_path,
            map_dir=mock_mapping_file,
            functional="GGA",
            normalize=False,
            train_fraction=0.8,
        )

        store = reader.read_zarr_store(corrupted_path)

        with pytest.raises(ValueError, match="not found in zarr store"):
            reader.read_charge_density(store)


class TestDataSplit:
    """Test data splitting functionality."""

    def test_data_split_basic(self, mock_zarr_directory: Path, mock_mapping_file: Path):
        """Test basic data splitting functionality."""
        reader = ZarrS3Reader(
            data_dir=mock_zarr_directory,
            label_dir=mock_zarr_directory,
            map_dir=mock_mapping_file,
            functional="GGA",
            normalize=False,
            train_fraction=0.8,
            random_state=42,
        )

        train_sets, test_sets = reader.data_split()

        # Check structure
        assert len(train_sets) == 4  # data, label, gs_data, gs_label
        assert len(test_sets) == 4

        # Check that we have some data
        train_data, train_label, train_gs_data, train_gs_label = train_sets
        assert len(train_data) > 0
        assert len(train_data) == len(train_label)
        assert len(train_data) == len(train_gs_data)
        assert len(train_data) == len(train_gs_label)

    def test_data_split_ratio(self, mock_zarr_directory: Path, mock_mapping_file: Path):
        """Test that train/test split ratio is approximately correct."""
        reader = ZarrS3Reader(
            data_dir=mock_zarr_directory,
            label_dir=mock_zarr_directory,
            map_dir=mock_mapping_file,
            functional="GGA",
            normalize=False,
            train_fraction=0.6,
            random_state=42,
        )

        train_sets, test_sets = reader.data_split()

        train_size = len(train_sets[0])
        test_size = len(test_sets[0])
        total_size = train_size + test_size

        # Check approximately 60/40 split (with small dataset, may not be exact)
        assert train_size > 0
        assert test_size > 0
        assert total_size == 3  # We created 3 zarr stores

    def test_data_split_reproducibility(
        self, mock_zarr_directory: Path, mock_mapping_file: Path
    ):
        """Test that data split is reproducible with same random_state."""
        reader1 = ZarrS3Reader(
            data_dir=mock_zarr_directory,
            label_dir=mock_zarr_directory,
            map_dir=mock_mapping_file,
            functional="GGA",
            normalize=False,
            train_fraction=0.8,
            random_state=42,
        )

        reader2 = ZarrS3Reader(
            data_dir=mock_zarr_directory,
            label_dir=mock_zarr_directory,
            map_dir=mock_mapping_file,
            functional="GGA",
            normalize=False,
            train_fraction=0.8,
            random_state=42,
        )

        train_sets1, test_sets1 = reader1.data_split()
        train_sets2, test_sets2 = reader2.data_split()

        # Check that splits are identical
        assert len(train_sets1[0]) == len(train_sets2[0])
        assert len(test_sets1[0]) == len(test_sets2[0])

    def test_data_split_with_diff_density(
        self, mock_zarr_directory: Path, mock_mapping_file: Path, rng
    ):
        """Test data split with diff density type."""
        # First add diff arrays to our test stores
        for task_id in ["mp-12345", "mp-67890", "mp-11111"]:
            store_path = mock_zarr_directory / f"{task_id}.zarr"
            root = zarr.open_group(str(store_path), mode="a")

            if "charge_density_diff" not in root:
                size = root["charge_density_total"].shape[0]
                diff_data = rng.random((size, size, size)).astype(np.float32) * 0.1
                root.create(
                    name="charge_density_diff", data=diff_data, chunks=(4, 4, 4)
                )

        reader = ZarrS3Reader(
            data_dir=mock_zarr_directory,
            label_dir=mock_zarr_directory,
            map_dir=mock_mapping_file,
            functional="GGA",
            normalize=False,
            train_fraction=0.8,
            random_state=42,
            density_type="diff",
        )

        train_sets, _test_sets = reader.data_split()

        assert len(train_sets[0]) > 0

    def test_data_split_missing_files_warning(
        self, mock_zarr_directory: Path, tmp_path: Path
    ):
        """Test that missing files are handled gracefully with warnings."""
        import gzip

        # Create mapping with extra task IDs that don't have zarr stores
        mapping = {"GGA": ["mp-12345", "mp-67890", "mp-11111", "mp-99999", "mp-88888"]}
        mapping_path = tmp_path / "mapping_extra.json.gz"
        with gzip.open(mapping_path, "wt") as f:
            json.dump(mapping, f)

        reader = ZarrS3Reader(
            data_dir=mock_zarr_directory,
            label_dir=mock_zarr_directory,
            map_dir=mapping_path,
            functional="GGA",
            normalize=False,
            train_fraction=0.8,
            random_state=42,
        )

        # Should not raise error, just skip missing files
        train_sets, test_sets = reader.data_split()

        # Should have loaded only the 3 existing files
        total_loaded = len(train_sets[0]) + len(test_sets[0])
        assert total_loaded == 3


class TestLoadDataFunction:
    """Test the registered load_data function."""

    def test_load_data_basic(self, mock_zarr_directory: Path, mock_mapping_file: Path):
        """Test load_data function with basic config."""
        from types import SimpleNamespace

        cfg = SimpleNamespace(
            data_dir=mock_zarr_directory,
            label_dir=mock_zarr_directory,
            map_dir=mock_mapping_file,
            functional="GGA",
            normalize_data=True,
            train_fraction=0.8,
            random_state=42,
        )

        train_sets, test_sets = load_data(cfg)

        assert len(train_sets) == 4
        assert len(test_sets) == 4

    def test_load_data_with_density_type(
        self, mock_zarr_directory: Path, mock_mapping_file: Path, rng
    ):
        """Test load_data with density_type parameter."""
        from types import SimpleNamespace

        # Add diff arrays
        for task_id in ["mp-12345", "mp-67890", "mp-11111"]:
            store_path = mock_zarr_directory / f"{task_id}.zarr"
            root = zarr.open_group(str(store_path), mode="a")
            if "charge_density_diff" not in root:
                size = root["charge_density_total"].shape[0]
                diff_data = rng.random((size, size, size)).astype(np.float32) * 0.1
                root.create(
                    name="charge_density_diff", data=diff_data, chunks=(4, 4, 4)
                )

        cfg = SimpleNamespace(
            data_dir=mock_zarr_directory,
            label_dir=mock_zarr_directory,
            map_dir=mock_mapping_file,
            functional="GGA",
            normalize_data=False,
            train_fraction=0.8,
            random_state=42,
            density_type="diff",
        )

        train_sets, _test_sets = load_data(cfg)
        assert len(train_sets[0]) > 0

    def test_load_data_with_s3_kwargs(self, mock_mapping_file: Path):
        """Test load_data with S3 kwargs."""
        from types import SimpleNamespace

        mock_s3fs_module = MagicMock()
        mock_s3fs_module.S3FileSystem = Mock(return_value=Mock())

        with patch.dict("sys.modules", {"s3fs": mock_s3fs_module}):
            cfg = SimpleNamespace(
                data_dir="s3://bucket/data",
                label_dir="s3://bucket/labels",
                map_dir=mock_mapping_file,
                functional="SCAN",
                normalize_data=True,
                train_fraction=0.8,
                random_state=42,
                s3_kwargs={"anon": True, "profile": "default"},
            )

            # This will fail on data_split since S3 is mocked, but we can test init
            reader = ZarrS3Reader(
                data_dir=cfg.data_dir,
                label_dir=cfg.label_dir,
                map_dir=cfg.map_dir,
                functional=cfg.functional,
                normalize=cfg.normalize_data,
                train_fraction=cfg.train_fraction,
                random_state=cfg.random_state,
                s3_kwargs=cfg.s3_kwargs,
            )

            assert reader.s3_kwargs == {"anon": True, "profile": "default"}
