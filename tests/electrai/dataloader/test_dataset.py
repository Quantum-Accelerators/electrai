from __future__ import annotations

import dataclasses
import json
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
import zarr
from torch.utils.data import Dataset

from electrai.dataloader.collate import collate_fn
from electrai.dataloader.dataset import AddDatasetID, DatasetSpec, RhoData, RhoRead
from electrai.dataloader.utils import load_zarr

if TYPE_CHECKING:
    from pathlib import Path


# --- Fixtures ---


@pytest.fixture
def rng():
    return np.random.default_rng(seed=0)


@pytest.fixture
def zarr_root(tmp_path: Path, rng) -> Path:
    """Create a minimal dataset root with data/ and label/ zarr stores."""
    structure = {"lattice": {"volume": 125.0}}
    for split in ("data", "label"):
        store_path = tmp_path / split / "mp-1.zarr"
        z = zarr.open_group(str(store_path), mode="w")
        charge = rng.random((8, 8, 8)).astype(np.float32)
        z.create(name="charge_density_total", data=charge)
        z.attrs["structure"] = json.dumps(structure)
    return tmp_path


@pytest.fixture
def filelist(tmp_path: Path, zarr_root: Path) -> Path:
    filelist_path = zarr_root / "mp_filelist.txt"
    filelist_path.write_text("mp-1\n")
    return filelist_path


class SimpleDataset(Dataset):
    def __init__(self, n: int):
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return {
            "data": torch.tensor([float(idx)]),
            "label": torch.tensor([float(idx)]),
            "index": str(idx),
        }


# --- TestLoadZarr ---


class TestLoadZarr:
    def test_returns_arrays_divided_by_volume(self, zarr_root: Path):
        data, label = load_zarr(zarr_root, "mp-1")
        assert isinstance(data, np.ndarray)
        assert isinstance(label, np.ndarray)
        # Volume is 125.0; all values should be in [0, 1/125]
        assert data.max() <= 1.0 / 125.0 + 1e-6

    def test_missing_structure_attr_raises(self, tmp_path: Path, rng):
        store_path = tmp_path / "data" / "bad.zarr"
        z = zarr.open_group(str(store_path), mode="w")
        z.create(
            name="charge_density_total", data=rng.random((4, 4, 4)).astype(np.float32)
        )
        with pytest.raises(KeyError, match="structure"):
            load_zarr(tmp_path, "bad")


# --- TestRhoDataFormatDetection ---


class TestRhoDataFormatDetection:
    def test_detects_zarr_format(self, filelist: Path):
        dataset = RhoData(str(filelist), precision="f32", augmentation=False)
        assert dataset.fmt == "zarr"

    def test_detects_chgcar_format(self, tmp_path: Path):
        chgcar_path = tmp_path / "data" / "mp-2.CHGCAR"
        chgcar_path.parent.mkdir(parents=True)
        chgcar_path.touch()
        filelist_path = tmp_path / "mp_filelist.txt"
        filelist_path.write_text("mp-2\n")
        dataset = RhoData(str(filelist_path), precision="f32", augmentation=False)
        assert dataset.fmt == "chgcar"

    def test_empty_filelist_raises(self, tmp_path: Path):
        filelist_path = tmp_path / "mp_filelist.txt"
        filelist_path.write_text("")
        with pytest.raises(ValueError, match="empty"):
            RhoData(str(filelist_path), precision="f32", augmentation=False)

    def test_unknown_format_raises(self, tmp_path: Path):
        filelist_path = tmp_path / "mp_filelist.txt"
        filelist_path.write_text("mp-99\n")
        with pytest.raises(ValueError, match=r"No \.zarr or \.CHGCAR"):
            RhoData(str(filelist_path), precision="f32", augmentation=False)

    def test_getitem_zarr(self, filelist: Path):
        dataset = RhoData(str(filelist), precision="f32", augmentation=False)
        item = dataset[0]
        assert "data" in item
        assert "label" in item
        assert "index" in item
        assert item["data"].shape[0] == 1  # unsqueeze adds channel dim


# --- TestAddDatasetID ---


class TestAddDatasetID:
    def test_len_preserved(self):
        base = SimpleDataset(10)
        wrapped = AddDatasetID(base, functional_id=3)
        assert len(wrapped) == 10

    def test_dataset_id_added(self):
        base = SimpleDataset(5)
        wrapped = AddDatasetID(base, functional_id=7)
        sample = wrapped[0]
        assert sample["Dataset_ID"] == 7

    def test_dataset_id_is_int(self):
        base = SimpleDataset(3)
        wrapped = AddDatasetID(base, functional_id="2")
        assert isinstance(wrapped[0]["Dataset_ID"], int)

    def test_original_keys_preserved(self):
        base = SimpleDataset(3)
        wrapped = AddDatasetID(base, functional_id=1)
        sample = wrapped[0]
        assert "data" in sample
        assert "label" in sample
        assert "index" in sample

    def test_custom_key(self):
        base = SimpleDataset(3)
        wrapped = AddDatasetID(base, functional_id=5, key="src_id")
        assert "src_id" in wrapped[0]
        assert wrapped[0]["src_id"] == 5


# --- TestCollateFn ---


class TestCollateFn:
    def test_uniform_shapes_uses_default_collate(self):
        batch = [
            {
                "data": torch.zeros(4, 4, 4),
                "label": torch.ones(4, 4, 4),
                "index": "a",
                "Dataset_ID": 0,
            },
            {
                "data": torch.zeros(4, 4, 4),
                "label": torch.ones(4, 4, 4),
                "index": "b",
                "Dataset_ID": 0,
            },
        ]
        result = collate_fn(batch)
        assert isinstance(result, dict)
        assert result["data"].shape == (2, 4, 4, 4)

    def test_mismatched_shapes_fallback_returns_dict(self):
        batch = [
            {
                "data": torch.zeros(4, 4, 4),
                "label": torch.ones(4, 4, 4),
                "index": "a",
                "Dataset_ID": 0,
            },
            {
                "data": torch.zeros(8, 8, 8),
                "label": torch.ones(8, 8, 8),
                "index": "b",
                "Dataset_ID": 1,
            },
        ]
        result = collate_fn(batch)
        assert isinstance(result, dict)
        assert len(result["data"]) == 2
        assert len(result["label"]) == 2

    def test_fallback_preserves_all_keys(self):
        batch = [
            {
                "data": torch.zeros(4, 4, 4),
                "label": torch.ones(4, 4, 4),
                "index": "a",
                "Dataset_ID": 0,
            },
            {
                "data": torch.zeros(8, 8, 8),
                "label": torch.ones(8, 8, 8),
                "index": "b",
                "Dataset_ID": 1,
            },
        ]
        result = collate_fn(batch)
        assert set(result.keys()) == {"data", "label", "index", "Dataset_ID"}

    def test_fallback_correct_values(self):
        batch = [
            {
                "data": torch.zeros(2, 2, 2),
                "label": torch.ones(2, 2, 2),
                "index": "x",
                "Dataset_ID": 3,
            },
            {
                "data": torch.zeros(3, 3, 3),
                "label": torch.ones(3, 3, 3),
                "index": "y",
                "Dataset_ID": 5,
            },
        ]
        result = collate_fn(batch)
        assert result["index"] == ["x", "y"]
        assert result["Dataset_ID"] == [3, 5]


# --- TestDatasetSpec ---


class TestDatasetSpec:
    def test_required_root(self):
        spec = DatasetSpec(root="/some/path")
        assert spec.root == "/some/path"

    def test_defaults_are_none(self):
        spec = DatasetSpec(root="/p")
        assert spec.split_file is None
        assert spec.val_frac is None
        assert spec.dataset_id is None

    def test_frozen(self):
        spec = DatasetSpec(root="/p")
        with pytest.raises(dataclasses.FrozenInstanceError):
            spec.root = "/other"


# --- TestRhoReadInit ---


class TestRhoReadInit:
    def test_empty_datasets_raises(self):
        with pytest.raises(ValueError, match="at least one"):
            RhoRead(datasets=[])

    def test_none_datasets_raises(self):
        with pytest.raises(ValueError, match="at least one"):
            RhoRead(datasets=None)

    def test_auto_assigns_dataset_id(self):
        reader = RhoRead(
            datasets=[{"root": "/a/filelist.txt"}, {"root": "/b/filelist.txt"}]
        )
        assert reader.specs[0].dataset_id == 0
        assert reader.specs[1].dataset_id == 1

    def test_explicit_dataset_id_preserved(self):
        reader = RhoRead(datasets=[{"root": "/a/filelist.txt", "dataset_id": 42}])
        assert reader.specs[0].dataset_id == 42

    def test_default_val_frac_applied(self):
        reader = RhoRead(datasets=[{"root": "/a/filelist.txt"}], default_val_frac=0.1)
        assert reader.specs[0].val_frac == 0.1

    def test_per_spec_val_frac_overrides_default(self):
        reader = RhoRead(
            datasets=[{"root": "/a/filelist.txt", "val_frac": 0.2}],
            default_val_frac=0.1,
        )
        assert reader.specs[0].val_frac == 0.2

    def test_default_split_file_applied(self):
        reader = RhoRead(
            datasets=[{"root": "/a/filelist.txt"}],
            default_split_file="/default/split.json",
        )
        assert reader.specs[0].split_file == "/default/split.json"

    def test_per_spec_split_file_overrides_default(self):
        reader = RhoRead(
            datasets=[{"root": "/a/filelist.txt", "split_file": "/custom/split.json"}],
            default_split_file="/default/split.json",
        )
        assert reader.specs[0].split_file == "/custom/split.json"


# --- TestRhoReadSetup ---


class TestRhoReadSetup:
    def _make_fake_splits(self, n: int = 5):
        ds = SimpleDataset(n)
        from torch.utils.data import Subset

        train = Subset(ds, list(range(4)))
        val = Subset(ds, [4])
        return {"train": train, "validation": val}

    @patch("electrai.dataloader.dataset.split_data")
    @patch("electrai.dataloader.dataset.RhoData")
    def test_fit_stage_builds_train_and_val(self, mock_rho, mock_split):
        mock_rho.return_value = MagicMock(spec=Dataset)
        mock_split.return_value = self._make_fake_splits()

        reader = RhoRead(datasets=[{"root": "/a/filelist.txt"}])
        reader.setup("fit")

        assert reader.train_set is not None
        assert reader.val_set is not None
        assert reader.test_set is None

    @patch("electrai.dataloader.dataset.split_data")
    @patch("electrai.dataloader.dataset.RhoData")
    def test_test_stage_does_not_build_train_or_val(self, mock_rho, mock_split):
        mock_rho.return_value = MagicMock(spec=Dataset)
        splits = self._make_fake_splits()
        splits["test"] = splits["train"]
        mock_split.return_value = splits

        reader = RhoRead(
            datasets=[{"root": "/a/filelist.txt", "split_file": "/split.json"}]
        )
        reader.setup("test")

        assert reader.test_set is not None
        assert reader.train_set is None
        assert reader.val_set is None

    @patch("electrai.dataloader.dataset.split_data")
    @patch("electrai.dataloader.dataset.RhoData")
    def test_multi_dataset_concat(self, mock_rho, mock_split):
        from torch.utils.data import ConcatDataset

        mock_rho.return_value = MagicMock(spec=Dataset)
        mock_split.return_value = self._make_fake_splits()

        reader = RhoRead(
            datasets=[{"root": "/a/filelist.txt"}, {"root": "/b/filelist.txt"}]
        )
        reader.setup("fit")

        assert isinstance(reader.train_set, ConcatDataset)
        assert isinstance(reader.val_set, ConcatDataset)

    @patch("electrai.dataloader.dataset.split_data")
    @patch("electrai.dataloader.dataset.RhoData")
    def test_single_dataset_no_concat(self, mock_rho, mock_split):
        from torch.utils.data import ConcatDataset

        mock_rho.return_value = MagicMock(spec=Dataset)
        mock_split.return_value = self._make_fake_splits()

        reader = RhoRead(datasets=[{"root": "/a/filelist.txt"}])
        reader.setup("fit")

        assert not isinstance(reader.train_set, ConcatDataset)

    @patch("electrai.dataloader.dataset.split_data")
    @patch("electrai.dataloader.dataset.RhoData")
    def test_dataset_id_propagated(self, mock_rho, mock_split):
        mock_rho.return_value = MagicMock(spec=Dataset)
        mock_split.return_value = self._make_fake_splits()

        reader = RhoRead(datasets=[{"root": "/a/filelist.txt", "dataset_id": 99}])
        reader.setup("fit")

        sample = reader.train_set[0]
        assert sample["Dataset_ID"] == 99
