from __future__ import annotations

import json
from typing import TYPE_CHECKING

import numpy as np
import pytest
import zarr

from electrai.dataloader.dataset import RhoData
from electrai.dataloader.utils import load_zarr

if TYPE_CHECKING:
    from pathlib import Path


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
