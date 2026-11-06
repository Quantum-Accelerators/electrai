from __future__ import annotations

import json
from typing import TYPE_CHECKING

import numpy as np
import pytest
import zarr

from electrai.dataloader.dataset import RhoData
from electrai.dataloader.utils import elf_buggy_to_correct, load_elf_data

if TYPE_CHECKING:
    from pathlib import Path

STRUCTURE = json.dumps({"lattice": {"volume": 8.0}})


@pytest.fixture
def elf_dirs(tmp_path: Path) -> tuple[Path, Path]:
    """Bare (suffix-less) zarr stores as in the r2scan datasets."""
    rng = np.random.default_rng(0)
    grid = (4, 4, 4)
    inp = zarr.open_group(str(tmp_path / "input" / "mp-1"), mode="w")
    inp.attrs["structure"] = STRUCTURE
    inp["charge_density_total"] = rng.random(grid)
    lab = zarr.open_group(str(tmp_path / "label" / "mp-1"), mode="w")
    lab.attrs["structure"] = STRUCTURE
    lab["charge_density_up"] = rng.uniform(0.05, 0.95, grid)
    lab["charge_density_down"] = lab["charge_density_up"][:]
    return tmp_path / "input", tmp_path / "label"


def test_elf_buggy_to_correct_values():
    assert elf_buggy_to_correct(0.5) == pytest.approx(1 / (1 + 2 ** (-4 / 3)))
    x = np.linspace(0.01, 0.99, 50)
    y = elf_buggy_to_correct(x)
    assert np.all(np.diff(y) > 0)
    assert np.all(y > x)
    assert 0.0 < elf_buggy_to_correct(0.0) < 1e-11
    assert 1.0 - 1e-11 < elf_buggy_to_correct(1.0) <= 1.0


def test_load_elf_data_corrects_label_only(elf_dirs):
    inp, lab = elf_dirs
    data0, label0 = load_elf_data(inp, lab, "mp-1", "f32", augmentation=False)
    data1, label1 = load_elf_data(
        inp, lab, "mp-1", "f32", augmentation=False, correct_elf_scale=True
    )
    assert np.array_equal(data0.numpy(), data1.numpy())
    np.testing.assert_allclose(
        label1.numpy(), elf_buggy_to_correct(label0.numpy()), rtol=1e-6
    )


def test_rhodata_passes_flag(elf_dirs, tmp_path: Path):
    inp, lab = elf_dirs
    filelist = tmp_path / "filelist.txt"
    filelist.write_text("mp-1\n")
    raw = RhoData(str(filelist), "f32", False, input_dir=str(inp), label_dir=str(lab))
    fixed = RhoData(
        str(filelist),
        "f32",
        False,
        input_dir=str(inp),
        label_dir=str(lab),
        correct_elf_scale=True,
    )
    assert (fixed[0]["label"] > raw[0]["label"]).all()


def test_flag_rejected_without_explicit_dirs(tmp_path: Path):
    (tmp_path / "data" / "mp-1.zarr").mkdir(parents=True)
    filelist = tmp_path / "mp_filelist.txt"
    filelist.write_text("mp-1\n")
    with pytest.raises(ValueError, match="correct_elf_scale"):
        RhoData(str(filelist), "f32", False, correct_elf_scale=True)
