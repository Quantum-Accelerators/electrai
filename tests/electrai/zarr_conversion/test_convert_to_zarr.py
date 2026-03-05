from __future__ import annotations

import json
from typing import TYPE_CHECKING

import numpy as np
import pytest
import zarr
from pymatgen.io.vasp.outputs import Chgcar  # type: ignore[import-not-found]

from electrai.zarr_conversion.convert_to_zarr import convert_chgcar_to_zarr, load_chgcar

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def dummy_chgcar(tmp_path: Path) -> tuple[Path, np.ndarray]:
    pymatgen_core = pytest.importorskip("pymatgen.core")
    pymatgen_outputs = pytest.importorskip("pymatgen.io.vasp.outputs")

    lattice = pymatgen_core.Lattice.cubic(3.0)
    structure = pymatgen_core.Structure(lattice, ["Li"], [[0.0, 0.0, 0.0]])
    total_density = np.arange(8, dtype=float).reshape((2, 2, 2))

    chgcar = pymatgen_outputs.Chgcar(structure, {"total": total_density})
    chgcar_path = tmp_path / "mp-test.CHGCAR"
    chgcar.write_file(chgcar_path)

    return chgcar_path, total_density


def test_load_chgcar_from_native_chgcar(dummy_chgcar: tuple[Path, np.ndarray]) -> None:
    chgcar_path, total_density = dummy_chgcar

    data = load_chgcar(chgcar_path)

    assert isinstance(data, Chgcar)
    assert data.task_id == "mp-test"
    total = np.asarray(data.data["total"])
    assert total.shape == total_density.shape
    np.testing.assert_allclose(total, total_density)
    assert data.structure.lattice.a > 0


def test_convert_chgcar_to_zarr_creates_expected_store(
    tmp_path: Path, dummy_chgcar: tuple[Path, np.ndarray]
) -> None:
    chgcar_path, total_density = dummy_chgcar
    output_path = tmp_path / "mp-test.zarr"

    convert_chgcar_to_zarr(chgcar_path, output_path, write_diff=False)

    root = zarr.open_group(str(output_path), mode="r")
    charge_total = root["charge_density_total"]
    assert isinstance(charge_total, zarr.Array)
    total_array = np.asarray(charge_total[:])
    np.testing.assert_allclose(total_array, total_density)

    metadata = json.loads(str(root.attrs["metadata"]))
    assert metadata["task_id"] == "mp-test"
