from __future__ import annotations

import json
from typing import TYPE_CHECKING

import numpy as np
import pytest
import zarr
from pymatgen.io.vasp.outputs import Chgcar  # type: ignore[import-not-found]

from electrai.zarr_conversion.convert_to_zarr import convert_chgcar_to_zarr, load_chgcar
from electrai.zarr_conversion.zarr_writer import write_chgcar_to_zarr

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

    convert_chgcar_to_zarr(chgcar_path, output_path)

    root = zarr.open_group(str(output_path), mode="r")
    charge_total = root["charge_density_total"]
    assert isinstance(charge_total, zarr.Array)
    total_array = np.asarray(charge_total[:])
    np.testing.assert_allclose(total_array, total_density)

    metadata = json.loads(str(root.attrs["metadata"]))
    assert metadata["task_id"] == "mp-test"

    # Non-spin-polarized inputs should not have a diff array.
    assert "charge_density_diff" not in root
    assert root.attrs["is_spin_polarized"] is False
    assert root.attrs["is_soc"] is False
    # data_aug round-trips as JSON (may be empty for a minimal fixture).
    assert isinstance(json.loads(str(root.attrs["data_aug"])), dict)


def test_convert_chgcar_to_zarr_writes_spin_polarized_components(
    tmp_path: Path,
) -> None:
    pymatgen_core = pytest.importorskip("pymatgen.core")
    pymatgen_outputs = pytest.importorskip("pymatgen.io.vasp.outputs")

    lattice = pymatgen_core.Lattice.cubic(3.0)
    structure = pymatgen_core.Structure(lattice, ["Li"], [[0.0, 0.0, 0.0]])
    total_density = np.arange(8, dtype=float).reshape((2, 2, 2))
    diff_density = -np.arange(8, dtype=float).reshape((2, 2, 2))
    data_aug = {
        "total": ["augmentation line 1\n", "augmentation line 2\n"],
        "diff": ["diff aug line\n"],
    }

    chgcar = pymatgen_outputs.Chgcar(
        structure, {"total": total_density, "diff": diff_density}, data_aug=data_aug
    )
    chgcar.task_id = "mp-spin"

    output_path = tmp_path / "mp-spin.zarr"
    write_chgcar_to_zarr(chgcar, output_path, chunks=(2, 2, 2), chunks_diff=(1, 2, 2))

    root = zarr.open_group(str(output_path), mode="r")
    np.testing.assert_allclose(
        np.asarray(root["charge_density_total"][:]), total_density
    )
    np.testing.assert_allclose(np.asarray(root["charge_density_diff"][:]), diff_density)

    # Total and diff use independent chunks.
    assert tuple(root["charge_density_total"].chunks) == (2, 2, 2)
    assert tuple(root["charge_density_diff"].chunks) == (1, 2, 2)

    assert root.attrs["is_spin_polarized"] is True

    aug = json.loads(str(root.attrs["data_aug"]))
    assert set(aug) == {"total", "diff"}
    assert aug["total"] == data_aug["total"]
    assert aug["diff"] == data_aug["diff"]


def test_write_diff_false_skips_diff_arrays(tmp_path: Path) -> None:
    pymatgen_core = pytest.importorskip("pymatgen.core")
    pymatgen_outputs = pytest.importorskip("pymatgen.io.vasp.outputs")

    lattice = pymatgen_core.Lattice.cubic(3.0)
    structure = pymatgen_core.Structure(lattice, ["Li"], [[0.0, 0.0, 0.0]])
    total_density = np.arange(8, dtype=float).reshape((2, 2, 2))
    diff_density = -np.arange(8, dtype=float).reshape((2, 2, 2))

    chgcar = pymatgen_outputs.Chgcar(
        structure, {"total": total_density, "diff": diff_density}
    )
    chgcar.task_id = "mp-nodiff"

    output_path = tmp_path / "mp-nodiff.zarr"
    write_chgcar_to_zarr(chgcar, output_path, write_diff=False)

    root = zarr.open_group(str(output_path), mode="r")
    assert "charge_density_total" in root
    assert "charge_density_diff" not in root
    # The flag only gates array writes; is_spin_polarized still reflects input.
    assert root.attrs["is_spin_polarized"] is True
