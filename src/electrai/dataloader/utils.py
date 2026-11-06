from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from pymatgen.io.vasp.outputs import Chgcar

if TYPE_CHECKING:
    import os

dtype_map = {"f32": torch.float32, "f16": torch.float16, "bf16": torch.bfloat16}


def load_numpy_rho(
    root: str | bytes | os.PathLike,
    category: str,
    index: str,
    precision: str,
    augmentation: bool,
    fmt: str = "chgcar",
):
    """
    Load rho data from root directory
    """
    root = Path(root)
    if fmt == "zarr":
        data, label = load_zarr(root, index)
    elif category == "mp":
        data, label = load_chgcar(root, index)
    elif category == "qm9":
        data, label = load_npy(root, index)
    data = torch.tensor(data, dtype=dtype_map[precision])
    label = torch.tensor(label, dtype=dtype_map[precision])
    if augmentation:
        data, label = rand_rotate([data, label])
    return data, label


ELF_SCALE_FIX = 2.0 ** (4.0 / 3.0)


def elf_buggy_to_correct(elf):
    """Map ELF from the buggy VASP spin-polarized scale onto the corrected one.

    Copied from elfe3net src/charge3net/data/dataset.py::elf_buggy_to_correct (behind its
    ``correct_elf_scale: true``), so electrai trains on the same labels. The MP r2SCAN
    labels were produced with VASP 6.1.1, whose src/elf.F computes the
    homogeneous-electron-gas reference D_h with the spin-degenerate formula applied to a
    *spin* density. The correct single-channel reference is larger by 2^(2/3), so the
    stored ELF is systematically underpredicted:

        chi_buggy = 2^(2/3) * chi_correct,   ELF = 1 / (1 + chi^2)
        ELF_correct = 1 / (1 + (1/ELF_buggy - 1) / 2^(4/3))

    Exactly invertible and monotonic, so basins/maxima locations are unchanged; only
    values and absolute thresholds move (label ELF 0.5 -> corrected 0.716).
    """
    elf = np.clip(np.asarray(elf, dtype=np.float64), 1e-12, 1.0 - 1e-12)
    return 1.0 / (1.0 + (1.0 / elf - 1.0) / ELF_SCALE_FIX)


def load_elf_data(
    input_dir: str | bytes | os.PathLike,
    label_dir: str | bytes | os.PathLike,
    index: str,
    precision: str,
    augmentation: bool,
    correct_elf_scale: bool = False,
):
    """Load charge density (input) and ELF (label) from explicit directories.

    Input is normalised by cell volume (e/Å³); label is kept raw since ELF is
    already dimensionless and in [0, 1]. With `correct_elf_scale` the label is
    mapped from the buggy VASP 6.1.1 scale to the corrected one
    (`elf_buggy_to_correct`).
    """
    input_path = _resolve_path(Path(input_dir), index)
    label_path = _resolve_path(Path(label_dir), index)

    data = _read_chgcar_zarr(input_path, normalise_by_volume=True)
    label = _read_chgcar_zarr(label_path, normalise_by_volume=False)
    if correct_elf_scale:
        label = elf_buggy_to_correct(label)

    data = torch.tensor(data, dtype=dtype_map[precision])
    label = torch.tensor(label, dtype=dtype_map[precision])
    if augmentation:
        data, label = rand_rotate([data, label])
    return data, label


def _is_zarr_store(path: Path) -> bool:
    """True if `path` is a zarr store directory (v3 zarr.json or v2 .zgroup/.zarray)."""
    return path.is_dir() and (
        (path / "zarr.json").exists()
        or (path / ".zgroup").exists()
        or (path / ".zarray").exists()
    )


def _resolve_path(directory: Path, index: str) -> Path:
    """Find the file for `index` in `directory`; supports zarr, CHGCAR, ELFCAR.

    Zarr stores may be named ``{index}.zarr`` or, as in the r2scan datasets,
    an extensionless directory ``{index}`` containing a zarr.json.
    """
    for ext in (".zarr", ".CHGCAR", ".ELFCAR"):
        p = directory / f"{index}{ext}"
        if p.exists():
            return p
    bare = directory / index
    if _is_zarr_store(bare):
        return bare
    raise FileNotFoundError(
        f"No .zarr, .CHGCAR or .ELFCAR file found for '{index}' in {directory}"
    )


def _read_chgcar_zarr(path: Path, normalise_by_volume: bool) -> np.ndarray:
    """Read a charge/ELF grid from a zarr store or CHGCAR/ELFCAR file.

    Zarr stores from r2scan datasets carry spin-up and spin-down channels
    separately; charge density is their sum.  ELF channels are identical so
    either suffices.
    """
    if path.suffix == ".zarr" or _is_zarr_store(path):
        import zarr as zarr_lib

        z = zarr_lib.open_group(str(path), mode="r")
        if normalise_by_volume:
            if "charge_density_total" in z:
                arr = np.array(z["charge_density_total"])
            else:
                arr = np.array(z["charge_density_up"]) + np.array(
                    z["charge_density_down"]
                )
            volume = json.loads(z.attrs["structure"])["lattice"]["volume"]
            arr = arr / volume
        else:
            # ELF: spin-up channel alone is in [0, 1]
            arr = np.array(z["charge_density_up"])
        return arr
    else:
        chg = Chgcar.from_file(path)
        arr = chg.data["total"]
        if normalise_by_volume:
            arr = arr / chg.structure.lattice.volume
        return arr


def load_zarr(root: str | bytes | os.PathLike, index: str):
    import zarr

    def _read(path):
        z = zarr.open_group(str(path), mode="r")
        if "structure" not in z.attrs:
            raise KeyError(f"'structure' attribute missing from zarr store at {path}")
        if "charge_density_total" in z:
            arr = np.array(z["charge_density_total"])
        else:
            arr = np.array(z["charge_density_up"]) + np.array(z["charge_density_down"])
        volume = json.loads(z.attrs["structure"])["lattice"]["volume"]
        return arr / volume

    root = Path(root)
    data = _read(_resolve_path(root / "data", index))
    label = _read(_resolve_path(root / "label", index))
    return data, label


def load_chgcar(root: str | bytes | os.PathLike, index: str):
    data = Chgcar.from_file(root / "data" / f"{index}.CHGCAR")
    label = Chgcar.from_file(root / "label" / f"{index}.CHGCAR")
    data = data.data["total"] / data.structure.lattice.volume
    label = label.data["total"] / label.structure.lattice.volume
    return data, label


def load_npy(root: str | bytes | os.PathLike, index: str):
    data_size = np.loadtxt(
        root / "data" / f"dsgdb9nsd_{index:06d}" / "grid_sizes_22.dat", dtype=int
    )
    label_size = np.loadtxt(
        root / "label" / f"dsgdb9nsd_{index:06d}" / "grid_sizes_22.dat", dtype=int
    )
    data = np.load(root / "data" / f"dsgdb9nsd_{index:06d}" / "rho_22.npy").reshape(
        data_size
    )
    label = np.load(root / "label" / f"dsgdb9nsd_{index:06d}" / "rho_22.npy").reshape(
        label_size
    )
    # convert a.u. to e/(A^3)
    factor = 1.88973**3
    return data * factor, label * factor


def rotate_x(data: torch.Tensor):
    """
    rotate 90 by x axis
    """
    return data.transpose(-1, -2).flip(-1)


def rotate_y(data: torch.Tensor):
    return data.transpose(-1, -3).flip(-1)


def rotate_z(data: torch.Tensor):
    return data.transpose(-2, -3).flip(-2)


def rand_rotate(data_lst: list[torch.Tensor]):
    rint = torch.randint(0, 3, ()).item()

    if rint == 0:

        def rotate(d):
            return rotate_x(d)
    elif rint == 1:

        def rotate(d):
            return rotate_y(d)
    else:

        def rotate(d):
            return rotate_z(d)

    r = torch.rand(()).item()
    if r < 0.1:
        return data_lst
    elif r < 0.4:
        return [rotate(d) for d in data_lst]
    elif r < 0.7:
        return [rotate(rotate(d)) for d in data_lst]
    else:
        return [rotate(rotate(rotate(d))) for d in data_lst]
