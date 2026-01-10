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
    downsample_data: int = 1,
    downsample_label: int = 1,
):
    """
    Load rho data from root directory
    """
    root = Path(root)
    cond = None
    if fmt == "zarr":
        data, label = load_zarr(root, index)
    elif category == "mp":
        data, label, cond = load_chgcar(root, index)
    elif category == "qm9":
        data, label = load_npy(root, index)
        if downsample_data != 1:
            ds_data = downsample_data
            ds_label = downsample_label  # noqa
            nx, ny, nz = data.shape[-3:]
            nx = nx // ds_data * ds_data
            ny = ny // ds_data * ds_data
            nz = nz // ds_data * ds_data
            data = data[..., :nx, :ny, :nz]
            nx, ny, nz = label.shape[-3:]
            nx = nx // ds_data * ds_data
            ny = ny // ds_data * ds_data
            nz = nz // ds_data * ds_data
            label = label[..., :nx, :ny, :nz]
    data = torch.tensor(data, dtype=dtype_map[precision])
    label = torch.tensor(label, dtype=dtype_map[precision])
    if cond is not None:
        cond = torch.tensor(cond, dtype=dtype_map[precision])
    if augmentation:
        data, label = rand_rotate([data, label])
    return data, label, cond


def load_zarr(root: str | bytes | os.PathLike, index: str):
    import zarr

    def _read(path):
        z = zarr.open(str(path))
        arr = np.array(z["charge_density_total"], dtype=np.float64)
        volume = json.loads(z.attrs["structure"])["lattice"]["volume"]
        return arr / volume

    data = _read(Path(root) / "data" / f"{index}.zarr")
    label = _read(Path(root) / "label" / f"{index}.zarr")
    return data, label


def load_chgcar(root: str | bytes | os.PathLike, index: str):
    data_chg = Chgcar.from_file(root / "data" / f"{index}.CHGCAR")
    label_chg = Chgcar.from_file(root / "label" / f"{index}.CHGCAR")
    data = data_chg.data["total"] / data_chg.structure.lattice.volume
    label = label_chg.data["total"] / label_chg.structure.lattice.volume

    # Gram matrix upper triangle: [a·a, a·b, a·c, b·b, b·c, c·c] in units of Å²
    # Orientation-invariant and captures lengths + angles in a physically natural form.
    # Divided by 100 (Å²) to keep values in a ~0.1–4 range for typical materials.
    mat = data_chg.structure.lattice.matrix  # (3, 3), rows are lattice vectors
    gram = mat @ mat.T
    cond = (gram[np.triu_indices(3)] / 100.0).astype(np.float32)

    return data, label, cond


def load_npy(root: str | bytes | os.PathLike, index: str):
    mol_dir = f"dsgdb9nsd_{int(index):06d}"
    data_size = np.loadtxt(
        root / "data" / mol_dir / "grid_sizes_22.dat", dtype=int
    )
    label_size = np.loadtxt(
        root / "label" / mol_dir / "grid_sizes_22.dat", dtype=int
    )
    data = np.load(root / "data" / mol_dir / "rho_22.npy").reshape(data_size)
    label = np.load(root / "label" / mol_dir / "rho_22.npy").reshape(label_size)
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
