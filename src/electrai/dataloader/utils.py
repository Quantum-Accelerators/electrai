from __future__ import annotations

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
    downsample_data: int,
    downsample_label: int,
):
    """
    Load rho data from root directory
    """
    root = Path(root)
    cond = None
    if category == "mp":
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
            data = data[
                ..., :nx, :ny, :nz
            ]  # [..., :nx:ds_data, :ny:ds_data, :nz:ds_data]
            nx, ny, nz = label.shape[-3:]
            nx = nx // ds_data * ds_data
            ny = ny // ds_data * ds_data
            nz = nz // ds_data * ds_data
            label = label[
                ..., :nx, :ny, :nz
            ]  # [..., :nx:ds_label, :ny:ds_label, :nz:ds_label]
    data = torch.tensor(data, dtype=dtype_map[precision])
    label = torch.tensor(label, dtype=dtype_map[precision])
    if cond is not None:
        cond = torch.tensor(cond, dtype=dtype_map[precision])
    if augmentation:
        data, label = rand_rotate([data, label])
    return data, label, cond


def load_chgcar(root: str | bytes | os.PathLike, index: str):
    data_chg = Chgcar.from_file(root / "data" / f"{index}.CHGCAR")
    label_chg = Chgcar.from_file(root / "label" / f"{index}.CHGCAR")
    data = data_chg.data["total"] / data_chg.structure.lattice.volume
    label = label_chg.data["total"] / label_chg.structure.lattice.volume

    # --- conditioning vector from the (data) lattice ---
    lat = data_chg.structure.lattice
    a, b, c = lat.abc
    # angles are in degrees in pymatgen
    ca = np.cos(np.deg2rad(lat.alpha))
    cb = np.cos(np.deg2rad(lat.beta))
    cg = np.cos(np.deg2rad(lat.gamma))

    cond = np.array([a / 10.0, b / 10.0, c / 10.0, ca, cb, cg], dtype=np.float32)

    return data, label, cond


def load_npy(root: str | bytes | os.PathLike, index: str):
    mol_dir = f"dsgdb9nsd_{int(index):06d}"
    data_size = np.loadtxt(
        root / "data" / mol_dir / "grid_sizes_22.dat", dtype=int
    )  # original grid size
    label_size = np.loadtxt(
        root / "label" / mol_dir / "grid_sizes_22.dat", dtype=int
    )  # original grid size
    data = np.load(root / "data" / mol_dir / "rho_22.npy").reshape(
        data_size
    )  # flattened 1D data
    label = np.load(root / "label" / mol_dir / "rho_22.npy").reshape(
        label_size
    )  # flattened 1D data
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
