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
):
    """
    Load rho data from root directory
    """
    root = Path(root)
    if category == "mp":
        data, label = load_chgcar(root, index)
    elif category == "qm9":
        data, label = load_npy(root, index)
    data = torch.tensor(data, dtype=dtype_map[precision])
    label = torch.tensor(label, dtype=dtype_map[precision])
    if augmentation:
        data, label = rand_rotate([data, label])
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
