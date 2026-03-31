from __future__ import annotations

import re
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
    elements = {"Li", "Na", "K", "Mg", "Ca", "Al", "Ga", "C", "Si", "N", "P", "O", "S"}
    nelecs = {
        "Li": 3,
        "Na": 7,
        "K": 9,
        "Mg": 8,
        "Ca": 10,
        "Al": 3,
        "Ga": 13,
        "C": 4,
        "Si": 4,
        "N": 5,
        "P": 5,
        "O": 6,
        "S": 6,
    }
    idx_to_element = dict(enumerate(sorted(elements)))
    root = Path(root)
    label_chg = Chgcar.from_file(root / "label" / f"{index}.CHGCAR")
    voxel_volume = label_chg.structure.lattice.volume / np.prod(
        label_chg.data["total"].shape
    )
    label = label_chg.data["total"] / label_chg.structure.lattice.volume
    files = Path.glob.glob(str(root / "data" / f"{index}_*.CHGCAR"))
    density = np.zeros((13, *label.shape), dtype=float)
    presence = np.zeros((13, *label.shape), dtype=float)
    chg_sum = np.zeros(label.shape, dtype=float)
    for f in files:
        path = Path(f)
        chg = Chgcar.from_file(path)
        m = re.match(rf"^{re.escape(index)}_(\d+)\.CHGCAR$", path.name)
        k = int(m.group(1))
        el = idx_to_element[k]
        n_atoms = int(chg.structure.composition[el])
        voxel_volume = chg.structure.lattice.volume / np.prod(chg.data["total"].shape)
        chg_data = chg.data["total"] / chg.structure.lattice.volume
        t = n_atoms * (nelecs[el] / voxel_volume) / np.sum(chg_data)
        chg_data *= t
        chg_sum += chg_data
        channel_sum = np.sum(chg_data)
        if channel_sum > 0:
            density[k] = chg_data / channel_sum
            presence[k] = 1.0
    stacked = np.concatenate([density, presence, chg_sum[None]], axis=0)
    return stacked, label


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
