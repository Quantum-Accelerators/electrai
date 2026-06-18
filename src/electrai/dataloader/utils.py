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


def load_zarr(root: str | bytes | os.PathLike, index: str):
    """Read a zarr store as either a directory tree (`<id>.zarr/`) or a single
    packed zip (`<id>.zarr.zip`). Volumes capped by inode count (e.g. Modal)
    benefit greatly from the packed form — one inode per store instead of ~8.
    """
    import zarr

    def _read(zarr_dir: Path, zarr_zip: Path):
        store = None
        if zarr_zip.exists():
            store = zarr.storage.ZipStore(str(zarr_zip), mode="r")
            z = zarr.open_group(store, mode="r")
        elif zarr_dir.exists():
            z = zarr.open_group(str(zarr_dir), mode="r")
        else:
            raise FileNotFoundError(f"No zarr store at {zarr_zip} or {zarr_dir}")
        try:
            if "structure" not in z.attrs:
                raise KeyError(
                    f"'structure' attribute missing from zarr store at "
                    f"{zarr_zip if zarr_zip.exists() else zarr_dir}"
                )
            arr = np.array(z["charge_density_total"])
            volume = json.loads(z.attrs["structure"])["lattice"]["volume"]
            return arr / volume
        finally:
            if store is not None:
                store.close()

    root = Path(root)
    data = _read(root / "data" / f"{index}.zarr", root / "data" / f"{index}.zarr.zip")
    label = _read(
        root / "label" / f"{index}.zarr", root / "label" / f"{index}.zarr.zip"
    )
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
