from __future__ import annotations

import gzip
from pathlib import Path
from typing import TYPE_CHECKING

import torch
import yaml
from pymatgen.io.vasp.outputs import Chgcar
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from .registry import register_data

if TYPE_CHECKING:
    import numpy as np

dtype_map = {"f32": torch.float32, "f16": torch.float16, "bf16": torch.bfloat16}


class RhoRead:
    def __init__(
        self,
        data_path: Path,
        label_path: Path,
        map_path: Path,
        functional: str,
        train_fraction: float,
        random_seed: int = 42,
    ):
        """
        Parameters
        ----------
        data_path: path of input chgcar or elfcar files.
        label_path: path of label chgcar or elfcar files.
        map_path: path of json file mapping functional to list of task_ids.
        functional: 'GGA', 'GG+U', 'PBEsol', 'SCAN', 'r2SCAN'.
        train_fraction: fraction of the data used for training (0 to 1).
        """
        self.data_path = Path(data_path)
        self.label_path = Path(label_path)
        self.map_path = Path(map_path)
        self.functional = functional
        self.tf = train_fraction
        self.seed = random_seed

    def data_split(self):
        with gzip.open(self.map_path, "rt") as f:
            mapping = yaml.safe_load(f)
        data_list = []

        for task_id in mapping[self.functional]:
            data = (
                self.data_path / f"{task_id}.CHGCAR",
                self.label_path / f"{task_id}.CHGCAR",
            )
            data_list.append(data)
        train_data, test_data = train_test_split(
            data_list, train_size=self.tf, random_state=self.seed
        )
        return train_data, test_data


class RhoData(Dataset):
    def __init__(
        self,
        data: list[tuple[Path, Path]],
        data_precision: str,
        rho_type: str,
        data_augmentation: bool = True,
    ):
        """
        Parameters
        ----------
        data: list of voxel data of length batch_size.
        rho_type: chgcar or elfcar.
        data_size: target size of data.
        label_size: target size of label.
        pyrho_uf: pyrho upsampling factor
        """
        self.data = data
        self.data_precision = data_precision
        self.rho_type = rho_type
        self.da = data_augmentation

    def __len__(self):
        return len(self.data)

    def rotate_x(self, data_in):
        """
        rotate 90 by x axis
        """
        return data_in.transpose(-1, -2).flip(-1)

    def rotate_y(self, data_in):
        return data_in.transpose(-1, -3).flip(-1)

    def rotate_z(self, data_in):
        return data_in.transpose(-2, -3).flip(-2)

    def rand_rotate(self, data_lst):
        rint = torch.randint(0, 3, ()).item()
        if rint == 0:

            def rotate(d):
                return self.rotate_x(d)
        elif rint == 1:

            def rotate(d):
                return self.rotate_y(d)
        else:

            def rotate(d):
                return self.rotate_z(d)

        r = torch.rand(()).item()
        if r < 0.1:
            return data_lst
        elif r < 0.4:
            return [rotate(d) for d in data_lst]
        elif r < 0.7:
            return [rotate(rotate(d)) for d in data_lst]
        else:
            return [rotate(rotate(rotate(d))) for d in data_lst]

    def __getitem__(self, idx: int):
        data_path = self.data[idx][0]
        label_path = self.data[idx][1]

        data = self.read_data(data_path)
        label = self.read_data(label_path)

        if self.rho_type == "chgcar":
            data = data.data["total"] / data.structure.lattice.volume
            label = label.data["total"] / label.structure.lattice.volume
        else:
            data = data.data["total"]
            label = label.data["total"]

        data = torch.tensor(data, dtype=dtype_map[self.data_precision]).unsqueeze(0)
        label = torch.tensor(label, dtype=dtype_map[self.data_precision]).unsqueeze(0)

        if self.da:
            data, label = self.rand_rotate([data, label])
        return data, label

    def read_data(self, data_path: Path) -> np.ndarray:
        """
        Parameters
        ----------
        data_dir: directory of chg or elfcar data.

        Returns
        ----------
        charge density array
        """
        if data_path.name.endswith(".CHGCAR"):
            cden = Chgcar.from_file(data_path)
        else:
            raise ValueError(f"Voxel data format not supported: {data_path}")
        return cden


@register_data("mp")
def load_data(cfg):
    train_set, test_set = RhoRead(
        data_path=cfg.data_path,
        label_path=cfg.label_path,
        map_path=cfg.map_path,
        functional=cfg.functional,
        train_fraction=cfg.train_fraction,
        random_seed=cfg.random_seed,
    ).data_split()

    train_data = RhoData(
        train_set, cfg.data_precision, cfg.rho_type, cfg.data_augmentation
    )

    test_data = RhoData(
        test_set, cfg.data_precision, cfg.rho_type, cfg.data_augmentation
    )
    return train_data, test_data
