from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from monty.serialization import loadfn
from pymatgen.io.vasp.outputs import Chgcar
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from .registry import register_data

dtype_map = {"f32": torch.float32, "f16": torch.float16, "bf16": torch.bfloat16}


class RhoRead:
    def __init__(
        self,
        data_path: Path,
        label_path: Path,
        map_path: Path,
        functional: str,
        train_fraction: float,
        random_state: int = 42,
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
        self.rs = random_state

    def data_split(self):
        mapping = loadfn(self.map_path)
        data_list = []

        for task_id in mapping[self.functional]:
            data = (
                self.data_path / f"{task_id}.CHGCAR",
                self.label_path / f"{task_id}.CHGCAR",
            )
            data_list.append(data)
        train_data, test_data = train_test_split(
            data_list, train_size=self.tf, random_state=self.rs
        )
        return train_data, test_data


class RhoData(Dataset):
    def __init__(
        self,
        data: list[tuple[Path, Path]],
        data_precision: str,
        rho_type: str,
        data_augmentation: bool = True,
        random_state: int = 42,
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
        self.rng = np.random.default_rng(random_state)

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
        rint = self.rng.integers(0, 3)
        if rint == 0:

            def rotate(d):
                return self.rotate_x(d)
        elif rint == 1:

            def rotate(d):
                return self.rotate_y(d)
        else:

            def rotate(d):
                return self.rotate_z(d)

        r = self.rng.random()
        if r < 0.1:
            return data_lst
        elif r < 0.4:
            return [rotate(d) for d in data_lst]
        elif r < 0.7:
            return [rotate(rotate(d)) for d in data_lst]
        else:
            return [rotate(rotate(rotate(d))) for d in data_lst]

    def __getitem__(self, idx: int):
        data = self.read_data(self.data[idx][0])
        label = self.read_data(self.data[idx][1])

        if self.rho_type == "chgcar":
            data = data.data["total"] / np.prod(data.data["total"].shape)
            label = label.data["total"] / np.prod(label.data["total"].shape)
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
        random_state=cfg.random_state,
    ).data_split()

    train_data = RhoData(
        train_set,
        cfg.data_precision,
        cfg.rho_type,
        cfg.data_augmentation,
        cfg.random_state,
    )

    test_data = RhoData(
        test_set,
        cfg.data_precision,
        cfg.rho_type,
        cfg.data_augmentation,
        cfg.random_state,
    )
    return train_data, test_data
