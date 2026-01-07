from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from .registry import register_data

dtype_map = {"f32": torch.float32, "f16": torch.float16, "bf16": torch.bfloat16}


class RhoRead:
    def __init__(
        self,
        data_path: Path,
        label_path: Path,
        exclude_path: Path,
        train_fraction: float,
        random_state: int = 42,
    ):
        """
        Parameters
        ----------
        data_path: path of input chgcar or elfcar files.
        label_path: path of label chgcar or elfcar files.
        train_fraction: fraction of the data used for training (0 to 1).
        """
        self.data_path = Path(data_path)
        self.label_path = Path(label_path)
        self.exclude_path = Path(exclude_path)
        self.tf = train_fraction
        self.rs = random_state

    def data_split(self):
        data_list = []
        exclude_inds = np.loadtxt(self.exclude_path)
        for mol_id in range(1, 133886):
            if mol_id in exclude_inds:
                continue
            data = (
                self.data_path / f"dsgdb9nsd_{mol_id:06d}" / "rho_22.npy",
                self.label_path / f"dsgdb9nsd_{mol_id:06d}" / "rho_22.npy",
                self.data_path / f"dsgdb9nsd_{mol_id:06d}" / "grid_sizes_22.dat",
                self.label_path / f"dsgdb9nsd_{mol_id:06d}" / "grid_sizes_22.dat",
            )
            data_list.append(data)
        train_data, test_data = train_test_split(
            data_list, train_size=self.tf, random_state=self.rs
        )
        return train_data, test_data


class RhoData(Dataset):
    def __init__(
        self,
        data: list[tuple[Path, Path, Path, Path]],
        data_precision: str,
        data_augmentation=True,
        downsample_data=1,
        downsample_label=1,
        normalize_to_den=False,
    ):
        """
        Parameters
        ----------
        data: list of (input voxel data, label voxel data, input gridsize, label gridsize) of length batch_size.
        """
        self.ds_data = downsample_data
        self.ds_label = downsample_label
        self.da = data_augmentation
        self.data = data
        self.data_precision = data_precision
        self.normalize_to_den = normalize_to_den
        self.rng = np.random.default_rng()

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

    def _normalize(self, data):
        factor = 1.88973**3
        return data * factor

    def __getitem__(self, idx):
        data_path = self.data[idx][0]
        label_path = self.data[idx][1]
        data_gs_path = self.data[idx][2]
        label_gs_path = self.data[idx][3]

        rho1 = torch.tensor(np.load(data_path), dtype=dtype_map[self.data_precision])
        size = np.loadtxt(data_gs_path, dtype=int)
        rho1 = rho1.reshape(1, *size)

        rho2 = torch.tensor(np.load(label_path), dtype=dtype_map[self.data_precision])
        size = np.loadtxt(label_gs_path, dtype=int)
        rho2 = rho2.reshape(1, *size)

        if self.normalize_to_den:
            rho1 = self._normalize(rho1)
            rho2 = self._normalize(rho2)

        if self.da:
            rho1, rho2 = self.rand_rotate([rho1, rho2])

        ds1 = self.ds_data
        ds2 = self.ds_label
        nx, ny, nz = rho1.size()[-3:]
        nx = nx // ds1 * ds1
        ny = ny // ds1 * ds1
        nz = nz // ds1 * ds1
        rho1 = rho1[..., :nx:ds1, :ny:ds1, :nz:ds1]
        nx, ny, nz = rho2.size()[-3:]
        nx = nx // ds1 * ds1
        ny = ny // ds1 * ds1
        nz = nz // ds1 * ds1
        rho2 = rho2[..., :nx:ds2, :ny:ds2, :nz:ds2]

        return (rho1, rho2)


@register_data("qm9")
def load_data(cfg):
    train_set, test_set = RhoRead(
        data_path=cfg.data_path,
        label_path=cfg.label_path,
        exclude_path=cfg.exclude_path,
        train_fraction=cfg.train_fraction,
        random_state=cfg.random_state,
    ).data_split()

    train_data = RhoData(
        train_set,
        cfg.data_precision,
        cfg.data_augmentation,
        cfg.downsample_data,
        cfg.downsample_label,
        cfg.normalize_to_den,
    )

    test_data = RhoData(
        test_set,
        cfg.data_precision,
        cfg.data_augmentation,
        cfg.downsample_data,
        cfg.downsample_label,
        cfg.normalize_to_den,
    )
    return train_data, test_data
