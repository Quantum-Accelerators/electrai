from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from .registry import register_data

dtype_map = {"f32": torch.float32, "f16": torch.float16, "bf16": torch.bfloat16}
num_samples = 133886
conversion_factor = 1.88973**3  # Bohr^3 to Angstrom^3


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
        exclude_inds = set(np.loadtxt(self.exclude_path))
        for mol_id in range(1, num_samples):
            if mol_id in exclude_inds:
                continue
            mol_dir = f"dsgdb9nsd_{mol_id:06d}"
            data = (
                self.data_path / mol_dir / "rho_22.npy",  # flattened array
                self.label_path / mol_dir / "rho_22.npy",  # flattened array
                self.data_path / mol_dir / "grid_sizes_22.dat",
                self.label_path / mol_dir / "grid_sizes_22.dat",
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

    def unit_conversion(self, data):
        factor = conversion_factor
        return data * factor

    def __getitem__(self, idx):
        data_path, label_path, data_gs_path, label_gs_path = self.data[idx]

        rho_input = torch.tensor(
            np.load(data_path), dtype=dtype_map[self.data_precision]
        )
        size = np.loadtxt(data_gs_path, dtype=int)
        rho_input = rho_input.reshape(1, *size)

        rho_label = torch.tensor(
            np.load(label_path), dtype=dtype_map[self.data_precision]
        )
        size = np.loadtxt(label_gs_path, dtype=int)
        rho_label = rho_label.reshape(1, *size)

        rho_input = self.unit_conversion(rho_input)
        rho_label = self.unit_conversion(rho_label)

        if self.da:
            rho_input, rho_label = self.rand_rotate([rho_input, rho_label])

        ds_input = self.ds_data
        ds_label = self.ds_label
        nx, ny, nz = rho_input.size()[-3:]
        nx = nx // ds_input * ds_input
        ny = ny // ds_input * ds_input
        nz = nz // ds_input * ds_input
        rho_input = rho_input[..., :nx:ds_input, :ny:ds_input, :nz:ds_input]
        nx, ny, nz = rho_label.size()[-3:]
        nx = nx // ds_input * ds_input
        ny = ny // ds_input * ds_input
        nz = nz // ds_input * ds_input
        rho_label = rho_label[..., :nx:ds_label, :ny:ds_label, :nz:ds_label]

        return (rho_input, rho_label)


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
    )

    test_data = RhoData(
        test_set,
        cfg.data_precision,
        cfg.data_augmentation,
        cfg.downsample_data,
        cfg.downsample_label,
    )
    return train_data, test_data
