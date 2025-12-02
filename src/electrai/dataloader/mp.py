from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from monty.serialization import loadfn
from pyrho.charge_density import ChargeDensity
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
        patch_size: int | None = None,
    ):
        """
        Parameters
        ----------
        data: list of voxel data of length batch_size.
        rho_type: chgcar or elfcar.
        data_augmentation: whether to apply random rotations.
        random_state: seed for reproducibility.
        patch_size: spatial patch size for training (None = full volume).
        """
        self.data = data
        self.data_precision = data_precision
        self.rho_type = rho_type
        self.da = data_augmentation
        self.rng = np.random.default_rng(random_state)
        self.patch_size = patch_size

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

    def extract_patch(
        self, data: torch.Tensor, label: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Extract random patch with periodic wrapping.

        Uses torch.roll to shift the volume by a random offset (which wraps
        around due to periodicity), then extracts a fixed-size patch from
        the origin. This correctly handles periodic boundary conditions.
        """
        if self.patch_size is None:
            return data, label

        D, H, W = data.shape[-3:]
        ps = self.patch_size

        # Random shift (handles periodicity via roll)
        shift_d = int(self.rng.integers(0, D))
        shift_h = int(self.rng.integers(0, H))
        shift_w = int(self.rng.integers(0, W))

        data = torch.roll(data, shifts=(shift_d, shift_h, shift_w), dims=(-3, -2, -1))
        label = torch.roll(label, shifts=(shift_d, shift_h, shift_w), dims=(-3, -2, -1))

        # Extract patch from origin
        data = data[..., :ps, :ps, :ps]
        label = label[..., :ps, :ps, :ps]

        return data, label

    def __getitem__(self, idx: int):
        data = self.read_data(self.data[idx][0])
        label = self.read_data(self.data[idx][1])

        if self.rho_type == "chgcar":
            data = data.pgrids["total"].grid_data / np.prod(data.grid_shape)
            label = label.pgrids["total"].grid_data / np.prod(label.grid_shape)
        else:
            data = data.pgrids["total"].grid_data
            label = label.pgrids["total"].grid_data

        data = torch.tensor(data, dtype=dtype_map[self.data_precision]).unsqueeze(0)
        label = torch.tensor(label, dtype=dtype_map[self.data_precision]).unsqueeze(0)

        # Extract patch before augmentation (for memory efficiency)
        data, label = self.extract_patch(data, label)

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
            cden = ChargeDensity.from_file(data_path)
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

    patch_size = getattr(cfg, "patch_size", None)

    train_data = RhoData(
        train_set,
        cfg.data_precision,
        cfg.rho_type,
        cfg.data_augmentation,
        cfg.random_state,
        patch_size=patch_size,  # Patches for training
    )

    test_data = RhoData(
        test_set,
        cfg.data_precision,
        cfg.rho_type,
        data_augmentation=False,
        random_state=cfg.random_state,
        patch_size=None,  # Full volume for validation
    )

    return train_data, test_data
