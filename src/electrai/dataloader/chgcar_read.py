from __future__ import annotations

from pathlib import Path

import numpy as np
from monty.serialization import loadfn
from sklearn.model_selection import train_test_split

from .registry import register_dataset


class RhoRead:
    def __init__(
        self,
        data_dir: str,
        label_dir: str,
        map_dir: str,
        rho_type: str,
        functional: str,
        normalize: bool,
        train_fraction: float,
        random_state: int = 42,
    ):
        """
        data_dir: directory of input chgcar or elfcar files.
        label_dir: directory of label chgcar or elfcar files.
        map_dir: directory of json file mapping functional to list of task_ids.
        rho_type: chgcar or elfcar.
        functional: 'GGA', 'GG+U', 'PBEsol', 'SCAN', 'r2SCAN'.
        normalize: whether to normalize the CHGCAR data so that its integral over the grid equals NELECT.
        train_fraction: fraction of the data used for training (0 to 1).
        """
        self.data_dir = Path(data_dir)
        self.label_dir = Path(label_dir)
        self.map_dir = Path(map_dir)
        self.rho_type = rho_type
        self.functional = functional
        self.normalize = normalize
        self.tf = train_fraction
        self.rs = random_state

    def read_label(self, data_dir):
        """
        data_dir: directory of json-formatted chg or elfcar files.
        """
        data = loadfn(data_dir)
        charge = data["data"].data["total"]
        gridsize = charge.shape
        if self.rho_type == "chgcar" and self.normalize:
            charge /= np.prod(gridsize)
        return charge.flatten(), gridsize

    def read_data(self, data_dir):  # will later change it to input directory
        """
        data_dir: directory of json-formatted chg or elfcar files.
        """
        data = loadfn(data_dir)
        charge = data["data"].data["total"]
        gridsize = charge.shape
        if self.rho_type == "chgcar" and self.normalize:
            charge /= np.prod(gridsize)
        return charge.flatten(), gridsize

    def data_split(self):
        mapping = loadfn(self.map_dir)
        data_list, label_list = [], []
        gs_data_list, gs_label_list = [], []

        for task_id in mapping[self.functional]:
            data_dir = self.label_dir / f"{task_id}.json.gz"
            label_dir = self.data_dir / f"{task_id}.json.gz"
            data, gs_data = self.read_data(data_dir)
            label, gs_label = self.read_label(label_dir)

            data_list.append(data)
            label_list.append(label)
            gs_data_list.append(gs_data)
            gs_label_list.append(gs_label)

        splits = train_test_split(
            data_list,
            label_list,
            gs_data_list,
            gs_label_list,
            train_size=self.tf,
            random_state=self.rs,
        )

        train_sets = splits[::2]
        test_sets = splits[1::2]

        return train_sets, test_sets


@register_dataset("chgcar_data")
def load_data(cfg):
    reader = RhoRead(
        data_dir=cfg.data_dir,
        label_dir=cfg.label_dir,
        map_dir=cfg.map_dir,
        rho_type=cfg.rho_type,
        functional=cfg.functional,
        normalize=cfg.normalize_data,
        train_fraction=cfg.train_fraction,
        random_state=cfg.random_state,
    )

    return reader.data_split()
