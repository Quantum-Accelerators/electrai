from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import torch
from lightning.pytorch import LightningDataModule
from src.electrai.dataloader import utils
from src.electrai.dataloader.collate import collate_fn
from src.electrai.dataloader.split import split_data
from torch.utils.data import DataLoader, Dataset

if TYPE_CHECKING:
    import os

dtype_map = {"f32": torch.float32, "f16": torch.float16, "bf16": torch.bfloat16}


class RhoRead(LightningDataModule):
    def __init__(
        self,
        root: str | bytes | os.PathLike,
        precision: str,
        batch_size: int = 2,
        train_workers: int = 8,
        val_workers: int = 2,
        pin_memory: bool = False,
        val_frac: float = 0.005,
        drop_last: bool = False,
        split_file: str | bytes | os.PathLike | None = None,
        augmentation: bool = False,
        random_seed: int = 42,
        **kwargs,  # noqa: ARG002
    ):
        super().__init__()
        self.save_hyperparameters()
        self.root = root
        self.batch_size = batch_size
        self.train_workers = train_workers
        self.val_workers = val_workers
        self.pin_memory = pin_memory
        self.val_frac = val_frac
        self.drop_last = drop_last
        self.split_file = split_file
        self.precision = precision
        self.augmentation = augmentation
        self.random_seed = random_seed

    def setup(self, stage=None):
        dataset = RhoData(
            self.root, precision=self.precision, augmentation=self.augmentation
        )
        self.subsets = split_data(
            dataset,
            val_frac=self.val_frac,
            split_file=self.split_file,
            random_seed=self.random_seed,
        )
        if stage == "fit":
            self.train_set = self.subsets["train"]
            self.val_set = self.subsets["validation"]
        elif stage == "test":
            self.test_set = self.subsets["test"]

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            self.batch_size,
            num_workers=self.train_workers,
            shuffle=True,
            collate_fn=collate_fn,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_set,
            self.batch_size,
            num_workers=self.val_workers,
            shuffle=False,
            collate_fn=collate_fn,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_set,
            batch_size=1,
            num_workers=self.val_workers,
            collate_fn=collate_fn,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
        )

    def on_exception(self, _exception: BaseException) -> None:
        return


class RhoData(Dataset):
    def __init__(self, datapath: str, precision: str, augmentation: bool, **kwargs):
        super().__init__(**kwargs)
        self.aug = augmentation
        self.precision = precision
        if isinstance(datapath, str) and Path(datapath).is_file():
            with Path(datapath).open() as f:
                lines = f.readlines()
            member_list = [line.replace("\n", "") for line in lines]
        else:
            raise ValueError("No filename found.")

        self.category = Path(datapath).name.split("_")[0]  # example: mp_filelist.txt
        self.root = Path(datapath).parent
        self.member_list = member_list

    def __len__(self):
        return len(self.member_list)

    def __getitem__(self, index):
        index = self.member_list[index]
        data, label = utils.load_numpy_rho(
            root=self.root,
            category=self.category,
            index=index,
            precision=self.precision,
            augmentation=self.aug,
        )
        data = data.unsqueeze(0)
        label = label.unsqueeze(0)
        return {"data": data, "label": label, "index": index}
