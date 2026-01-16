from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import torch
from src.electrai.dataloader import utils
from src.electrai.dataloader.collate import collate_fn
from src.electrai.dataloader.split import split_data
from torch.utils.data import DataLoader, Dataset

if TYPE_CHECKING:
    import os

dtype_map = {"f32": torch.float32, "f16": torch.float16, "bf16": torch.bfloat16}


class RhoRead:
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
        elf: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.root = root
        self.batch_size = batch_size
        self.train_workers = train_workers
        self.val_workers = val_workers
        self.pin_memory = pin_memory
        self.val_frac = val_frac
        self.drop_last = drop_last
        self.split_file = split_file

        dataset = RhoData(
            self.root, precision=precision, augmentation=augmentation, elf=elf
        )

        self.subsets = split_data(
            dataset, val_frac=self.val_frac, split_file=self.split_file
        )

    def train_dataloader(self):
        return DataLoader(
            self.subsets["train"],
            self.batch_size,
            num_workers=self.train_workers,
            shuffle=True,
            # sampler=DistributedSampler(self.train_set, drop_last=self.drop_last), do we need this eventhough we use pytorch lightning? important question
            collate_fn=collate_fn,  # originally it was partial(collate_list_of_dicts, pin_memory=self.pin_memory) should we be concerned about partial and pin_memory?
        )

    def val_dataloader(self):
        return DataLoader(
            self.subsets["validation"],
            self.batch_size,
            num_workers=self.val_workers,
            shuffle=False,  # I added this
            # collate_fn=partial(collate_list_of_dicts, pin_memory=self.pin_memory),
            # note: no sampler, so all devices will get full set
        )

    def test_dataloader(self):
        return DataLoader(
            self.subsets["test"],
            batch_size=1,
            num_workers=self.val_workers,
            collate_fn=collate_fn,  # partial(collate_list_of_dicts, pin_memory=self.pin_memory),
            # note: distributed sampler will shuffle and distribute different parts of dataset
            # to different nodes/devices
            # sampler=DistributedEvalSampler(self.test_set),
        )


class RhoData(Dataset):
    def __init__(
        self, datapath: str, precision: str, augmentation: bool, elf: bool, **kwargs
    ):
        super().__init__(**kwargs)
        self.aug = augmentation
        self.precision = precision
        self.elf = elf
        if isinstance(datapath, str) and Path(datapath).is_file():
            with open(datapath) as f:
                lines = f.readlines()
            member_list = [line.replace("\n", "") for line in lines]
        else:
            raise ValueError("No filename found.")

        self.category = Path(datapath).name.split("_")[0]
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
            augmentation=self.aug,
            elf=self.elf,
        )
        data = torch.tensor(data, dtype=dtype_map[self.precision]).unsqueeze(0)
        label = torch.tensor(label, dtype=dtype_map[self.precision]).unsqueeze(0)
        return {"data": data, "label": label, "index": index}
