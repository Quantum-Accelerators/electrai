from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from lightning.pytorch import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from electrai.dataloader import utils
from electrai.dataloader.collate import collate_fn
from electrai.dataloader.split import split_data

dtype_map = {"f32": torch.float32, "f16": torch.float16, "bf16": torch.bfloat16}


@dataclass(frozen=True)
class DatasetSpec:
    root: str
    split_file: str | None = None
    val_frac: float | None = None
    dataset_id: int | None = None


class AddDatasetID(Dataset):
    """Wrap dataset so every sample carries a constant Dataset_ID"""

    def __init__(self, base: Dataset, functional_id: int, key: str = "Dataset_ID"):
        self.base = base
        self.functional_id = int(functional_id)
        self.key = key

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        out = dict(self.base[idx])
        out[self.key] = self.functional_id
        return out


class RhoRead(LightningDataModule):
    """
    Works based on these keys for one or more datasets:
    `root`, `split_file`, `val_frac` (if `split_file` is null) and dataset_id
    """

    def __init__(
        self,
        datasets: list[dict[str, Any]] | list[DatasetSpec] | None = None,
        default_val_frac: float = 0.005,
        default_split_file: str | None = None,
        precision: str = "f32",
        batch_size: int = 2,
        train_workers: int = 8,
        val_workers: int = 2,
        pin_memory: bool = False,
        drop_last: bool = False,
        augmentation: bool = False,
        random_seed: int = 42,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.batch_size = batch_size
        self.train_workers = train_workers
        self.val_workers = val_workers
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.precision = precision
        self.augmentation = augmentation
        self.random_seed = random_seed

        if datasets is None or len(datasets) == 0:
            raise ValueError("`datasets` must contain at least one dataset spec.")

        specs: list[DatasetSpec] = [
            d if isinstance(d, DatasetSpec) else DatasetSpec(**d) for d in datasets
        ]
        filled: list[DatasetSpec] = []
        for i, s in enumerate(specs):
            dataset_id = s.dataset_id if s.dataset_id is not None else i
            filled.append(
                DatasetSpec(
                    root=s.root,
                    split_file=(
                        s.split_file if s.split_file is not None else default_split_file
                    ),
                    val_frac=(
                        s.val_frac if s.val_frac is not None else default_val_frac
                    ),
                    dataset_id=dataset_id,
                )
            )
        self.specs = filled

        self.train_set: Dataset | None = None
        self.val_set: Dataset | None = None
        self.test_set: Dataset | None = None

    def setup(self, stage: str):
        train_parts: list[Dataset] = []
        val_parts: list[Dataset] = []
        test_parts: list[Dataset] = []

        for spec in self.specs:
            if stage == "test" and spec.split_file is None:
                warnings.warn(
                    f"Dataset with root '{spec.root}' has no split_file and will be "
                    "skipped for the test stage (test indices require a split_file).",
                    UserWarning,
                    stacklevel=2,
                )
                continue

            ds = RhoData(
                spec.root, precision=self.precision, augmentation=self.augmentation
            )
            dataset_id = int(spec.dataset_id)

            if stage == "fit":
                splits = split_data(
                    ds,
                    val_frac=float(spec.val_frac),
                    split_file=spec.split_file,
                    random_seed=self.random_seed,
                )
                train_parts.append(AddDatasetID(splits["train"], dataset_id))
                val_parts.append(AddDatasetID(splits["validation"], dataset_id))
            elif stage == "test":
                splits = split_data(
                    ds,
                    val_frac=float(spec.val_frac),
                    split_file=spec.split_file,
                    random_seed=self.random_seed,
                )
                if "test" in splits and splits["test"] is not None:
                    test_parts.append(AddDatasetID(splits["test"], dataset_id))

        if stage == "fit":
            self.train_set = (
                train_parts[0] if len(train_parts) == 1 else ConcatDataset(train_parts)
            )
            self.val_set = (
                val_parts[0] if len(val_parts) == 1 else ConcatDataset(val_parts)
            )
        elif stage == "test":
            self.test_set = (
                test_parts[0] if len(test_parts) == 1 else ConcatDataset(test_parts)
            )

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


class RhoData(Dataset):
    def __init__(self, datapath: str, precision: str, augmentation: bool, **kwargs):
        super().__init__(**kwargs)
        self.aug = augmentation
        self.precision = precision
        if isinstance(datapath, str) and Path(datapath).is_file():
            with Path(datapath).open() as f:
                lines = f.readlines()
            member_list = [line.strip() for line in lines if line.strip()]
        else:
            raise ValueError("No filename found.")

        self.category = Path(datapath).name.split("_")[0]  # example: mp_filelist.txt
        self.root = Path(datapath).parent
        if not member_list:
            raise ValueError(f"Filelist at {datapath} is empty.")
        self.member_list = member_list
        # Detect zarr vs CHGCAR by checking which extension the first entry has
        first = member_list[0]
        if (self.root / "data" / f"{first}.zarr").exists():
            self.fmt = "zarr"
        elif (self.root / "data" / f"{first}.CHGCAR").exists():
            self.fmt = "chgcar"
        else:
            raise ValueError(
                f"No .zarr or .CHGCAR file found for '{first}' in {self.root / 'data'}"
            )

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
            fmt=self.fmt,
        )
        data = data.unsqueeze(0)
        label = label.unsqueeze(0)
        return {"data": data, "label": label, "index": index}
