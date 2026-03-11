from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from lightning.pytorch import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from electrai.dataloader import utils
from electrai.dataloader.collate import collate_fn
from electrai.dataloader.split import split_data

if TYPE_CHECKING:
    import os


dtype_map = {"f32": torch.float32, "f16": torch.float16, "bf16": torch.bfloat16}


@dataclass(frozen=True)
class DatasetSpec:
    root: str
    split_file: str | None = None
    val_frac: float | None = None
    functional_id: int | None = None


class WithFunctionalID(Dataset):
    """Wrap any dataset so every sample carries a constant functional_id."""

    def __init__(self, base: Dataset, functional_id: int, key: str = "functional_id"):
        self.base = base
        self.functional_id = int(functional_id)
        self.key = key

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        out = self.base[idx]
        out[self.key] = self.functional_id
        return out


class RhoRead(LightningDataModule):
    """
    One datamodule that works for:
      1) Single dataset: pass `root=...` (old behavior)
      2) Multiple datasets: pass `datasets=[...]` and it concatenates splits.
    """

    def __init__(
        self,
        # single-dataset style
        root: str | bytes | os.PathLike | None = None,
        split_file: str | bytes | os.PathLike | None = None,
        val_frac: float = 0.005,
        # multi-dataset style
        datasets: list[dict[str, Any]] | list[DatasetSpec] | None = None,
        # common
        precision: str = "f32",
        batch_size: int = 2,
        train_workers: int = 8,
        val_workers: int = 2,
        pin_memory: bool = False,
        drop_last: bool = False,
        augmentation: bool = False,
        random_seed: int = 42,
        # what key name to attach the id under
        functional_id_key: str = "functional_id",
    ):
        super().__init__()
        self.save_hyperparameters()

        self.precision = precision
        self.batch_size = batch_size
        self.train_workers = train_workers
        self.val_workers = val_workers
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.augmentation = augmentation
        self.random_seed = random_seed
        self.functional_id_key = functional_id_key

        # ---- Normalize config into a list[DatasetSpec] ----
        if datasets is not None and root is not None:
            raise ValueError("Provide either `root` OR `datasets`, not both.")

        if datasets is None:
            if root is None:
                raise ValueError("You must provide either `root` or `datasets`.")
            # single dataset -> one spec
            self.specs = [
                DatasetSpec(
                    root=str(root),
                    split_file=str(split_file) if split_file is not None else None,
                    val_frac=val_frac,
                    functional_id=0,
                )
            ]
        else:
            specs: list[DatasetSpec] = [
                d if isinstance(d, DatasetSpec) else DatasetSpec(**d) for d in datasets
            ]
            if not specs:
                raise ValueError("`datasets` must contain at least one dataset spec.")

            # Fill defaults: if per-spec val_frac/split_file missing, use top-level ones
            filled: list[DatasetSpec] = []
            for i, s in enumerate(specs):
                fid = s.functional_id if s.functional_id is not None else i
                filled.append(
                    DatasetSpec(
                        root=s.root,
                        split_file=s.split_file
                        if s.split_file is not None
                        else (str(split_file) if split_file is not None else None),
                        val_frac=s.val_frac if s.val_frac is not None else val_frac,
                        functional_id=fid,
                    )
                )

            # Ensure unique fids (contiguity not required)
            fids = [s.functional_id for s in filled]  # type: ignore[list-item]
            if len(set(fids)) != len(fids):
                raise ValueError(f"Duplicate functional_id values found: {fids}")

            self.specs = filled

        self.train_set: Dataset | None = None
        self.val_set: Dataset | None = None
        self.test_set: Dataset | None = None

    def setup(self, stage: str | None = None):
        train_parts: list[Dataset] = []
        val_parts: list[Dataset] = []
        test_parts: list[Dataset] = []

        # We build splits per dataset, then concatenate corresponding splits.
        for spec in self.specs:
            ds = RhoData(
                spec.root, precision=self.precision, augmentation=self.augmentation
            )
            splits = split_data(
                ds,
                val_frac=float(spec.val_frac),  # guaranteed filled above
                split_file=spec.split_file,
                random_seed=self.random_seed,
            )

            fid = int(spec.functional_id)  # guaranteed filled above

            train_parts.append(
                WithFunctionalID(splits["train"], fid, key=self.functional_id_key)
            )
            val_parts.append(
                WithFunctionalID(splits["validation"], fid, key=self.functional_id_key)
            )
            if "test" in splits and splits["test"] is not None:
                test_parts.append(
                    WithFunctionalID(splits["test"], fid, key=self.functional_id_key)
                )

        # Fit
        if stage is None or stage == "fit":
            self.train_set = (
                train_parts[0] if len(train_parts) == 1 else ConcatDataset(train_parts)
            )
            self.val_set = (
                val_parts[0] if len(val_parts) == 1 else ConcatDataset(val_parts)
            )

        # Test
        if stage is None or stage == "test":
            if test_parts:
                self.test_set = (
                    test_parts[0] if len(test_parts) == 1 else ConcatDataset(test_parts)
                )
            else:
                # If split_data didn't create a test set, keep it None (Lightning won't call test_dataloader unless you run test)
                self.test_set = None

    def train_dataloader(self):
        assert self.train_set is not None
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            num_workers=self.train_workers,
            shuffle=True,
            collate_fn=collate_fn,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
        )

    def val_dataloader(self):
        assert self.val_set is not None
        return DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            num_workers=self.val_workers,
            shuffle=False,
            collate_fn=collate_fn,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
        )

    def test_dataloader(self):
        if self.test_set is None:
            raise ValueError("No test set available. (split_data did not produce one.)")
        return DataLoader(
            self.test_set,
            batch_size=1,
            num_workers=self.val_workers,
            shuffle=False,
            collate_fn=collate_fn,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
        )


class RhoData(Dataset):
    def __init__(self, datapath: str, precision: str, augmentation: bool, **kwargs):
        super().__init__(**kwargs)
        self.aug = augmentation
        self.precision = precision

        p = Path(datapath)
        if p.is_file():
            member_list = [
                line.strip() for line in p.read_text().splitlines() if line.strip()
            ]
        else:
            raise ValueError(f"No filename found at: {datapath}")

        self.category = p.name.split("_")[0]  # e.g. mp_filelist.txt -> "mp"
        self.root = p.parent
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
