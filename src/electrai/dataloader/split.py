from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset, Subset


def split_data(
    dataset: Dataset,
    val_frac: float = 0.005,
    split_file: str | None = None,
    random_seed: int = 42,
):
    # Load or generate splits
    if split_file is not None:
        with Path(split_file).open() as fp:
            splits = json.load(fp)
    else:
        data_size = len(dataset)
        validation_size = int(data_size * val_frac)
        g = torch.Generator()
        g.manual_seed(random_seed)

        indices = torch.randperm(data_size, generator=g)

        splits = {
            "train": indices[validation_size:].tolist(),
            "validation": indices[:validation_size].tolist(),
        }

    # Split the dataset
    datasplits = {}
    for key, indices in splits.items():
        datasplits[key] = Subset(dataset, indices)
    return datasplits
