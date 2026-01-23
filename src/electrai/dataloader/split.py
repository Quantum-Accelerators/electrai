from __future__ import annotations

import json

import torch
from torch.utils.data import Subset


def split_data(dataset, val_frac=0.005, split_file=None, random_seed=42):
    # Load or generate splits
    if split_file is not None:
        with open(split_file) as fp:  # noqa: PTH123
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
