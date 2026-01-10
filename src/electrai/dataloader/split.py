from __future__ import annotations

import json

import numpy as np
from torch.utils.data import Subset


def split_data(dataset, val_frac=0.005, split_file=None):
    # Load or generate splits
    if split_file is not None:
        with open(split_file) as fp:
            splits = json.load(fp)
    else:
        data_size = len(dataset)
        validation_size = int(data_size * val_frac)
        indices = np.random.permutation(data_size)
        splits = {
            "train": indices[validation_size:].tolist(),
            "validation": indices[:validation_size].tolist(),
        }

    # Split the dataset
    datasplits = {}
    for key, indices in splits.items():
        datasplits[key] = Subset(dataset, indices)
    return datasplits
