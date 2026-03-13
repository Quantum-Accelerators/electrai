from __future__ import annotations

from torch.utils.data import default_collate


def collate_fn(batch):
    try:
        return default_collate(batch)
    except RuntimeError:
        x, y, index, dataset_id = zip(*batch, strict=True)
        return list(x), list(y), list(index), list(dataset_id)
