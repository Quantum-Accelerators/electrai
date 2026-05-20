from __future__ import annotations

from torch.utils.data import default_collate


def collate_fn(batch):
    try:
        return default_collate(batch)
    except RuntimeError:
        result = {}
        for k in batch[0]:
            try:
                result[k] = default_collate([d[k] for d in batch])
            except RuntimeError:
                result[k] = [d[k] for d in batch]
        return result
