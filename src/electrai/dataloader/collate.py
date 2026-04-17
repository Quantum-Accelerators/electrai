from __future__ import annotations

from torch.utils.data import default_collate


def collate_fn(batch):
    # Patchified samples: "data" is a list of patch tensors instead of a single tensor.
    # batch_size must be 1 for this path; return the item dict directly.
    if isinstance(batch[0], dict) and isinstance(batch[0].get("data"), list):
        if len(batch) != 1:
            raise ValueError(
                "Patchified samples (data is a list of patches) require batch_size=1, "
                f"got batch_size={len(batch)}."
            )
        item = dict(batch[0])
        # Wrap scalar index in a list to match the batched format used in test_step
        if isinstance(item.get("index"), str):
            item["index"] = [item["index"]]
        return item

    try:
        return default_collate(batch)
    except RuntimeError:
        # Fallback for variable-shape tensors with batch_size > 1
        data = [item["data"] for item in batch]
        labels = [item["label"] for item in batch]
        indices = [item["index"] for item in batch]
        return {"data": data, "label": labels, "index": indices}
