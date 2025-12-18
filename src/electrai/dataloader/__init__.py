from __future__ import annotations

import importlib

from .registry import DATASET_REGISTRY, get_data, register_data

for module in ["mp", "mp_zarr_s3_data", "qm9"]:
    importlib.import_module(f"{__name__}.{module}")
