from __future__ import annotations

import importlib

from .registry import DATASET_REGISTRY, get_data, register_data

for module in ["mp", "zarr_s3_read"]:
    importlib.import_module(f"{__name__}.{module}")
