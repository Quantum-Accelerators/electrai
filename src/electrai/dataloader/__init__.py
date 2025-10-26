from __future__ import annotations

import importlib

from .registry import DATASET_REGISTRY, get_dataset, register_dataset

for module in ["chgcar_read"]:
    importlib.import_module(f"{__name__}.{module}")
