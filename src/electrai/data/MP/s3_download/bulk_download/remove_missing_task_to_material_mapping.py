"""
Filter out all task IDs within chgcars with no associated material ID.
"""

from __future__ import annotations

import gzip
import json
import os
from shutil import move

from tqdm import tqdm

for folder in ("elfcars", "chgcars"):
    task_ids = [f.split(".")[0] for f in os.listdir(folder)]

    map_file = "metadata/task_id_to_material_id.json.gz"
    with gzip.open(map_file, "rt") as file:
        map_task_material = json.load(file)

    files_to_remove = []
    for task_id in tqdm(task_ids):
        if task_id not in map_task_material:
            files_to_remove.append(task_id)

    os.makedirs("trash/missing_task_material_mapping/chgcars/", exist_ok=True)
    for file_to_remove in files_to_remove:
        move(
            f"{folder}/{file_to_remove}.json.gz",
            f"trash/missing_task_material_mapping/{folder}/{file_to_remove}.json.gz",
        )
