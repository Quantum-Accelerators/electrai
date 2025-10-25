"""
Filter out all CHGCARs written out during an NSCF run.
"""

from __future__ import annotations

import os
from shutil import move

from monty.serialization import loadfn
from tqdm import tqdm

folders = ("elfcars", "chgcars", "aeccar0s", "aeccar2s")
task_ids = []
for folder in folders:
    task_ids.extend([f.split(".")[0] for f in os.listdir(folder)])
task_ids = list(set(task_ids))

files_to_remove = []
for task_id in tqdm(task_ids):
    if not os.path.exists(f"task_docs/{task_id}.json.gz"):
        continue
    task_doc = loadfn(f"task_docs/{task_id}.json.gz")
    if task_doc["input"]["incar"].get("ICHARG", 0) >= 10:
        files_to_remove.append(task_id)

for file_type in ("elfcars", "chgcars", "aeccar0s", "aeccar2s"):
    os.makedirs(f"trash/nscf/{file_type}", exist_ok=True)
    for file_to_remove in files_to_remove:
        if os.path.exists(f"{file_type}/{file_to_remove}.json.gz"):
            move(
                f"{file_type}/{file_to_remove}.json.gz",
                f"trash/nscf/{file_type}/{file_to_remove}.json.gz",
            )
