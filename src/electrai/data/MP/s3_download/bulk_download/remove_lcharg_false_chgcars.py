"""
Filter out all CHGCARs used as inputs and not written out as outputs
"""

from __future__ import annotations

import os
from shutil import move

from monty.serialization import loadfn
from tqdm import tqdm

task_ids = [f.split(".")[0] for f in os.listdir("chgcars")]

files_to_remove = []
for task_id in tqdm(task_ids):
    if not os.path.exists(f"task_docs/{task_id}.json.gz"):
        continue
    task_doc = loadfn(f"task_docs/{task_id}.json.gz")
    if task_doc["input"]["incar"].get("LCHARG", False) is False:
        files_to_remove.append(task_id)

os.makedirs("trash/lcharg_false/chgcars", exist_ok=True)
for file_to_remove in files_to_remove:
    move(
        f"chgcars/{file_to_remove}.json.gz",
        f"trash/lcharg_false/chgcars/{file_to_remove}.json.gz",
    )
