"""
Filter out all volumetric files missing task data.
"""

from __future__ import annotations

import os
from shutil import move

from tqdm import tqdm

task_docs = [f.split(".")[0] for f in os.listdir("task_docs")]
for folder in ("elfcars", "chgcars"):
    os.makedirs(f"trash/missing_task_data/{folder}", exist_ok=True)
    task_ids = [f.split(".")[0] for f in os.listdir(folder)]
    files_to_remove = list(set(task_ids) - set(task_docs))
    for file_to_remove in tqdm(files_to_remove):
        move(
            f"{folder}/{file_to_remove}.json.gz",
            f"trash/missing_task_data/{folder}/{file_to_remove}.json.gz",
        )
