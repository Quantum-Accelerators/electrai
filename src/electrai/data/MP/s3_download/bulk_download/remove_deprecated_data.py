"""
Filter out all CHGCARs associated with a deprecated ID
"""

from __future__ import annotations

import csv
import os
from shutil import move

from tqdm import tqdm

task_ids = []
with open("metadata/deprecated_task_ids.csv") as f:
    reader = csv.reader(f)
    for row in reader:
        task_ids.extend(row)

for task_id in tqdm(task_ids):
    for folder in ("elfcars", "chgcars"):
        os.makedirs(f"trash/deprecated/{folder}", exist_ok=True)
        if os.path.exists(f"{folder}/{task_id}.json.gz"):
            move(
                f"{folder}/{task_id}.json.gz",
                f"trash/deprecated/{folder}/{task_id}.json.gz",
            )
