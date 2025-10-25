"""
Writes out a CSV file of all deprecated material IDs and task IDs
"""

from __future__ import annotations

import csv
import os

from mp_api.client import MPRester

with MPRester() as mpr:
    docs = mpr.materials.summary.search(
        deprecated=True, fields=["material_id", "task_ids"]
    )

deprecated_task_ids = []
deprecated_material_ids = []
for doc in docs:
    deprecated_task_ids.extend([str(task_id) for task_id in doc.task_ids])
    deprecated_material_ids.append(doc.material_id)
deprecated_task_ids.sort()
deprecated_material_ids.sort()

os.makedirs("metadata", exist_ok=True)
with open("metadata/deprecated_task_ids.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerow(deprecated_task_ids)

with open("metadata/deprecated_material_ids.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerow(deprecated_material_ids)
