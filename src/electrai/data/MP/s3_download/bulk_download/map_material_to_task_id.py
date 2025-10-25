"""
Maps material_id to task_id and vice versa
"""

from __future__ import annotations

import os

from monty.serialization import dumpfn
from mp_api.client import MPRester
from tqdm import tqdm

task_to_mat_id_map = {}
mat_to_task_id_map = {}


with MPRester() as mpr:
    docs = mpr.materials.summary.search(
        deprecated=False, fields=["material_id", "task_ids"]
    )

for doc in tqdm(docs):
    mp_id = str(doc.material_id)
    mat_to_task_id_map[mp_id] = []
    for task_id in doc.task_ids:
        task_id = str(task_id)
        task_to_mat_id_map[task_id] = mp_id
        mat_to_task_id_map[mp_id].append(task_id)

os.makedirs("metadata", exist_ok=True)
dumpfn(task_to_mat_id_map, "metadata/task_id_to_material_id.json.gz")
dumpfn(mat_to_task_id_map, "metadata/material_id_to_task_ids.json.gz")
