"""
Downloads task documents corresponding to a list of task IDs
"""

from __future__ import annotations

import os

from emmet.core.utils import jsanitize
from monty.serialization import dumpfn
from mp_api.client.routes.materials.tasks import TaskRester

os.makedirs("task_docs", exist_ok=True)
task_ids_to_fetch = [
    f.split(".")[0] for f in os.listdir("elfcars") + os.listdir("chgcars")
]
fetched_task_ids = [f.split(".")[0] for f in os.listdir("task_docs")]
task_ids = list(set(task_ids_to_fetch) - set(fetched_task_ids))
batch_size = 10000
with TaskRester() as tpr:
    for i in range(0, len(task_ids), batch_size):
        docs = tpr.search(task_ids[i : i + batch_size])
        for doc in docs:
            dumpfn(jsanitize(doc), f"task_docs/{doc.task_id}.json.gz")
