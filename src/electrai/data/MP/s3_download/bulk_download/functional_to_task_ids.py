from __future__ import annotations

import os
import sys
from collections import defaultdict
from multiprocessing import Pool, cpu_count

from monty.serialization import dumpfn, loadfn


def get_run_type(task_id):
    doc = loadfn(f"task_docs/{task_id}.json.gz")
    run_type = doc["calcs_reversed"][0]["run_type"]
    return task_id, run_type


if __name__ == "__main__":
    folder = sys.argv[1]
    task_ids = [f.split(".")[0] for f in os.listdir(folder)]

    nproc = cpu_count()  # min(8, cpu_count())
    print(f"Using {nproc} parallel workers...")

    functional_to_task = defaultdict(list)

    with Pool(nproc) as pool:
        for counter, (task_id, run_type) in enumerate(
            pool.imap_unordered(get_run_type, task_ids), 1
        ):
            print(counter)
            functional_to_task[run_type].append(task_id)

    dumpfn(functional_to_task, f"metadata/{folder}_functional_to_task_ids.json.gz")
