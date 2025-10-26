from __future__ import annotations

import random
import shutil
from pathlib import Path

from monty.serialization import loadfn

# Source folder
src = Path("/scratch/gpfs/ROSENGROUP/common/mp/chgcars/")

# Destination folder
dest = Path("./")
# dest.mkdir(parents=True, exist_ok=True)

# Load mapping
mapping = loadfn(Path("../map/chgcars_functional_to_task_ids.json.gz"))
# with open(map_dir) as f:
#    mapping = json.load(f)

# Randomly sample 10 task IDs
task_ids_list = mapping["GGA"]
sampled = random.sample(task_ids_list, 10)
print("Sampled task IDs:", sampled)

# Copy files
for sample in sampled:
    src_file = src / f"{sample}.json.gz"
    shutil.copy(src_file, dest)
