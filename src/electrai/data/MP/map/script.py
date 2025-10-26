from __future__ import annotations

import gzip
import json

sample_map = {
    "GGA": [
        "mp-2355719",
        "mp-1933176",
        "mp-2507978",
        "mp-2255579",
        "mp-1800415",
        "mp-1923722",
        "mp-2452291",
        "mp-1790998",
        "mp-2632472",
        "mp-1802556",
    ]
}

with gzip.open("map_sample.json.gz", "wt") as f:
    json.dump(sample_map, f)
