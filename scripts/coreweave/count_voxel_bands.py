"""Count capped-dataset structures above the per-width cuDNN kernel cliffs.

cuDNN's fast conv engines fall back to reference kernels (~400x slower) when
the U-Net decoder-concat tensor exceeds 2^31 elements. For concat channels
2C, the per-structure voxel ceiling is 2^31 / (2 * n_channels). This script
reads zarr shape metadata for every id in the capped filelists (data must be
staged locally; run on a warm node) and reports how many structures exceed
each width's ceiling — i.e. the data cost of a width-specific tighter cap.
"""

from __future__ import annotations

import statistics
from multiprocessing import Pool
from pathlib import Path

import zarr

STAGE_ROOT = Path("/uv/cache/electrai/mp/chg_datasets/functionals")
CEILINGS = {
    "W192 (2*192 ch concat)": 2**31 // 384,  # 5,592,405 voxels
    "W256 (2*256 ch concat)": 2**31 // 512,  # 4,194,304 voxels
}
CURRENT_CAP = 5_832_000


def voxels(args):
    root, mpid = args
    try:
        g = zarr.open_group(str(root / "data" / f"{mpid}.zarr"), mode="r")
        s = g["charge_density_total"].shape
        return s[0] * s[1] * s[2]
    except Exception:
        return -1


def main():
    tasks = []
    for func in ("gga", "gga+u"):
        root = STAGE_ROOT / func
        ids = (root / "mp_filelist_capped.txt").read_text().split()
        tasks += [(root, i) for i in ids]

    with Pool(32) as p:
        sizes = p.map(voxels, tasks, chunksize=256)

    ok = [s for s in sizes if s > 0]
    print(f"total {len(sizes)}, unreadable {len(sizes) - len(ok)}")
    print(f"max voxels: {max(ok)} (current cap {CURRENT_CAP})")
    print(f"mean voxels: {sum(ok) / len(ok):.0f}, median: {statistics.median(ok):.0f}")
    for label, ceiling in CEILINGS.items():
        over = sum(1 for s in ok if s > ceiling)
        print(f"over {label} ceiling {ceiling}: {over} ({100 * over / len(ok):.3f}%)")
    print("COUNT DONE")


if __name__ == "__main__":
    main()
