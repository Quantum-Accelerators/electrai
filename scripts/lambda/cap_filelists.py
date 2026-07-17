#!/usr/bin/env python3
"""Generate a grid-size-capped filelist + remapped split for a functional dir.

Why: the W64 fp16 full-MP run crash-loops because the largest structures
(charge-density grids up to ~540^3 vs a ~109^3 median) make a single DDP step
exceed the 30-min NCCL watchdog timeout (and approach the 80 GB OOM ceiling).
Capping the grid size drops that tail (~1.5% of structures at ~180^3).

Reads  <root>/mp_filelist.txt and <root>/split.json. split.json stores
POSITIONAL indices into the filelist, so filtering the filelist requires
remapping the split indices — done here.

Writes <root>/mp_filelist_<suffix>.txt and <root>/split_<suffix>.json.

Usage:
  python cap_filelists.py <functional_root> [--cap-voxels N] [--suffix capped] [--workers 32]
"""

from __future__ import annotations

import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import zarr


def _voxels(args):
    root, i, idx = args
    d = Path(root) / "data"
    zp, dp = d / f"{idx}.zarr.zip", d / f"{idx}.zarr"
    try:
        if zp.exists():
            s = zarr.storage.ZipStore(str(zp), mode="r")
            try:
                shp = zarr.open_group(s, mode="r")["charge_density_total"].shape
            finally:
                s.close()
        elif dp.exists():
            shp = zarr.open_group(str(dp), mode="r")["charge_density_total"].shape
        else:
            return (i, None)
        nx, ny, nz = (int(x) for x in shp)
        return (i, nx * ny * nz)
    except Exception:
        return (i, None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--cap-voxels", type=int, default=5_832_000)  # ~180^3
    ap.add_argument("--suffix", default="capped")
    ap.add_argument("--workers", type=int, default=32)
    a = ap.parse_args()

    root = Path(a.root)
    ids = [
        line.strip()
        for line in (root / "mp_filelist.txt").read_text().splitlines()
        if line.strip()
    ]
    n = len(ids)
    with Pool(a.workers) as p:
        sizes = dict(
            p.map(
                _voxels,
                [(str(root), i, idx) for i, idx in enumerate(ids)],
                chunksize=64,
            )
        )

    keep = [
        i for i in range(n) if sizes.get(i) is not None and sizes[i] <= a.cap_voxels
    ]
    missing = sum(1 for i in range(n) if sizes.get(i) is None)
    remap = {old: new for new, old in enumerate(keep)}

    (root / f"mp_filelist_{a.suffix}.txt").write_text(
        "\n".join(ids[i] for i in keep) + "\n"
    )
    split = json.loads((root / "split.json").read_text())
    new_split = {k: [remap[o] for o in lst if o in remap] for k, lst in split.items()}
    (root / f"split_{a.suffix}.json").write_text(json.dumps(new_split))

    print(  # noqa: T201
        f"{root.name}: total={n} keep={len(keep)} drop={n - len(keep)} "
        f"({100 * (n - len(keep)) / n:.2f}%) missing={missing} "
        f"cap={a.cap_voxels} (~{a.cap_voxels ** (1 / 3):.0f}^3) "
        f"splits={ {k: len(v) for k, v in new_split.items()} }"
    )


if __name__ == "__main__":
    main()
