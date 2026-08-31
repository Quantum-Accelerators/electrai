#!/usr/bin/env python3
"""Subsample the train indices of a split JSON for cheap sweep trials.

Reads a split file as consumed by electrai.dataloader.split.split_data
({"train": [indices...], "validation": [...], ...}), draws a deterministic
random fraction of the *train* indices, and writes a new split file with
every other key (validation, test) copied through untouched — so val_loss
from sweep trials stays directly comparable to full-dataset runs.

Run once per functional against a local copy of split_capped.json, e.g.:

    rclone copy cw:mp/chg_datasets/functionals/gga/split_capped.json /tmp/gga/
    uv run python scripts/coreweave/make_sweep_split.py \
        /tmp/gga/split_capped.json data/MP/sweep_splits/gga_split_sweep12k.json \
        --frac 0.1112

Using the same --frac for gga and gga+u keeps the 76/24 functional mix of
the full capped train set.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("infile", type=Path, help="source split JSON")
    parser.add_argument("outfile", type=Path, help="destination split JSON")
    parser.add_argument(
        "--frac",
        type=float,
        required=True,
        help="fraction of train indices to keep (0, 1]",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not 0 < args.frac <= 1:
        parser.error(f"--frac must be in (0, 1], got {args.frac}")

    with args.infile.open() as fp:
        splits = json.load(fp)

    train = splits["train"]
    n_keep = round(len(train) * args.frac)
    rng = random.Random(args.seed)
    # Sorted so epoch iteration order is index order, matching full-set splits
    kept = sorted(rng.sample(train, n_keep))

    out = {**splits, "train": kept}
    args.outfile.parent.mkdir(parents=True, exist_ok=True)
    with args.outfile.open("w") as fp:
        json.dump(out, fp)

    other = {k: len(v) for k, v in splits.items() if k != "train"}
    print(f"{args.infile} -> {args.outfile}")
    print(f"  train: {len(train)} -> {n_keep} (frac={args.frac}, seed={args.seed})")
    print(f"  passed through unchanged: {other}")


if __name__ == "__main__":
    main()
