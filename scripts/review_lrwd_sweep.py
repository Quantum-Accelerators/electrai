#!/usr/bin/env python3
"""Aggregate the LR/WD width-sweep W&B runs into per-(width, lr, wd) results.

Preemption restarts create a fresh W&B run per segment (no explicit id=), so
runs are grouped by (n_channels, lr, weight_decay) from run config and merged
per epoch before picking best/final val_loss. val_loss is NormMAE; x100 = %.

    uv run python scripts/review_lrwd_sweep.py [project]
"""
# ruff: noqa: T201

from __future__ import annotations

import sys

import wandb
import wandb.sdk.lib.server as _wandb_server

# Server returns "flags": null, crashing the viewer query inside Api() login
# (same wandb bug that forces offline mode + sidecar sync on the cluster).
_orig_query = _wandb_server.Server.query_with_timeout


def _query_tolerant(self, *args, **kwargs):
    try:
        _orig_query(self, *args, **kwargs)
    except TypeError:
        self._flags = {}


_wandb_server.Server.query_with_timeout = _query_tolerant

ENTITY = "PrinceOA"
PROJECT = sys.argv[1] if len(sys.argv) > 1 else "mp-gga-ggau-lrwd"


def main() -> None:
    api = wandb.Api()
    runs = api.runs(f"{ENTITY}/{PROJECT}")

    trials: dict[tuple[int, float, float, str], dict[int, float]] = {}
    for run in runs:
        cfg = run.config
        model = cfg.get("model") or {}
        width = model.get("n_channels")
        lr = cfg.get("lr")
        wd = cfg.get("weight_decay", 0.0)
        if width is None or lr is None:
            print(f"  (skipping run {run.name}: no width/lr in config)")
            continue
        # The _rep2 noise-bar reruns share (width, lr, wd) with their stage-A
        # twins; the config run_name (the config stem) keeps them apart while
        # still merging preemption-restart segments, which share it.
        rep = "rep2" if str(cfg.get("run_name", "")).endswith("_rep2") else ""
        per_epoch = trials.setdefault((width, float(lr), float(wd), rep), {})
        for h in run.scan_history(keys=["epoch", "val_loss_epoch"]):
            ep, val = h.get("epoch"), h.get("val_loss_epoch")
            if ep is None or val is None:
                continue
            # Restart segments can re-log an epoch; keep the better value
            per_epoch[int(ep)] = min(val, per_epoch.get(int(ep), float("inf")))

    print(
        f"{'width':>5} {'lr':>8} {'wd':>7} {'rep':>4} {'epochs':>6} {'best val':>9} "
        f"{'best%':>6} {'final val':>9} {'@ep':>3}"
    )
    best_by_width: dict[int, tuple[float, float]] = {}
    for (width, lr, wd, rep), per_epoch in sorted(trials.items()):
        if not per_epoch:
            continue
        best_ep = min(per_epoch, key=per_epoch.get)
        last_ep = max(per_epoch)
        best = per_epoch[best_ep]
        print(
            f"{width:>5} {lr:>8g} {wd:>7g} {rep:>4} {len(per_epoch):>6} {best:>9.6f} "
            f"{best * 100:>6.3f} {per_epoch[last_ep]:>9.6f} {best_ep:>3}"
        )
        if (
            wd == 0.0
            and not rep
            and best < best_by_width.get(width, (float("inf"),))[0]
        ):
            best_by_width[width] = (best, lr)

    if best_by_width:
        args = " ".join(f"{w}={lr:g}" for w, (_, lr) in sorted(best_by_width.items()))
        print(f"\nBest LR per width (wd=0 trials): {args}")
        print(
            f"Stage B: uv run python scripts/coreweave/gen_lrwd_sweep.py "
            f"--stage b --best-lr {args}"
        )


if __name__ == "__main__":
    main()
