"""Dry run for the CoreWeave W96 campaign.

Validates, on one GB200, everything the real run depends on except multi-GPU:
  1. the staged dataset (filelists, symlink layout, zarr reads via RhoRead),
  2. dataloader batch structure and read throughput,
  3. the W96 memory envelope: forward+backward+Adam step at the grid-size cap
     (180^3, the largest structure the capped filelists allow) in bf16,
     with activation checkpointing off.

Run after scripts/coreweave/stage_data.sh:
  uv run --no-sync python scripts/coreweave/dry_run.py \
      --config src/electrai/configs/MP/config_gga_gga+u_w96.yaml
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import yaml
from hydra.utils import instantiate


def describe(obj):
    if isinstance(obj, torch.Tensor):
        return f"Tensor{tuple(obj.shape)} {obj.dtype}"
    if isinstance(obj, dict):
        return {k: describe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [describe(v) for v in obj]
    return repr(obj)[:60]


def check_data(cfg, n_batches):
    dm = instantiate(cfg["data"])
    dm.setup("fit")
    train = dm.train_dataloader()
    val = dm.val_dataloader()
    print(f"train batches: {len(train)}, val batches: {len(val)}")

    start = time.monotonic()
    for i, batch in enumerate(train):
        if i == 0:
            print("batch structure:", describe(batch))
        if i + 1 >= n_batches:
            break
    dt = time.monotonic() - start
    print(f"read {n_batches} train batches in {dt:.1f}s ({dt / n_batches:.2f}s/batch)")


def check_memory(cfg, grids=(128, 180), iters=1):
    model = instantiate(cfg["model"]).cuda()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model params: {n_params / 1e6:.1f}M")
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    # 180^3 = 5,832,000 voxels: the exact cap from cap_filelists.py, i.e. the
    # worst case any capped structure can present. With iters > 1 the first
    # iteration absorbs cuDNN algorithm search (benchmark mode) and the last
    # one is the steady state.
    total = torch.cuda.get_device_properties(0).total_memory / 2**30
    for n in grids:
        shape = (n, n, n)
        torch.cuda.reset_peak_memory_stats()
        x = torch.randn(1, 1, *shape, device="cuda")
        times = []
        for _ in range(iters):
            start = time.monotonic()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(x).float().square().mean()
            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            times.append(time.monotonic() - start)
        peak = torch.cuda.max_memory_allocated() / 2**30
        print(
            f"grid {shape}: fwd+bwd+step {' / '.join(f'{t:.2f}s' for t in times)}, "
            f"peak {peak:.1f} GiB / {total:.1f} GiB",
            flush=True,
        )
        del x, loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--n-batches", type=int, default=10)
    parser.add_argument("--grids", default="128,180", help="comma-separated cube edges")
    parser.add_argument("--iters", type=int, default=1, help="timed steps per grid")
    parser.add_argument("--cudnn-benchmark", action="store_true")
    parser.add_argument(
        "--skip-data", action="store_true", help="model/memory probe only"
    )
    args = parser.parse_args()

    with Path(args.config).open() as f:
        cfg = yaml.safe_load(f)

    print("== gpu ==")
    print(
        torch.cuda.get_device_name(0), "capability", torch.cuda.get_device_capability(0)
    )
    if args.cudnn_benchmark:
        torch.backends.cudnn.benchmark = True
        print("cudnn.benchmark = True")
    if not args.skip_data:
        print("== data ==")
        check_data(cfg, args.n_batches)
    print("== memory envelope ==")
    check_memory(cfg, grids=[int(g) for g in args.grids.split(",")], iters=args.iters)
    print("DRY RUN PASSED")


if __name__ == "__main__":
    main()
