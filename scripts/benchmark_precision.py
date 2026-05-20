"""
Benchmark: f32 vs bf16-mixed training, one task_id at a time.

For each (task_id, precision) pair:
  - Build model and load the single zarr sample
  - Run 3 epochs of forward+backward (full training step)
  - Record peak GPU memory and per-epoch wall time
  - Catch OOM and record it

Usage:
    uv run python scripts/benchmark_precision.py \
        --config path/to/config.yaml \
        --zarr_root path/to/zarr_root \
        --results path/to/results.json
"""

from __future__ import annotations

import argparse
import gc
import json
import time
import warnings
from pathlib import Path
from types import SimpleNamespace

import torch
import yaml
from hydra.utils import instantiate
from torch.utils.data import DataLoader

from electrai.dataloader.collate import collate_fn
from electrai.dataloader.large_grid_json_gz import LargeGridZarrDataset
from electrai.model.loss.charge import NormMAE

# ---------------------------------------------------------------------------
# Defaults (override via CLI arguments)
# ---------------------------------------------------------------------------
TASK_IDS = [
    "mp-1862536",
    "mp-1936557",
    "mp-1847208",
    "mp-1850168",
    "mp-1890579",
    "mp-1871122",
    "mp-1889246",
    "mp-1849767",
    "mp-1851604",
    "mp-1887804",
]
PRECISIONS = ["f32", "bf16-mixed"]
EPOCHS = 3
DEVICE = torch.device("cuda:0")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Per-sample f32 vs bf16-mixed benchmark"
    )
    parser.add_argument(
        "--config", type=Path, required=True, help="Path to training config YAML"
    )
    parser.add_argument(
        "--zarr_root", type=str, required=True, help="Path to zarr root directory"
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("benchmark_results/per_sample_precision.json"),
        help="Output JSON path (default: benchmark_results/per_sample_precision.json)",
    )
    parser.add_argument(
        "--task_ids",
        nargs="+",
        default=None,
        help="Override task IDs to benchmark (default: all 10)",
    )
    return parser.parse_args()


def load_cfg(config_path: Path):
    with config_path.open() as f:
        d = yaml.safe_load(f)
    return SimpleNamespace(**d)


def build_model(cfg):
    model = instantiate(cfg.model)
    return model.to(DEVICE)


def get_grid_shape(task_id: str, zarr_root: str) -> tuple[int, ...]:
    """Return the charge density grid shape for a task_id."""
    import zarr

    store = zarr.open_group(f"{zarr_root}/{task_id}.zarr", mode="r")
    return tuple(store["charge_density_total"].shape)


def run_experiment(task_id: str, precision: str, cfg, zarr_root: str) -> dict:
    use_bf16 = precision == "bf16-mixed"

    # Initialize so all are in scope for the finally block even on early OOM
    model = optimizer = loss_fn = dataset = loader = None
    x = y = pred = loss = None

    # ---- build dataloader (single sample, no split needed) ----
    dataset = LargeGridZarrDataset(
        task_ids=[task_id],
        zarr_root=zarr_root,
        precision="f32",  # load as f32; autocast handles the cast
    )
    loader = DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_fn
    )

    # ---- build model & optimizer ----
    model = build_model(cfg)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg.lr))
    loss_fn = NormMAE().to(DEVICE)

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(DEVICE)

    epoch_times: list[float] = []
    fwd_times: list[float] = []
    bwd_times: list[float] = []
    oom = False
    peak_mem_mb: float | None = None

    try:
        for _epoch in range(EPOCHS):
            epoch_start = time.perf_counter()

            for batch in loader:
                x = batch["data"]
                y = batch["label"]

                # collate_fn may return a list when shapes vary
                if isinstance(x, list):
                    x = x[0].unsqueeze(0).to(DEVICE)
                    y = y[0].unsqueeze(0).to(DEVICE)
                else:
                    x = x.to(DEVICE)
                    y = y.to(DEVICE)

                optimizer.zero_grad()

                # ---- forward ----
                fwd_start = torch.cuda.Event(enable_timing=True)
                fwd_end = torch.cuda.Event(enable_timing=True)
                fwd_start.record()

                if use_bf16:
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        pred = model(x)
                        loss = loss_fn(pred, y)
                else:
                    pred = model(x)
                    loss = loss_fn(pred, y)

                fwd_end.record()
                torch.cuda.synchronize()
                fwd_ms = fwd_start.elapsed_time(fwd_end)

                # ---- backward ----
                bwd_start = torch.cuda.Event(enable_timing=True)
                bwd_end = torch.cuda.Event(enable_timing=True)
                bwd_start.record()
                loss.backward()
                bwd_end.record()
                torch.cuda.synchronize()
                bwd_ms = bwd_start.elapsed_time(bwd_end)

                optimizer.step()

                fwd_times.append(fwd_ms / 1000.0)
                bwd_times.append(bwd_ms / 1000.0)

            epoch_end = time.perf_counter()
            epoch_times.append(epoch_end - epoch_start)

        peak_mem_mb = torch.cuda.max_memory_allocated(DEVICE) / 1024**2

    except torch.cuda.OutOfMemoryError:
        oom = True
        peak_mem_mb = torch.cuda.max_memory_allocated(DEVICE) / 1024**2

    finally:
        del model, optimizer, loss_fn, dataset, loader, x, y, pred, loss
        gc.collect()
        torch.cuda.empty_cache()

    return {
        "task_id": task_id,
        "precision": precision,
        "oom": oom,
        "peak_mem_mb": round(peak_mem_mb, 1) if peak_mem_mb is not None else None,
        "epoch_times_s": [round(t, 2) for t in epoch_times],
        "fwd_times_s": [round(t, 3) for t in fwd_times],
        "bwd_times_s": [round(t, 3) for t in bwd_times],
        "avg_epoch_s": round(sum(epoch_times) / len(epoch_times), 2)
        if epoch_times
        else None,
        "avg_fwd_s": round(sum(fwd_times) / len(fwd_times), 3) if fwd_times else None,
        "avg_bwd_s": round(sum(bwd_times) / len(bwd_times), 3) if bwd_times else None,
    }


def get_gpu_info() -> dict:
    props = torch.cuda.get_device_properties(DEVICE)
    return {
        "name": props.name,
        "total_mem_gb": round(props.total_memory / 1024**3, 1),
        "major": props.major,
        "minor": props.minor,
        "multi_processor_count": props.multi_processor_count,
    }


def main():
    args = parse_args()
    args.results.parent.mkdir(parents=True, exist_ok=True)
    cfg = load_cfg(args.config)
    task_ids = args.task_ids if args.task_ids is not None else TASK_IDS

    gpu_info = get_gpu_info()

    # get grid shapes upfront
    grid_shapes = {}
    for tid in task_ids:
        try:
            grid_shapes[tid] = get_grid_shape(tid, args.zarr_root)
        except Exception as e:
            warnings.warn(f"could not read grid shape for {tid}: {e}", stacklevel=2)
            grid_shapes[tid] = None

    results = []
    for tid in task_ids:
        for prec in PRECISIONS:
            r = run_experiment(tid, prec, cfg, args.zarr_root)
            r["grid_shape"] = grid_shapes.get(tid)
            results.append(r)
            # save incrementally
            args.results.write_text(
                json.dumps({"gpu": gpu_info, "results": results}, indent=2)
            )


if __name__ == "__main__":
    main()
