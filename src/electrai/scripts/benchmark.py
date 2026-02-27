"""Benchmark script for ElectrAI ResUNet3D compute estimation.

Single-GPU script with three measurement phases:
  1. Data loading throughput (CPU-only)
  2. GPU compute (forward + backward + optimizer)
  3. End-to-end wall clock (data loading + GPU together)

Usage:
    uv run python src/electrai/scripts/benchmark.py \
        --config src/electrai/configs/MP/config_benchmark.yaml
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import torch
import yaml
from hydra.utils import instantiate

from electrai.model.loss.charge import NormMAE


def get_metadata(device: torch.device) -> dict:
    """Collect environment metadata."""
    meta = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "hostname": platform.node(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(device)
        meta.update(
            {
                "gpu_name": props.name,
                "gpu_memory_gb": round(props.total_memory / 1e9, 2),
                "cuda_version": torch.version.cuda,
                "cudnn_version": torch.backends.cudnn.version(),
            }
        )
    return meta


def get_model_summary(model: torch.nn.Module) -> dict:
    """Compute model parameter count and size."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    param_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (
        1024 * 1024
    )
    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "model_size_mb": round(param_size_mb, 3),
    }


def shape_key(tensor: torch.Tensor) -> str:
    """Return spatial dims as a string key for shape bucketing."""
    return "x".join(str(d) for d in tensor.shape[2:])


def measure_file_sizes(datamodule, max_samples: int = 200) -> dict:
    """Sample CHGCAR file sizes from the dataset for storage estimates."""
    sizes_mb = []
    try:
        dataset = datamodule.train_set
        # Unwrap Subset to get underlying RhoData
        raw_dataset = dataset.dataset if hasattr(dataset, "dataset") else dataset
        root = Path(raw_dataset.root)
        indices = raw_dataset.member_list
        for idx in indices[:max_samples]:
            for subdir in ("data", "label"):
                fpath = root / subdir / f"{idx}.CHGCAR"
                if fpath.exists():
                    sizes_mb.append(fpath.stat().st_size / (1024 * 1024))
    except Exception:
        return {}
    if not sizes_mb:
        return {}
    sizes_sorted = sorted(sizes_mb)
    n = len(sizes_sorted)
    return {
        "n_files_sampled": n,
        "mean_mb": round(sum(sizes_sorted) / n, 2),
        "median_mb": round(sizes_sorted[n // 2], 2),
        "min_mb": round(sizes_sorted[0], 2),
        "max_mb": round(sizes_sorted[-1], 2),
    }


def phase1_data_loading(datamodule, n_steps: int, warmup_steps: int) -> dict:
    """Phase 1: Measure data loading throughput (CPU-only).

    Times the next() call on the DataLoader iterator, which captures
    the full worker process time (file I/O, pymatgen parsing, tensor
    conversion, collation).
    """
    print(f"\n{'=' * 60}")
    print(f"Phase 1: Data Loading Throughput ({n_steps} steps)")
    print(f"{'=' * 60}")

    datamodule.setup(stage="fit")
    loader = datamodule.train_dataloader()
    iterator = iter(loader)

    times = []
    shapes = []
    total_steps = warmup_steps + n_steps

    for step in range(total_steps):
        try:
            t0 = time.perf_counter()
            batch = next(iterator)
            t1 = time.perf_counter()
        except StopIteration:
            break

        if step >= warmup_steps:
            data = batch["data"]
            elapsed = t1 - t0
            times.append(elapsed)

            if isinstance(data, torch.Tensor):
                shapes.append(data.shape[2:])
            elif isinstance(data, list):
                shapes.extend(d.shape[1:] for d in data)

        if (step + 1) % 20 == 0:
            print(f"  Step {step + 1}/{total_steps}...")

    if not times:
        return {"error": "No data loading measurements collected"}

    times_arr = sorted(times)
    n = len(times_arr)
    mean_time = sum(times_arr) / n
    median_time = times_arr[n // 2]
    p95_time = times_arr[int(n * 0.95)]

    # Shape histogram
    shape_counts = defaultdict(int)
    for s in shapes:
        key = "x".join(str(d) for d in s)
        shape_counts[key] += 1

    # Measure file sizes for storage estimates
    file_sizes = measure_file_sizes(datamodule)

    results = {
        "n_samples": n,
        "num_workers": loader.num_workers,
        "mean_time_s": round(mean_time, 4),
        "median_time_s": round(median_time, 4),
        "p95_time_s": round(p95_time, 4),
        "samples_per_sec": round(1.0 / mean_time, 2) if mean_time > 0 else 0,
        "shape_distribution": dict(
            sorted(shape_counts.items(), key=lambda x: x[1], reverse=True)
        ),
        "file_sizes": file_sizes,
    }

    print(f"  Mean load time:  {mean_time:.4f}s")
    print(f"  Median:          {median_time:.4f}s")
    print(f"  P95:             {p95_time:.4f}s")
    print(f"  Throughput:      {results['samples_per_sec']:.2f} samples/sec")
    print(f"  Unique shapes:   {len(shape_counts)}")
    if file_sizes:
        print(f"  Mean file size:  {file_sizes['mean_mb']:.1f} MB")

    return results


def phase2_gpu_compute(
    model: torch.nn.Module,
    datamodule,
    device: torch.device,
    n_steps: int,
    warmup_steps: int,
    lr: float,
) -> dict:
    """Phase 2: Measure GPU compute (forward + backward + optimizer).

    Pre-loads real samples from the dataset, groups by shape,
    and times forward/loss/backward/optimizer with CUDA events.
    """
    print(f"\n{'=' * 60}")
    print(f"Phase 2: GPU Compute Timing ({n_steps} steps)")
    print(f"{'=' * 60}")

    loss_fn = NormMAE()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Pre-load samples from the dataloader, grouped by shape
    if not hasattr(datamodule, "train_set"):
        datamodule.setup(stage="fit")
    loader = datamodule.train_dataloader()

    shape_buckets = defaultdict(list)
    max_preload = warmup_steps + n_steps + 10
    count = 0
    for batch in loader:
        data = batch["data"]
        label = batch["label"]
        if isinstance(data, list):
            for d_i, l_i in zip(data, label, strict=True):
                key = shape_key(d_i.unsqueeze(0))
                shape_buckets[key].append(
                    (d_i.unsqueeze(0).to(device), l_i.unsqueeze(0).to(device))
                )
                count += 1
        else:
            key = shape_key(data)
            shape_buckets[key].append((data.to(device), label.to(device)))
            count += 1
        if count >= max_preload:
            break

    print(f"  Pre-loaded {count} samples across {len(shape_buckets)} shape buckets")

    results_by_shape = {}
    overall_times = []
    peak_memory_by_shape = {}

    # Process shapes smallest-to-largest so peak_reserved_mb is not inflated
    # by cached allocations from earlier, larger shapes
    def _shape_volume(key: str) -> int:
        dims = key.split("x")
        vol = 1
        for d in dims:
            vol *= int(d)
        return vol

    sorted_buckets = sorted(shape_buckets.items(), key=lambda kv: _shape_volume(kv[0]))

    for sk, samples in sorted_buckets:
        torch.cuda.reset_peak_memory_stats(device)
        times = []

        for step, (data, label) in enumerate(samples):
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            model.train()
            optimizer.zero_grad()

            start_event.record()
            pred = model(data)
            loss = loss_fn(pred, label)
            loss.backward()
            optimizer.step()
            end_event.record()

            torch.cuda.synchronize()
            elapsed_ms = start_event.elapsed_time(end_event)

            if step >= warmup_steps:
                times.append(elapsed_ms)
                overall_times.append(elapsed_ms)

        if times:
            times_sorted = sorted(times)
            n = len(times_sorted)
            results_by_shape[sk] = {
                "n_samples": n,
                "mean_ms": round(sum(times_sorted) / n, 2),
                "median_ms": round(times_sorted[n // 2], 2),
                "min_ms": round(times_sorted[0], 2),
                "max_ms": round(times_sorted[-1], 2),
            }

        peak_alloc = torch.cuda.max_memory_allocated(device) / (1024**2)
        peak_reserved = torch.cuda.max_memory_reserved(device) / (1024**2)
        peak_memory_by_shape[sk] = {
            "peak_allocated_mb": round(peak_alloc, 1),
            "peak_reserved_mb": round(peak_reserved, 1),
        }
        print(
            f"  Shape {sk}: {len(times)} steps, "
            f"mean={results_by_shape.get(sk, {}).get('mean_ms', 'N/A')}ms, "
            f"peak_mem={peak_alloc:.0f}MB"
        )

    # Overall stats
    overall_summary = {}
    if overall_times:
        overall_sorted = sorted(overall_times)
        n = len(overall_sorted)
        mean_ms = sum(overall_sorted) / n
        overall_summary = {
            "n_total_steps": n,
            "mean_ms": round(mean_ms, 2),
            "median_ms": round(overall_sorted[n // 2], 2),
            "p95_ms": round(overall_sorted[int(n * 0.95)], 2),
            "samples_per_sec": round(1000.0 / mean_ms, 2) if mean_ms > 0 else 0,
        }
        print(
            f"\n  Overall: mean={mean_ms:.2f}ms, "
            f"throughput={overall_summary['samples_per_sec']:.2f} samples/sec"
        )

    return {
        "by_shape": results_by_shape,
        "overall": overall_summary,
        "peak_memory": peak_memory_by_shape,
    }


def phase3_end_to_end(
    model: torch.nn.Module,
    datamodule,
    device: torch.device,
    n_steps: int,
    warmup_steps: int,
    lr: float,
) -> dict:
    """Phase 3: End-to-end wall clock (real DataLoader + GPU compute).

    Measures how training actually runs with data loading and compute together.
    """
    print(f"\n{'=' * 60}")
    print(f"Phase 3: End-to-End Wall Clock ({n_steps} steps)")
    print(f"{'=' * 60}")

    loss_fn = NormMAE()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()

    if not hasattr(datamodule, "train_set"):
        datamodule.setup(stage="fit")
    loader = datamodule.train_dataloader()

    times = []
    total_steps = warmup_steps + n_steps

    for step, batch in enumerate(loader):
        t0 = time.perf_counter()

        data = batch["data"]
        label = batch["label"]

        optimizer.zero_grad()

        if isinstance(data, list):
            losses = []
            for d_i, l_i in zip(data, label, strict=True):
                d_gpu = d_i.unsqueeze(0).to(device)
                l_gpu = l_i.unsqueeze(0).to(device)
                pred = model(d_gpu)
                losses.append(loss_fn(pred, l_gpu))
            loss = torch.stack(losses).mean()
        else:
            data = data.to(device)
            label = label.to(device)
            pred = model(data)
            loss = loss_fn(pred, label)

        loss.backward()
        optimizer.step()

        torch.cuda.synchronize()
        t1 = time.perf_counter()

        if step >= warmup_steps:
            times.append(t1 - t0)

        if step + 1 >= total_steps:
            break

        if (step + 1) % 10 == 0:
            print(f"  Step {step}/{total_steps}...")

    if not times:
        return {"error": "No end-to-end measurements collected"}

    times_sorted = sorted(times)
    n = len(times_sorted)
    mean_time = sum(times_sorted) / n
    median_time = times_sorted[n // 2]
    p95_time = times_sorted[int(n * 0.95)]

    results = {
        "n_steps": n,
        "mean_time_s": round(mean_time, 4),
        "median_time_s": round(median_time, 4),
        "p95_time_s": round(p95_time, 4),
        "samples_per_sec": round(1.0 / mean_time, 2) if mean_time > 0 else 0,
    }

    print(f"  Mean step time:  {mean_time:.4f}s")
    print(f"  Median:          {median_time:.4f}s")
    print(f"  P95:             {p95_time:.4f}s")
    print(f"  Throughput:      {results['samples_per_sec']:.2f} samples/sec")

    # Bottleneck detection
    # Compare data-loading-only throughput vs compute-only throughput
    # This is done in the analysis script with both phase results

    return results


def run_benchmark(config_path: str) -> None:
    """Run all benchmark phases and save results."""
    # Load config
    with Path(config_path).open() as f:
        cfg_dict = yaml.safe_load(f)
    cfg = SimpleNamespace(**cfg_dict)
    bench_cfg = cfg.benchmark

    warmup_steps = bench_cfg["warmup_steps"]
    output_file = bench_cfg["output_file"]

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(device)}")

    # Collect metadata
    metadata = get_metadata(device)
    print(f"Metadata: {json.dumps(metadata, indent=2)}")

    # Instantiate model
    model = instantiate(cfg.model).to(device)
    model_summary = get_model_summary(model)
    print(
        f"\nModel: {model_summary['total_params']:,} params "
        f"({model_summary['model_size_mb']:.2f} MB)"
    )

    # Instantiate datamodule
    datamodule = instantiate(cfg.data)

    # Save initial model state so we can restore between phases
    initial_state = {k: v.clone() for k, v in model.state_dict().items()}

    # Phase 1: Data Loading
    phase1_results = phase1_data_loading(
        datamodule, n_steps=bench_cfg["data_loading_steps"], warmup_steps=warmup_steps
    )

    # Phase 2: GPU Compute
    phase2_results = phase2_gpu_compute(
        model,
        datamodule,
        device,
        n_steps=bench_cfg["gpu_compute_steps"],
        warmup_steps=warmup_steps,
        lr=float(cfg.lr),
    )

    # Restore model to initial state before Phase 3 so it starts clean
    # (Phase 2 optimizer steps corrupt weights for timing-only purposes)
    model.load_state_dict(initial_state)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Phase 3: End-to-End
    phase3_results = phase3_end_to_end(
        model,
        datamodule,
        device,
        n_steps=bench_cfg["end_to_end_steps"],
        warmup_steps=warmup_steps,
        lr=float(cfg.lr),
    )

    # Assemble results
    results = {
        "metadata": metadata,
        "model_summary": model_summary,
        "config": {
            "model": cfg_dict.get("model", {}),
            "data": {
                k: v for k, v in cfg_dict.get("data", {}).items() if k != "_target_"
            },
            "benchmark": bench_cfg,
            "epochs": cfg_dict.get("epochs", 50),
        },
        "phase1_data_loading": phase1_results,
        "phase2_gpu_compute": phase2_results,
        "phase3_end_to_end": phase3_results,
    }

    # Save
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"Benchmark complete. Results saved to: {output_path}")
    print(f"{'=' * 60}")


def main():
    parser = argparse.ArgumentParser(description="ElectrAI Benchmark")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to benchmark YAML config"
    )
    args = parser.parse_args()
    run_benchmark(args.config)


if __name__ == "__main__":
    main()
