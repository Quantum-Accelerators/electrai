"""Standalone nn.Conv3d peak-memory benchmark across dtypes.

Fixed tensor (consistent across all torch versions):
    Input:  (N=1, C_in=32, D=64, H=64, W=64)
    Conv3d: in=32, out=32, kernel=5, padding=2, padding_mode='zeros'

Collects:
    - torch / CUDA / cuDNN versions
    - GPU name and total VRAM
    - Forward and backward peak GPU memory (MB) for float32 and bfloat16
    - Forward and backward wall-clock time (ms)

Usage:
    python benchmark_conv3d_memory.py --output results.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

# ── Fixed benchmark configuration ─────────────────────────────────────────────
N, CIN, COUT = 1, 32, 32
D = H = W = 64
K = 5
PADDING = K // 2
WARMUP_ITERS = 3
BENCH_ITERS = 5  # average over multiple runs for stable timing


# ── Memory helpers ────────────────────────────────────────────────────────────


def reset_mem():
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def peak_mb() -> float:
    return torch.cuda.max_memory_allocated() / 1024**2


# ── Per-dtype measurement ─────────────────────────────────────────────────────


def bench_dtype(dtype_str: str, device: torch.device) -> dict:
    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16}[dtype_str]

    torch.manual_seed(42)
    conv = nn.Conv3d(CIN, COUT, K, padding=PADDING).to(device=device, dtype=dtype)
    x_base = torch.randn(N, CIN, D, H, W, device=device, dtype=dtype)

    # ── Warmup (no grad, no memory tracking) ──────────────────────────────────
    for _ in range(WARMUP_ITERS):
        with torch.no_grad():
            _ = conv(x_base)
    torch.cuda.synchronize()

    # ── Forward pass ──────────────────────────────────────────────────────────
    fwd_peaks, fwd_times = [], []
    for _ in range(BENCH_ITERS):
        x = x_base.detach().requires_grad_(True)
        reset_mem()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = conv(x)
        torch.cuda.synchronize()
        fwd_times.append(time.perf_counter() - t0)
        fwd_peaks.append(peak_mb())

    # ── Backward pass ─────────────────────────────────────────────────────────
    bwd_peaks, bwd_times = [], []
    for _ in range(BENCH_ITERS):
        x = x_base.detach().requires_grad_(True)
        out = conv(x)  # fresh forward to build the computation graph
        reset_mem()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out.sum().backward()
        torch.cuda.synchronize()
        bwd_times.append(time.perf_counter() - t0)
        bwd_peaks.append(peak_mb())

    def avg(lst: list) -> float:
        return round(sum(lst) / len(lst), 2)

    return {
        "fwd_peak_mb": avg(fwd_peaks),
        "bwd_peak_mb": avg(bwd_peaks),
        "fwd_time_ms": round(avg(fwd_times) * 1e3, 3),
        "bwd_time_ms": round(avg(bwd_times) * 1e3, 3),
    }


# ── Entry point ───────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Conv3d memory benchmark")
    parser.add_argument("--output", required=True, help="Path to write JSON results")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        result = {"error": "CUDA not available"}
        with Path.open(args.output, "w") as f:
            json.dump(result, f, indent=2)
        sys.exit(1)

    device = torch.device("cuda:0")
    gpu_props = torch.cuda.get_device_properties(0)

    # Decode cuDNN version integer.
    # cuDNN < 9 : MAJOR*1000 + MINOR*100 + PATCH  (e.g. 8904 → "8.9.4")
    # cuDNN 9+  : MAJOR*10000 + MINOR*1000 + PATCH*100 + BUILD (e.g. 91002 → "9.1.0.2")
    raw_cudnn = torch.backends.cudnn.version()
    if not isinstance(raw_cudnn, int):
        cudnn_str = str(raw_cudnn)
    elif raw_cudnn >= 10000:
        major = raw_cudnn // 10000
        minor = (raw_cudnn % 10000) // 1000
        patch = (raw_cudnn % 1000) // 100
        build = raw_cudnn % 100
        cudnn_str = f"{major}.{minor}.{patch}.{build}"
    else:
        major = raw_cudnn // 1000
        minor = (raw_cudnn % 1000) // 100
        patch = raw_cudnn % 100
        cudnn_str = f"{major}.{minor}.{patch}"

    results = {
        "torch_version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn_version": cudnn_str,
        "gpu_name": gpu_props.name,
        "gpu_total_memory_gb": round(gpu_props.total_memory / 1024**3, 1),
        "gpu_sm_count": gpu_props.multi_processor_count,
        "tensor_shape": [N, CIN, D, H, W],
        "conv_config": f"Conv3d(in={CIN}, out={COUT}, k={K}, padding={PADDING})",
        "bench_iters": BENCH_ITERS,
        "measurements": {},
    }

    for dtype in ("float32", "bfloat16"):
        try:
            results["measurements"][dtype] = bench_dtype(dtype, device)
        except Exception as exc:
            results["measurements"][dtype] = {"error": str(exc)}

    with Path.open(args.output, "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
