"""
Compare per-sample benchmark results across two GPUs (e.g. A100 vs H200).

Reads two JSON files produced by benchmark_precision.py and prints
a markdown report section with a combined table (memory, time, and ratios).

Usage:
    uv run python scripts/benchmark_gpus.py \
        --gpu1 benchmark_results/per_task_precision.json \
        --gpu2 benchmark_results/per_task_precision_h200.json \
        --out  benchmark_results/gpu_comparison.md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def load(path: Path) -> tuple[dict, list[dict]]:
    d = json.loads(path.read_text())
    return d["gpu"], d["results"]


def by_key(results: list[dict]) -> dict[tuple[str, str], dict]:
    return {(r["task_id"], r["precision"]): r for r in results}


def ratio(a, b):
    """Return a/b ratio string."""
    if a is None or b is None:
        return "—"
    return f"{a / b:.2f}x"


def status(r):
    if r is None:
        return "—"
    return "❌ OOM" if r["oom"] else "✅"


def peak_gb(r):
    if r is None or r["oom"] or r["peak_mem_mb"] is None:
        return None
    return round(r["peak_mem_mb"] / 1024, 2)


def epoch_s(r):
    if r is None or r["oom"]:
        return None
    return r["avg_epoch_s"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gpu1", type=Path, required=True, help="First GPU results JSON (reference)"
    )
    parser.add_argument(
        "--gpu2", type=Path, required=True, help="Second GPU results JSON"
    )
    parser.add_argument(
        "--out", type=Path, default=None, help="Output markdown file (default: stdout)"
    )
    args = parser.parse_args()

    gpu1_info, res1 = load(args.gpu1)
    gpu2_info, res2 = load(args.gpu2)

    idx1 = by_key(res1)
    idx2 = by_key(res2)

    gpu1_name = gpu1_info["name"]
    gpu2_name = gpu2_info["name"]

    # grid shape lookup
    grid_shapes = {}
    for r in res1 + res2:
        if r.get("grid_shape"):
            grid_shapes[r["task_id"]] = r["grid_shape"]

    def shape_str(tid):
        s = grid_shapes.get(tid)
        return f"{s[0]} x {s[1]} x {s[2]}" if s else "—"

    def voxels(tid):
        s = grid_shapes.get(tid)
        return f"{s[0] * s[1] * s[2] / 1e6:.1f} M" if s else "—"

    all_task_ids = sorted({k[0] for k in set(idx1) | set(idx2)})

    lines = []
    a = lines.append

    a("| GPU | Model | VRAM |")
    a("|-----|-------|:----:|")
    a(f"| GPU 1 (reference) | {gpu1_name} | {gpu1_info['total_mem_gb']} GB |")
    a(f"| GPU 2 | {gpu2_name} | {gpu2_info['total_mem_gb']} GB |")
    a("")

    for prec in ["f32", "bf16-mixed"]:
        a(f"### {prec}\n")
        a(
            f"| Task ID | Grid shape | Voxels "
            f"| {gpu1_name} status | {gpu1_name} peak (GB) | {gpu1_name} epoch (s) "
            f"| {gpu2_name} status | {gpu2_name} peak (GB) | {gpu2_name} epoch (s) "
            f"| Peak mem ratio (GPU1/GPU2) | Epoch time ratio (GPU1/GPU2) |"
        )
        a(
            "|---------|-----------|:------:"
            "|:---------:|:-------------------:|:--------------------:"
            "|:---------:|:-------------------:|:--------------------:"
            "|:-------------------------:|:----------------------------:|"
        )

        for tid in all_task_ids:
            key = (tid, prec)
            r1 = idx1.get(key)
            r2 = idx2.get(key)

            p1, p2 = peak_gb(r1), peak_gb(r2)
            e1, e2 = epoch_s(r1), epoch_s(r2)

            p1_str = "—" if p1 is None else f"{p1:.2f}"
            p2_str = "—" if p2 is None else f"{p2:.2f}"
            e1_str = "—" if e1 is None else f"{e1:.2f}"
            e2_str = "—" if e2 is None else f"{e2:.2f}"

            a(
                f"| {tid} | {shape_str(tid)} | {voxels(tid)} "
                f"| {status(r1)} | {p1_str} | {e1_str} "
                f"| {status(r2)} | {p2_str} | {e2_str} "
                f"| {ratio(p1, p2)} | {ratio(e1, e2)} |"
            )
        a("")

    output = "\n".join(lines)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(output)
    else:
        sys.stdout.write(output + "\n")


if __name__ == "__main__":
    main()
