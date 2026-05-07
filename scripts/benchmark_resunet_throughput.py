"""
Benchmark ResUNet3D inference throughput on a fixed set of MP materials.

Designed to be a drop-in match for the charge3net throughput benchmark
(charge3net-benchmark/inference_benchmark/benchmark_inference.py): same
materials, same per-material schema, so the two output CSVs can be joined
on `filename` for a direct apples-to-apples comparison.

Usage:
    uv run --no-sync python scripts/benchmark_resunet_throughput.py \\
        --checkpoint <path/to/last.ckpt> \\
        --filelist /scratch/.../charge3net-benchmark/data_preprocessed/filelist.txt \\
        --data-root /scratch/.../chg_datasets/dataset_4 \\
        --output /scratch/.../charge3net-benchmark/resunet_results

ResUNet's natural production input is the low-res CHGCAR at
<data-root>/data/<mpid>.CHGCAR — same as electrai's training pipeline. The
model upsamples to a high-res grid; we time the forward pass per material.

Output: <output>/throughput_by_material.csv with columns matching the
charge3net benchmark (filename, num_atoms, grid_voxels, forward_ms,
load_s, e2e_s, voxels_per_sec_forward, voxels_per_sec_e2e, warmup).
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch
from pymatgen.io.vasp import Chgcar

from electrai.model.resunet import ResUNet3D

PRECISION = {"f32": torch.float32, "f16": torch.float16, "bf16": torch.bfloat16}


def load_state_dict_flexibly(
    model: torch.nn.Module, ckpt_path: Path, device: str
) -> dict:
    """Strip common Lightning prefixes ('model.', 'model.backbone.') and load."""
    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    sd = ckpt.get("state_dict", ckpt.get("model", ckpt))
    cleaned = {}
    for k, v in sd.items():
        stripped = k
        for prefix in ("model.backbone.", "model."):
            if stripped.startswith(prefix):
                stripped = stripped[len(prefix) :]
                break
        cleaned[stripped] = v
    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    return {
        "missing": list(missing),
        "unexpected": list(unexpected),
        "step": ckpt.get("global_step", ckpt.get("step", "?")),
    }


def load_chgcar_pair(input_dir: Path, label_dir: Path, mpid: str, dtype: torch.dtype):
    """Match electrai.dataloader.utils.load_chgcar normalization. Paths are
    explicit because dataset_4 has a double-level layout
    (data/data/<mpid>.CHGCAR, label/label/<mpid>.CHGCAR)."""
    in_chg = Chgcar.from_file(str(input_dir / f"{mpid}.CHGCAR"))
    lab_chg = Chgcar.from_file(str(label_dir / f"{mpid}.CHGCAR"))
    data = in_chg.data["total"] / in_chg.structure.lattice.volume
    label = lab_chg.data["total"] / lab_chg.structure.lattice.volume
    # ResUNet expects (B, C, X, Y, Z) — bs=1, single channel.
    data = torch.tensor(data, dtype=dtype).unsqueeze(0).unsqueeze(0)
    label = torch.tensor(label, dtype=dtype).unsqueeze(0).unsqueeze(0)
    num_atoms = len(lab_chg.structure)
    return data, label, num_atoms


@torch.no_grad()
def benchmark(args):
    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires a GPU.")
    device = "cuda"

    model = ResUNet3D(
        in_channels=1,
        out_channels=1,
        n_channels=args.n_channels,
        depth=args.depth,
        n_residual_blocks=args.n_residual_blocks,
        kernel_size=args.kernel_size,
        use_checkpoint=False,
    )
    info = load_state_dict_flexibly(model, Path(args.checkpoint), device)
    print(
        f"Loaded checkpoint (step={info['step']}, "
        f"missing={len(info['missing'])}, unexpected={len(info['unexpected'])})",
        flush=True,
    )
    if info["missing"][:3]:
        print(f"  missing (first 3): {info['missing'][:3]}", flush=True)
    if info["unexpected"][:3]:
        print(f"  unexpected (first 3): {info['unexpected'][:3]}", flush=True)
    model.to(device).eval()

    mpids = [m for m in Path(args.filelist).read_text().splitlines() if m.strip()]
    if args.shakedown_limit > 0:
        mpids = mpids[: args.shakedown_limit]
    print(f"Benchmarking {len(mpids)} materials...", flush=True)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "throughput_by_material.csv"

    dtype = PRECISION[args.precision]
    input_dir = Path(args.input_dir)
    label_dir = Path(args.label_dir)

    headers = [
        "filename",
        "num_atoms",
        "grid_voxels",
        "forward_ms",
        "load_s",
        "e2e_s",
        "voxels_per_sec_forward",
        "voxels_per_sec_e2e",
        "warmup",
    ]
    rows = []

    for i, mpid in enumerate(mpids):
        try:
            wall_start = time.time()
            data, _label, num_atoms = load_chgcar_pair(
                input_dir, label_dir, mpid, dtype
            )
            data = data.to(device, non_blocking=True)
            grid_voxels = int(np.prod(data.shape[2:]))
            load_s = time.time() - wall_start

            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            _ = model(data)
            end.record()
            torch.cuda.synchronize(device)
            forward_ms = start.elapsed_time(end)
            e2e_s = time.time() - wall_start

            is_warmup = i < args.warmup_samples
            rows.append(
                {
                    "filename": mpid,
                    "num_atoms": num_atoms,
                    "grid_voxels": grid_voxels,
                    "forward_ms": forward_ms,
                    "load_s": load_s,
                    "e2e_s": e2e_s,
                    "voxels_per_sec_forward": grid_voxels / (forward_ms / 1000.0),
                    "voxels_per_sec_e2e": grid_voxels / e2e_s,
                    "warmup": is_warmup,
                }
            )
            if (i + 1) % 25 == 0 or i == 0:
                print(
                    f"  [{i + 1}/{len(mpids)}] {mpid} "
                    f"voxels={grid_voxels} fwd={forward_ms:.1f}ms e2e={e2e_s:.2f}s"
                    f"{' (warmup)' if is_warmup else ''}",
                    flush=True,
                )
        except Exception as e:
            print(f"  [{i + 1}/{len(mpids)}] ERROR on {mpid}: {e}", flush=True)

    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=headers)
        w.writeheader()
        for row in rows:
            w.writerow(row)
    print(f"Wrote {len(rows)} rows to {csv_path}", flush=True)

    valid = [r for r in rows if not r["warmup"]]
    if valid:
        fwd = sorted(r["forward_ms"] for r in valid)
        e2e = sorted(r["e2e_s"] for r in valid)

        def p50(xs):
            return xs[len(xs) // 2]

        def p95(xs):
            return xs[int(0.95 * len(xs))]

        print("\n" + "=" * 70)
        print(f"THROUGHPUT SUMMARY (n={len(valid)}, warmup excluded)")
        print("=" * 70)
        print(f"  forward_ms  median={p50(fwd):.1f}  p95={p95(fwd):.1f}")
        print(f"  e2e_s       median={p50(e2e):.3f}  p95={p95(e2e):.3f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--filelist", required=True)
    p.add_argument(
        "--input-dir",
        required=True,
        help="Directory containing low-res input <mpid>.CHGCAR files",
    )
    p.add_argument(
        "--label-dir",
        required=True,
        help="Directory containing high-res label <mpid>.CHGCAR files",
    )
    p.add_argument("--output", required=True)
    p.add_argument(
        "--shakedown-limit",
        type=int,
        default=0,
        help="0 = no cap; otherwise process at most this many materials",
    )
    p.add_argument("--warmup-samples", type=int, default=3)
    p.add_argument("--precision", default="f32", choices=list(PRECISION))
    # ResUNet3D config — must match the checkpoint's training config.
    # Defaults track configs/MP/config_resunet_baseline.yaml.
    p.add_argument("--n-channels", type=int, default=32)
    p.add_argument("--n-residual-blocks", type=int, default=1)
    p.add_argument("--kernel-size", type=int, default=5)
    p.add_argument("--depth", type=int, default=2)
    args = p.parse_args()
    benchmark(args)


if __name__ == "__main__":
    main()
