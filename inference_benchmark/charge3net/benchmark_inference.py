"""
Benchmark ChargE3Net inference throughput in a production-like setting.

Predicts the FULL charge density grid for every structure in the MP test split
and records, per material:
    - atom-model forward time (cuda.Event)
    - probe-model forward time, summed over probe-count chunks (cuda.Event)
    - end-to-end wall-clock time including data load + graph construction
    - number of atoms, full grid voxel count, number of partials and chunks

Two layers of chunking are involved (both production-realistic):

  1. DensityGraphDataset splits a material's grid into partials of up to
     max_grid_size probes to bound graph-construction memory. Each partial is
     one DataLoader sample.
  2. Within one partial, if num_probes > max_predict_batch_probes, the forward
     pass itself is chunked via pred_utils.split_batch (atom representation is
     computed once and reused across probe chunks).

We report per-material totals so the headline metric is "wall time to predict
one full charge density grid on a single A100", which is the production
question being asked.

The distributed scaffold (DistributedEvalSampler + per-rank CSVs merged at the
end) is reused from loss_distribution_analysis so the run can be sharded
across multiple GPUs purely to finish faster — model is NOT wrapped in DDP,
each rank runs an independent forward, and the per-rank throughput is what
gets reported as "production-like single-GPU throughput".

Usage:
    python inference_benchmark/launch_benchmark.py
"""

from __future__ import annotations

import os
import sys
import time
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.distributed import barrier, destroy_process_group, init_process_group
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.charge3net.data.collate import collate_list_of_dicts
from src.charge3net.data.dataset import (
    DensityData,
    DensityGraphDataset,
    DistributedEvalSampler,
)
from src.charge3net.data.graph_construction import KdTreeGraphConstructor
from src.charge3net.data.split import split_data
from src.charge3net.models.e3 import E3DensityModel
from src.utils import predictions as pred_utils

# ─── Configuration ────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent

CHECKPOINT_PATH = PROJECT_ROOT / "models" / "charge3net_mp.pt"
# Self-owned dataset preprocessed from chg_datasets/rho_gga zarrs (test
# partition only) by inference_benchmark/preprocess_density.py. The same
# mpids are read directly from rho_gga/data by electrai's ResUNet runs,
# so the throughput comparison is apples-to-apples.
DATA_DIR = Path(
    "/scratch/gpfs/ROSENGROUP/bb9080/charge3net-benchmark/data_preprocessed_rho_gga"
)
DATA_ROOT = DATA_DIR / "filelist.txt"
SPLIT_FILE = DATA_DIR / "split.json"
GRID_SIZE_FILE = DATA_DIR / "probe_counts.csv"
OUTPUT_DIR = Path(__file__).resolve().parent / "results"

# Production-like inference: predict the FULL grid for the test split only.
SPLITS = ["test"]
CUTOFF = 4.0

# Caps probe count per partial sample to bound graph-construction memory.
# 1e7 is the project default in mp_data.yaml.
MAX_GRID_SIZE = int(1e7)

# Within one partial, further chunks the forward pass to fit GPU memory.
# 2500 is the trainer default.
MAX_PREDICT_BATCH_PROBES = 2500

# Discard timings from the first N structures per rank (CUDA / cuDNN warmup).
WARMUP_SAMPLES = 3

# Optional cap on materials per rank, for smoke-testing without burning the
# full wall time. Set the SHAKEDOWN_LIMIT env var (0 = no cap).
SHAKEDOWN_LIMIT = int(os.environ.get("SHAKEDOWN_LIMIT", 0))
# ──────────────────────────────────────────────────────────────────────────────


def collate_skip_none(batch, pin_memory=False):
    """Filter out None samples (failed reads), then collate the rest."""
    batch = [s for s in batch if s is not None]
    if len(batch) == 0:
        return None
    return collate_list_of_dicts(batch, pin_memory=pin_memory)


def _to_device(batch, device):
    return {
        k: (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v)
        for k, v in batch.items()
    }


@torch.no_grad()
def _timed_forward(model, batch, device):
    """
    Run the production-style forward pass with cuda.Event timing split between
    atom model (computed once) and probe model (chunked if needed).

    Returns dict with:
        atom_ms, probe_ms_total, num_chunks
    """
    atom_start = torch.cuda.Event(enable_timing=True)
    atom_end = torch.cuda.Event(enable_timing=True)

    num_probes = batch["num_probes"].item()

    if num_probes <= MAX_PREDICT_BATCH_PROBES:
        # Single-shot forward — atom + probe in one call.
        atom_start.record()
        out = model(batch)
        atom_end.record()
        torch.cuda.synchronize(device)
        # We can't cleanly separate atom vs probe in the single-shot case;
        # report the total under probe and leave atom=0 to mark the case.
        return {
            "atom_ms": 0.0,
            "probe_ms_total": atom_start.elapsed_time(atom_end),
            "num_chunks": 1,
            "first_chunk_preds": out.detach().float(),
        }

    # Chunked path matches trainer._test_step: atom representation once,
    # probe model per sub-batch. Process each sub_batch immediately —
    # pred_utils.split_batch mutates num_probes/num_probe_edges in place on
    # each yield, so materializing the generator (e.g. list(...)) breaks the
    # invariant and produces wrong sizes downstream.
    atom_repr = None
    atom_ms = 0.0
    probe_ms = 0.0
    num_chunks = 0
    first_chunk_preds = None  # kept on the first chunk only for sanity logging

    for sub_batch in pred_utils.split_batch(batch, MAX_PREDICT_BATCH_PROBES):
        if atom_repr is None:
            atom_start.record()
            atom_repr = model.atom_model(sub_batch)
            atom_end.record()
            torch.cuda.synchronize(device)
            atom_ms = atom_start.elapsed_time(atom_end)

        probe_start = torch.cuda.Event(enable_timing=True)
        probe_end = torch.cuda.Event(enable_timing=True)
        probe_start.record()
        out = model.probe_model(sub_batch, atom_repr)
        probe_end.record()
        torch.cuda.synchronize(device)
        probe_ms += probe_start.elapsed_time(probe_end)
        num_chunks += 1
        if first_chunk_preds is None:
            first_chunk_preds = out.detach().float()

    return {
        "atom_ms": atom_ms,
        "probe_ms_total": probe_ms,
        "num_chunks": num_chunks,
        "first_chunk_preds": first_chunk_preds,
    }


def run(rank, cfg, env):
    global_rank = env["group_rank"] * cfg["nprocs"] + rank
    world_size = env["world_size"]

    print(
        f"[Rank {global_rank}] init "
        f"(node_rank={env['group_rank']}, local_rank={rank}, world_size={world_size})"
    )

    torch.manual_seed(42)
    np.random.seed(42)

    distributed = world_size > 1
    if distributed:
        init_process_group(
            backend="nccl",
            init_method="env://",
            rank=global_rank,
            world_size=world_size,
        )
    torch.cuda.set_device(rank)
    device = f"cuda:{rank}"

    # Disable cuDNN autotune so first-of-shape kernels don't pay an
    # autotune cost mid-benchmark — atom representations and probe-edge
    # tensors vary in shape per material.
    torch.backends.cudnn.benchmark = False

    # Model — note: NOT wrapped in DDP. Each rank runs an independent forward
    # on its shard of the data; we measure single-GPU throughput per rank.
    model = E3DensityModel(
        num_interactions=3,
        num_neighbors=20,
        mul=500,
        lmax=4,
        cutoff=CUTOFF,
        basis="gaussian",
        num_basis=20,
    )
    checkpoint = torch.load(str(CHECKPOINT_PATH), map_location=device)
    state_dict = checkpoint.get("model", checkpoint.get("state_dict", checkpoint))
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    if global_rank == 0:
        print(f"Loaded checkpoint (step={checkpoint.get('step', '?')})")
        print(
            f"max_grid_size={MAX_GRID_SIZE}, max_predict_batch_probes={MAX_PREDICT_BATCH_PROBES}"
        )

    # Full-grid mode: num_probes=None + grid_size_file triggers splitting_mode
    # in DensityGraphDataset, which produces one partial per (file, probe_offset)
    # covering up to MAX_GRID_SIZE probes each.
    dataset = DensityData(str(DATA_ROOT))
    gc = KdTreeGraphConstructor(cutoff=CUTOFF, num_probes=None, disable_pbc=False)
    subsets = split_data(dataset, split_file=str(SPLIT_FILE))

    tmp_dir = OUTPUT_DIR / ".tmp"
    if global_rank == 0:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        tmp_dir.mkdir(parents=True, exist_ok=True)
    if distributed:
        barrier()

    for split_name in SPLITS:
        graph_ds = DensityGraphDataset(
            subsets[split_name],
            gc,
            grid_size_file=str(GRID_SIZE_FILE),
            max_grid_size=MAX_GRID_SIZE,
        )
        sampler = DistributedEvalSampler(graph_ds) if distributed else None
        dl = DataLoader(
            graph_ds,
            batch_size=1,
            sampler=sampler,
            num_workers=0,  # Single worker per rank so load_time is honest.
            collate_fn=partial(collate_skip_none, pin_memory=False),
        )

        if global_rank == 0:
            print(
                f"\nBenchmarking {split_name} "
                f"({len(graph_ds)} partials total, ~{len(dl)} per rank)..."
            )

        results = evaluate(model, dl, split_name, device, global_rank, len(dl))

        df = pd.DataFrame(results)
        df.to_csv(tmp_dir / f"{split_name}_rank{global_rank:04d}.csv", index=False)

        if distributed:
            barrier()

    if global_rank == 0:
        merge_and_summarize(tmp_dir)

    if distributed:
        destroy_process_group()


@torch.no_grad()
def evaluate(model, dataloader, split_name, device, global_rank, total):
    """One row per (filename, probe_offset). Aggregation by filename happens at merge."""
    results = []

    for i, batch in enumerate(dataloader):
        if batch is None:
            continue
        if SHAKEDOWN_LIMIT and i >= SHAKEDOWN_LIMIT:
            print(
                f"  [Rank {global_rank}] hit SHAKEDOWN_LIMIT={SHAKEDOWN_LIMIT}, stopping",
                flush=True,
            )
            break

        load_time = (
            float(batch["load_time"][0]) if "load_time" in batch else float("nan")
        )
        filename = (
            batch["filename"][0]
            if isinstance(batch["filename"], list)
            else batch["filename"]
        )

        # Atoms (count from atom_xyz). num_atoms is per-batch but with bs=1 it's a 1-tensor.
        if "num_nodes" in batch:
            num_atoms = int(batch["num_nodes"].sum().item())
        elif "atom_xyz" in batch:
            num_atoms = int(batch["atom_xyz"].shape[1])
        else:
            num_atoms = -1

        grid_shape = batch["grid_shape"][0].tolist()
        grid_voxels = int(np.prod([s for s in grid_shape if s > 0]))
        chunk_voxels = int(batch["num_probes"].item())
        probe_offset = int(batch["probe_offset"].item())
        partial_flag = bool(batch["partial"].item())

        try:
            wall_start = time.time()
            batch = _to_device(batch, device)
            timing = _timed_forward(model, batch, device)
            wall_end = time.time()
        except Exception as e:
            print(
                f"  [Rank {global_rank}] Error on {filename} (offset={probe_offset}): {e}"
            )
            continue

        forward_ms = timing["atom_ms"] + timing["probe_ms_total"]
        e2e_chunk_s = (wall_end - wall_start) + load_time
        is_warmup = i < WARMUP_SAMPLES

        # Sanity print on the very first material — surfaces a silent
        # weight-loading or normalization bug that wouldn't otherwise show
        # up in throughput numbers (all-zero / NaN preds, etc.).
        if i == 0 and global_rank == 0 and timing.get("first_chunk_preds") is not None:
            p = timing["first_chunk_preds"]
            print(
                f"  first prediction stats (chunk 0): "
                f"min={p.min().item():.4g} max={p.max().item():.4g} "
                f"mean={p.mean().item():.4g} std={p.std().item():.4g}"
            )

        results.append(
            {
                "filename": filename,
                "split": split_name,
                "probe_offset": probe_offset,
                "partial": partial_flag,
                "num_atoms": num_atoms,
                "grid_voxels": grid_voxels,
                "chunk_voxels": chunk_voxels,
                "num_chunks": timing["num_chunks"],
                "atom_ms": timing["atom_ms"],
                "probe_ms": timing["probe_ms_total"],
                "forward_ms": forward_ms,
                "load_s": load_time,
                "e2e_chunk_s": e2e_chunk_s,
                "warmup": is_warmup,
            }
        )

        if ((i + 1) % 25 == 0 or i == 0) and global_rank == 0:
            print(
                f"  [{i + 1}/{total}] {filename} "
                f"voxels={chunk_voxels} chunks={timing['num_chunks']} "
                f"fwd={forward_ms:.1f}ms e2e={e2e_chunk_s:.2f}s"
                f"{' (warmup)' if is_warmup else ''}"
            )

    return results


def merge_and_summarize(tmp_dir):
    import shutil

    chunk_files = sorted(tmp_dir.glob("*.csv"))
    dfs = [pd.read_csv(f) for f in chunk_files if os.path.getsize(f) > 0]
    if not dfs:
        print("No results to merge.")
        return

    df_partials = pd.concat(dfs, ignore_index=True)
    df_partials.to_csv(OUTPUT_DIR / "throughput_partials.csv", index=False)

    # Aggregate to one row per material. We must drop ENTIRE materials whose
    # any partial fell inside the warmup window — otherwise materials whose
    # first partial is warmup but whose later partials aren't would have
    # their per-material forward_ms summed without that warmup partial,
    # biasing the per-material total low. (For the typical case where each
    # material is one partial this collapses to "drop the first N materials
    # per rank", which matches the intended semantic of warmup.)
    warmup_files = set(
        df_partials.loc[df_partials["warmup"], "filename"].unique().tolist()
    )
    if warmup_files:
        print(f"Dropping {len(warmup_files)} materials with warmup partials")
    df = df_partials[~df_partials["filename"].isin(warmup_files)].copy()
    by_file = df.groupby(["filename", "split"], as_index=False).agg(
        num_atoms=("num_atoms", "first"),
        grid_voxels=("grid_voxels", "first"),
        num_partials=("probe_offset", "count"),
        num_chunks_total=("num_chunks", "sum"),
        atom_ms=("atom_ms", "sum"),
        probe_ms=("probe_ms", "sum"),
        forward_ms=("forward_ms", "sum"),
        load_s=("load_s", "sum"),
        e2e_s=("e2e_chunk_s", "sum"),
    )
    by_file["voxels_per_sec_forward"] = by_file["grid_voxels"] / (
        by_file["forward_ms"] / 1000.0
    )
    by_file["voxels_per_sec_e2e"] = by_file["grid_voxels"] / by_file["e2e_s"]
    by_file.to_csv(OUTPUT_DIR / "throughput_by_material.csv", index=False)

    print("\n" + "=" * 70)
    print("THROUGHPUT SUMMARY (per-material, warmup excluded)")
    print("=" * 70)
    summary = (
        by_file.groupby("split")
        .agg(
            n_materials=("filename", "count"),
            forward_ms_median=("forward_ms", "median"),
            forward_ms_p95=("forward_ms", lambda s: s.quantile(0.95)),
            e2e_s_median=("e2e_s", "median"),
            e2e_s_p95=("e2e_s", lambda s: s.quantile(0.95)),
            voxels_per_sec_forward_median=("voxels_per_sec_forward", "median"),
            voxels_per_sec_e2e_median=("voxels_per_sec_e2e", "median"),
        )
        .round(3)
    )
    print(summary.to_string())
    print(f"\nResults saved to {OUTPUT_DIR}/")
    print("  - throughput_partials.csv   (one row per partial chunk)")
    print("  - throughput_by_material.csv (one row per full-grid material)")

    shutil.rmtree(tmp_dir)
