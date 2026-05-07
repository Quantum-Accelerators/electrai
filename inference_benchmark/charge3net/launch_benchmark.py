"""
Launcher for the ChargE3Net inference-throughput benchmark.

Defaults to NNODES=1, NPROCS=1 — i.e. single-GPU production-like throughput,
which is the headline number we want. Increase NPROCS only to shard the test
split across multiple GPUs to finish faster; per-rank throughput is still
single-GPU.

Usage:
    python inference_benchmark/launch_benchmark.py
    NPROCS=4 python inference_benchmark/launch_benchmark.py
"""

from __future__ import annotations

import os
import subprocess
import sys

import torch.multiprocessing as mp

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_MAX_THREADS"] = "1"

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from inference_benchmark.benchmark_inference import run

NNODES = int(os.environ.get("NNODES", 1))
NPROCS = int(os.environ.get("NPROCS", 1))


def main():
    cfg = {"nnodes": NNODES, "nprocs": NPROCS}
    env = {
        "master_port": str(29500),
        "master_addr": "localhost",
        "world_size": NNODES * NPROCS,
        "group_rank": int(os.environ.get("SLURM_NODEID", "0")),
    }

    if NNODES > 1:
        cmd = "scontrol show hostnames " + os.getenv("SLURM_JOB_NODELIST")
        env["master_addr"] = (
            subprocess.check_output(cmd.split()).decode().splitlines()[0]
        )
        print("master_addr", env["master_addr"])

    os.environ["MASTER_ADDR"] = env["master_addr"]
    os.environ["MASTER_PORT"] = env["master_port"]
    os.environ["WORLD_SIZE"] = str(env["world_size"])

    print(
        f"Launching {NPROCS} process(es) on node {env['group_rank']} "
        f"(world_size={env['world_size']})"
    )

    if NPROCS == 1:
        # mp.spawn(nprocs=1) still forks; for single-GPU benchmark just call run directly.
        run(0, cfg, env)
    else:
        mp.spawn(run, args=(cfg, env), nprocs=NPROCS)


if __name__ == "__main__":
    main()
