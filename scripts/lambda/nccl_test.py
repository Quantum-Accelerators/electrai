"""Minimal cross-node NCCL allreduce smoke test.

Run with torchrun on BOTH nodes (same args as run_training_multinode.sh):

    # head:
    NODE_RANK=0 MASTER_ADDR=<head-ip> NCCL_DEBUG=INFO \\
      NCCL_SOCKET_IFNAME=<iface> NCCL_IB_DISABLE=1 \\
      uv run torchrun --nnodes=2 --node_rank=0 --nproc_per_node=8 \\
        --master_addr=<head-ip> --master_port=29500 \\
        scripts/lambda/nccl_test.py

    # worker:
    NODE_RANK=1 MASTER_ADDR=<head-ip> NCCL_DEBUG=INFO \\
      NCCL_SOCKET_IFNAME=<iface> NCCL_IB_DISABLE=1 \\
      uv run torchrun --nnodes=2 --node_rank=1 --nproc_per_node=8 \\
        --master_addr=<head-ip> --master_port=29500 \\
        scripts/lambda/nccl_test.py

Verifies process group init + an all_reduce across world_size ranks. If this
hangs at init, fix the NCCL_SOCKET_IFNAME / firewall before touching the
real training command.
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist


def main() -> None:
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")

    tensor = torch.full((1,), float(rank), device=f"cuda:{local_rank}")
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

    # Expected sum: 0+1+...+(world_size-1) == world_size*(world_size-1)/2
    expected = world_size * (world_size - 1) / 2
    got = tensor.item()
    ok = abs(got - expected) < 1e-6
    print(  # noqa: T201
        f"[rank {rank}/{world_size} local={local_rank}] "
        f"allreduce got={got} expected={expected} ok={ok}",
        flush=True,
    )

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
