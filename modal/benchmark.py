"""Modal GPU benchmark for electrai.

Runs a configurable training benchmark on Modal GPUs with data from the
electrai-data Volume, logs metrics to WandB, and reports wall-clock time.

Two data sources on the Volume:
- "s3" (default): 205 samples from s3://openathena/electrai/ (≤25MB, matches EC2 benchmark)
- "dataset_4": 2,885 samples from Globus/Della dataset_4 (large grids, needs A100 for prod config)

Usage:
    # Default: 50 samples from S3 set, 5 epochs, 32ch/16blk, L4 (matches EC2 benchmark)
    modal run modal/benchmark.py

    # Dataset_4 with tiny model on L4
    modal run modal/benchmark.py --dataset dataset_4 --channels 8 --residual-blocks 2

    # Production model on A100 with dataset_4
    MODAL_GPU=A100 modal run modal/benchmark.py --dataset dataset_4 --channels 32 --residual-blocks 16

    # All S3 samples
    modal run modal/benchmark.py --samples 0

    # Quick smoke test
    modal run modal/benchmark.py --samples 10 --epochs 2 --channels 8 --residual-blocks 2
"""

from __future__ import annotations

import os
from pathlib import Path

import modal

ROOT = Path(__file__).parent.parent

data_volume = modal.Volume.from_name("electrai-data")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install_from_pyproject(
        str(ROOT / "pyproject.toml"), optional_dependencies=["dev"]
    )
    .add_local_dir(str(ROOT / "src"), remote_path="/root/electrai/src", copy=True)
    .add_local_dir(
        str(ROOT / "scripts"), remote_path="/root/electrai/scripts", copy=True
    )
    .add_local_file(
        str(ROOT / "pyproject.toml"),
        remote_path="/root/electrai/pyproject.toml",
        copy=True,
    )
    .run_commands("cd /root/electrai && pip install --no-deps -e .")
)

app = modal.App("electrai-benchmark", image=image)

# Data roots on the Volume
DATASETS = {
    # Mirrors s3://openathena/electrai/ — same data as EC2 gpu-benchmark.yml
    # Note: S3 uses input/ but RhoRead expects data/, so we symlink
    "s3": {
        "root": "/data/s3/openathena/electrai",
        "input_dir": "input",  # S3 naming
        "max_file_size": 25,  # matches EC2 benchmark default
    },
    # Globus/Della dataset_4 — 2,885 samples, large grids
    "dataset_4": {
        "root": "/data/mp/chg_datasets/dataset_4",
        "input_dir": "data",  # Della naming
        "max_file_size": 100,
    },
}


# e.g. "L4", "A100", "A100:2" for 2x A100
GPU_SPEC = os.environ.get("MODAL_GPU", "L4")


@app.function(
    gpu=GPU_SPEC,
    volumes={"/data": data_volume},
    secrets=[modal.Secret.from_name("wandb-credentials")],
    timeout=7200,
    retries=0,
)
def run_benchmark(
    epochs: int = 5,
    channels: int = 32,
    residual_blocks: int = 16,
    samples: int = 50,
    max_file_size: float = -1,
    seed: int = 42,
    wandb_project: str = "elf-net-ci",
    gpu_type: str = "L4",
    dataset: str = "s3",
    local_copy: bool = False,
):
    """Run benchmark and return results."""
    import logging
    import sys

    log = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)

    sys.path.insert(0, "/root/electrai/scripts")
    from e2e_train import run_training

    ds = DATASETS[dataset]
    ds_root = Path(ds["root"])
    input_dir = ds["input_dir"]

    # Use dataset-specific default if max_file_size not explicitly set
    if max_file_size < 0:
        max_file_size = ds["max_file_size"]

    # Build filelist from files on disk (S3 set has no mp_filelist.txt)
    data_dir = ds_root / input_dir
    all_ids = sorted(p.stem for p in data_dir.glob("*.CHGCAR"))
    log.info("Dataset %r: %d total samples in %s", dataset, len(all_ids), data_dir)

    # Filter by file size (avoid OOM on large grids)
    if max_file_size > 0:
        max_bytes = int(max_file_size * 1024 * 1024)
        eligible = [
            sid
            for sid in all_ids
            if (data_dir / f"{sid}.CHGCAR").stat().st_size <= max_bytes
        ]
        log.info(
            "File size filter: %d/%d eligible (<=%.0fMB)",
            len(eligible),
            len(all_ids),
            max_file_size,
        )
    else:
        eligible = all_ids

    # Select samples: first N (lexicographic, matching s3_sync.py behavior)
    if 0 < samples < len(eligible):
        subset = eligible[:samples]
        log.info("Selected first %d/%d eligible samples", samples, len(eligible))
    else:
        subset = eligible
        samples = len(subset)
        log.info("Using all %d eligible samples", samples)

    if not subset:
        raise ValueError(
            f"No eligible samples (dataset={dataset}, total={len(all_ids)}, "
            f"max_file_size={max_file_size}MB)"
        )

    data_root = "/tmp/benchmark_data"
    Path(data_root).mkdir(parents=True, exist_ok=True)

    if local_copy:
        # Copy selected samples to local disk (Volume I/O is ~15x slower)
        import shutil

        local_data = Path(data_root) / "data"
        local_label = Path(data_root) / "label"
        local_data.mkdir(parents=True, exist_ok=True)
        local_label.mkdir(parents=True, exist_ok=True)
        for sid in subset:
            shutil.copy2(
                ds_root / input_dir / f"{sid}.CHGCAR", local_data / f"{sid}.CHGCAR"
            )
            shutil.copy2(
                ds_root / "label" / f"{sid}.CHGCAR", local_label / f"{sid}.CHGCAR"
            )
        log.info("Copied %d samples to local disk", len(subset))
    else:
        # Symlink to volume (slower I/O but no copy overhead for large datasets)
        data_link = Path(data_root) / "data"
        if not data_link.exists():
            data_link.symlink_to(ds_root / input_dir)
        label_link = Path(data_root) / "label"
        if not label_link.exists():
            label_link.symlink_to(ds_root / "label")
        log.info("Using volume directly (no local copy)")

    Path(data_root, "mp_filelist.txt").write_text("\n".join(subset) + "\n")

    # Always use gradient checkpointing (32ch/16blk needs it even for ≤25MB files)
    use_grad_ckpt = True

    log.info(
        "Benchmark: gpu=%s, epochs=%d, channels=%d, blocks=%d, samples=%d, "
        "dataset=%s, grad_ckpt=%s",
        gpu_type,
        epochs,
        channels,
        residual_blocks,
        samples,
        dataset,
        use_grad_ckpt,
    )

    # Set WandB run name and env vars for platform tagging
    import os
    import time

    # Detect GPU count from MODAL_GPU spec (e.g. "A100:2" → 2)
    import torch

    num_gpus = torch.cuda.device_count()
    os.environ["INSTANCE_TYPE"] = f"modal-{gpu_type}x{num_gpus}"
    log.info("GPUs detected: %d (%s)", num_gpus, gpu_type)
    # Set workflow-like name so WandB run name is descriptive.
    # GHA sets GITHUB_RUN_NUMBER; for local runs, use timestamp.
    os.environ["GITHUB_WORKFLOW"] = "Modal Benchmark"
    if "GITHUB_RUN_NUMBER" not in os.environ:
        os.environ["GITHUB_RUN_NUMBER"] = time.strftime("%y%m%d-%H%M")

    if num_gpus > 1:
        # Multi-GPU: use torchrun with the production entrypoint (real DDP, no GIL)
        import subprocess
        import sys
        from time import monotonic

        import yaml

        cfg = {
            "data": {
                "_target_": "electrai.dataloader.dataset.RhoRead",
                "root": f"{data_root}/mp_filelist.txt",
                "split_file": None,
                "precision": "f32",
                "batch_size": 1,
                "train_workers": 4 * num_gpus,
                "val_workers": 2,
                "pin_memory": False,
                "val_frac": 0.4,
                "drop_last": False,
                "augmentation": False,
                "random_seed": seed,
            },
            "model": {
                "_target_": "electrai.model.resunet.ResUNet3D",
                "in_channels": 1,
                "out_channels": 1,
                "n_channels": channels,
                "n_residual_blocks": residual_blocks,
                "kernel_size": 3,
                "depth": 2,
                "use_checkpoint": use_grad_ckpt,
            },
            "precision": 32,
            "epochs": epochs,
            "lr": 0.001,
            "weight_decay": 0.0,
            "warmup_length": 1,
            "gradient_clip_value": 1.0,
            "wandb_mode": "online" if wandb_project else "disabled",
            "entity": "PrinceOA",
            "wb_pname": wandb_project or "elf-net-ci-test",
            "ckpt_path": "/tmp/checkpoints",
        }
        config_path = Path("/tmp/benchmark_config.yaml")
        with config_path.open("w") as f:
            yaml.dump(cfg, f, default_flow_style=False)

        # Pass naming env vars so the production entrypoint's WandB logger uses them
        run_env = {
            **os.environ,
            "INSTANCE_TYPE": f"modal-{gpu_type}x{num_gpus}",
            "GITHUB_WORKFLOW": os.environ.get("GITHUB_WORKFLOW", "Modal Benchmark"),
            "GITHUB_RUN_NUMBER": os.environ.get(
                "GITHUB_RUN_NUMBER", time.strftime("%y%m%d-%H%M")
            ),
        }

        log.info("Launching torchrun with %d GPUs", num_gpus)
        t0 = monotonic()
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "torch.distributed.run",
                f"--nproc_per_node={num_gpus}",
                "-m",
                "electrai.entrypoints.main",
                "train",
                "--config",
                str(config_path),
            ],
            cwd="/root/electrai",
            capture_output=True,
            text=True,
            env=run_env,
            check=False,
        )
        wallclock = monotonic() - t0

        # Log torchrun output
        if proc.stdout:
            for line in proc.stdout.splitlines()[-20:]:
                log.info("[torchrun] %s", line)
        if proc.stderr:
            for line in proc.stderr.splitlines()[-20:]:
                log.info("[torchrun:err] %s", line)

        if proc.returncode != 0:
            raise RuntimeError(f"torchrun failed with exit code {proc.returncode}")

        # Parse val_loss from torchrun output (Lightning logs it)
        import re

        final_val = 0.0
        final_train = 0.0
        wandb_url = None
        for line in (proc.stdout + proc.stderr).splitlines():
            m = re.search(r"val_loss_epoch=(\d+\.\d+)", line)
            if m:
                final_val = float(m.group(1))
            m = re.search(r"train_loss_epoch=(\d+\.\d+)", line)
            if m:
                final_train = float(m.group(1))
            m = re.search(r"(https://wandb\.ai/\S+)", line)
            if m:
                wandb_url = m.group(1)

        results = {
            "final_val_loss": final_val,
            "final_train_loss": final_train,
            "epoch_times": [],
            "wallclock_s": wallclock,
            "wandb_run_url": wandb_url,
        }
    else:
        # Single GPU: use run_training directly (better metrics capture)
        results = run_training(
            channels=channels,
            residual_blocks=residual_blocks,
            epochs=epochs,
            seed=seed,
            gpu=True,
            gradient_checkpoint=use_grad_ckpt,
            data_root=data_root,
            max_file_size=0,  # already filtered above
            devices=1,
            train_workers=4,
            wandb_project=wandb_project,
            verbose=True,
        )

    log.info("val_loss: %s", results.get("final_val_loss", "?"))
    log.info("train_loss: %s", results.get("final_train_loss", "?"))
    log.info("Wallclock: %.1fs", results["wallclock_s"])
    log.info("GPU: %s x%d (Modal)", gpu_type, num_gpus)
    epoch_times = results.get("epoch_times", [])
    if epoch_times:
        for i, t in enumerate(epoch_times):
            log.info("  Epoch %d: %.1fs", i, t)
        log.info("  Mean epoch: %.1fs", sum(epoch_times) / len(epoch_times))
        log.info(
            "  Overhead (wallclock - sum epochs): %.1fs",
            results["wallclock_s"] - sum(epoch_times),
        )

    return results


@app.local_entrypoint()
def main(
    gpu: str = "L4",
    epochs: int = 5,
    channels: int = 32,
    residual_blocks: int = 16,
    samples: int = 50,
    max_file_size: float = -1,
    seed: int = 42,
    wandb_project: str = "elf-net-ci",
    dataset: str = "s3",
    local_copy: bool = False,
):
    import logging

    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger(__name__)

    if gpu != GPU_SPEC:
        log.warning(
            "GPU override via --gpu %s, but function is pinned to %s. "
            "Set MODAL_GPU=%s env var before running.",
            gpu,
            GPU_SPEC,
            gpu,
        )

    results = run_benchmark.remote(
        epochs=epochs,
        channels=channels,
        residual_blocks=residual_blocks,
        samples=samples,
        max_file_size=max_file_size,
        seed=seed,
        wandb_project=wandb_project,
        gpu_type=gpu,
        dataset=dataset,
        local_copy=local_copy,
    )

    log.info(
        "Benchmark complete: val_loss=%.6f, wallclock=%.1fs on %s",
        results["final_val_loss"],
        results["wallclock_s"],
        gpu,
    )

    # Print parseable output for GHA summary
    wandb_url = results.get("wandb_run_url") or ""
    print(f"BENCHMARK_VAL_LOSS={results['final_val_loss']:.6f}")  # noqa: T201
    print(f"BENCHMARK_TRAIN_LOSS={results['final_train_loss']:.6f}")  # noqa: T201
    print(f"BENCHMARK_WALLCLOCK={results['wallclock_s']:.0f}")  # noqa: T201
    print(f"BENCHMARK_GPU={gpu}")  # noqa: T201
    print(f"BENCHMARK_DATASET={dataset}")  # noqa: T201
    print(f"BENCHMARK_SAMPLES={samples}")  # noqa: T201
    if wandb_url:
        print(f"BENCHMARK_WANDB_URL={wandb_url}")  # noqa: T201
