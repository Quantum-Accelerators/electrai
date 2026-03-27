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
    modal run modal/benchmark.py --dataset dataset_4 --gpu A100 --channels 32 --residual-blocks 16

    # All S3 samples
    modal run modal/benchmark.py --samples 0

    # Quick smoke test
    modal run modal/benchmark.py --samples 10 --epochs 2 --channels 8 --residual-blocks 2
"""

from __future__ import annotations

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


@app.function(
    gpu="L4",
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

    os.environ["INSTANCE_TYPE"] = f"modal-{gpu_type}"
    # Set workflow-like name so WandB run name is descriptive.
    # GHA sets GITHUB_RUN_NUMBER; for local runs, use timestamp.
    os.environ["GITHUB_WORKFLOW"] = "Modal Benchmark"
    if "GITHUB_RUN_NUMBER" not in os.environ:
        os.environ["GITHUB_RUN_NUMBER"] = time.strftime("%y%m%d-%H%M")

    results = run_training(
        channels=channels,
        residual_blocks=residual_blocks,
        epochs=epochs,
        seed=seed,
        gpu=True,
        gradient_checkpoint=use_grad_ckpt,
        data_root=data_root,
        max_file_size=0,  # already filtered above
        wandb_project=wandb_project,
        verbose=True,
    )

    log.info("val_loss: %.6f", results["final_val_loss"])
    log.info("train_loss: %.6f", results["final_train_loss"])
    log.info("Wallclock: %.1fs", results["wallclock_s"])
    log.info("GPU: %s (Modal)", gpu_type)

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

    benchmark_fn = run_benchmark
    if gpu != "L4":
        benchmark_fn = run_benchmark.with_options(gpu=gpu)

    results = benchmark_fn.remote(
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
