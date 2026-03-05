#!/usr/bin/env python
"""
End-to-end training test with deterministic seeding.

Runs a minimal training loop on the sample data in data/MP/ and verifies
that the final validation loss matches an expected value (within tolerance).

Expected values are platform-specific (darwin-arm64 vs darwin-x86_64 vs linux)
since floating-point operations can produce slightly different results across
platforms and CPU architectures.

Usage:
    # Run with defaults (5 epochs, checks val_loss)
    uv run python tests/e2e_train.py

    # Run more epochs, update expected values for current platform
    uv run python tests/e2e_train.py --epochs 10 --update-expected

    # Just train, don't check (for exploration)
    uv run python tests/e2e_train.py --no-check
"""

from __future__ import annotations

import hashlib
import json
import os
import platform as platform_mod
import sys
from pathlib import Path

from click import command, echo, option


def get_platform(gpu: bool = False) -> str:
    """Get platform key for expected values.

    Returns platform-architecture combinations:
    - darwin-arm64 (Apple Silicon Macs, CPU)
    - darwin-x86_64 (Intel Macs, CPU)
    - linux (Linux x86_64, CPU)
    - linux-gpu (Linux x86_64, CUDA GPU)

    Args:
        gpu: Whether GPU acceleration is being used
    """
    machine = platform_mod.machine()
    if sys.platform == "darwin":
        # Distinguish Apple Silicon from Intel Macs
        if machine == "arm64":
            return "darwin-arm64"
        else:
            return "darwin-x86_64"
    elif sys.platform.startswith("linux"):
        return "linux-gpu" if gpu else "linux"
    else:
        return f"{sys.platform}-{machine}"


# fmt: off
@command()
@option("-B", "--residual-blocks", default=2, help="Number of residual blocks (default: 2, production: 16)")
@option("-C", "--channels", default=8, help="Number of model channels (default: 8, production: 32-64)")
@option("-c", "--check/--no-check", default=True, help="Check val_loss against expected value")
@option("-d", "--data-path", default=None, help="Path to input data (default: data/MP/chgcars/input)")
@option("-e", "--epochs", default=5, help="Number of training epochs")
@option("-G", "--gradient-checkpoint", is_flag=True, help="Enable gradient checkpointing (saves VRAM, slower)")
@option("-g", "--gpu", is_flag=True, help="Use GPU acceleration (if available)")
@option("-l", "--label-path", default=None, help="Path to label data (default: data/MP/chgcars/label)")
@option("-M", "--max-file-size", default=0, type=float, help="Skip input files larger than N MB (0=no limit)")
@option("-m", "--map-path", default=None, help="Path to map file (default: data/MP/map/map_sample.json.gz)")
@option("-s", "--seed", default=42, help="Random seed for reproducibility")
@option("-t", "--tolerance", default=0.001, help="Tolerance for val_loss comparison (absolute)")
@option("-U", "--update-expected", is_flag=True, help="Update expected_values.json for current platform")
@option("-v", "--verbose", is_flag=True, help="Verbose output")
@option("-W", "--wandb-project", default=None, help="Enable WandB logging with this project name")
# fmt: on
def main(
    residual_blocks: int,
    channels: int,
    check: bool,
    data_path: str | None,
    epochs: int,
    gradient_checkpoint: bool,
    gpu: bool,
    label_path: str | None,
    max_file_size: float,
    map_path: str | None,
    seed: int,
    tolerance: float,
    update_expected: bool,
    verbose: bool,
    wandb_project: str | None,
):
    """Run deterministic e2e training test."""
    from time import monotonic

    import torch
    from lightning.pytorch import Callback, Trainer, seed_everything
    from torch.utils.data import DataLoader

    from electrai.dataloader.collate import collate_fn
    from electrai.dataloader.registry import get_data
    from electrai.lightning import LightningGenerator

    # Paths
    repo_root = Path(__file__).parent.parent
    expected_values_file = Path(__file__).parent / "expected_values.json"

    # Force deterministic behavior (skip if WandB logging, which implies benchmark mode)
    if not wandb_project:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    seed_everything(seed, workers=True)

    # Minimal config for testing
    class Config:
        pass

    cfg = Config()

    # Dataset
    cfg.dataset_name = "mp"
    cfg.data_path = data_path or str(repo_root / "data/MP/chgcars/input")
    cfg.label_path = label_path or str(repo_root / "data/MP/chgcars/label")
    cfg.map_path = map_path or str(repo_root / "data/MP/map/map_sample.json.gz")
    cfg.rho_type = "chgcar"
    cfg.functional = "GGA"
    cfg.train_fraction = 0.6  # 3 train, 2 val from 5 samples
    cfg.num_workers = 0  # Single-threaded for determinism
    cfg.downsample_label = 1
    cfg.downsample_data = 0
    cfg.data_augmentation = False
    cfg.random_seed = seed
    cfg.normalize = True
    cfg.data_precision = "f32"

    # Model (configurable: small for fast CI, larger for benchmarks)
    cfg.n_channels = channels
    cfg.n_residual_blocks = residual_blocks
    cfg.n_upscale_layers = 0  # No upscaling, same resolution
    cfg.kernel_size1 = 3
    cfg.kernel_size2 = 3
    cfg.use_checkpoint = gradient_checkpoint

    # Hydra-style model config for LightningGenerator (uses hydra.utils.instantiate)
    cfg.model = {
        "_target_": "electrai.model.srgan_layernorm_pbc.GeneratorResNet",
        "n_residual_blocks": residual_blocks,
        "n_upscale_layers": 0,
        "n_channels": channels,
        "kernel_size1": 3,
        "kernel_size2": 3,
        "normalize": True,
        "use_checkpoint": gradient_checkpoint,
    }

    # Training
    cfg.epochs = epochs
    cfg.nbatch = 1
    cfg.lr = 0.001
    cfg.weight_decay = 0.0
    cfg.warmup_length = 1
    cfg.model_precision = 32
    cfg.gradient_clip_value = 1.0

    # Determine accelerator
    if gpu:
        if torch.cuda.is_available():
            accelerator = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            accelerator = "mps"
        else:
            echo(
                "Warning: --gpu requested but no GPU available, falling back to CPU",
                err=True,
            )
            accelerator = "cpu"
    else:
        accelerator = "cpu"

    # Platform detection based on resolved accelerator (not just --gpu flag)
    platform = get_platform(gpu=(accelerator == "cuda"))

    if verbose:
        echo(f"Platform: {platform}")
        echo(f"Config: epochs={cfg.epochs}, seed={seed}, channels={cfg.n_channels}, blocks={cfg.n_residual_blocks}")
        echo(f"Accelerator: {accelerator}")
        echo(f"Data: {cfg.data_path}")

    # Load data
    train_data, test_data = get_data(cfg)

    # Filter by file size if requested (avoid OOM on large grids)
    if max_file_size > 0:
        max_bytes = int(max_file_size * 1024 * 1024)
        for name, dataset in [("train", train_data), ("val", test_data)]:
            original = len(dataset.data)
            kept = []
            for inp, lbl in dataset.data:
                size = inp.stat().st_size
                if size > max_bytes:
                    echo(f"  skip {inp.name} ({size / 1048576:.1f}MB > {max_file_size}MB)", err=True)
                else:
                    kept.append((inp, lbl))
            dataset.data = kept
            filtered = original - len(kept)
            if filtered > 0:
                echo(f"Filtered {filtered}/{original} {name} samples > {max_file_size}MB", err=True)
        if len(train_data) == 0 or len(test_data) == 0:
            echo("Error: no samples remain after filtering", err=True)
            sys.exit(1)

    if verbose:
        echo(f"Train samples: {len(train_data)}, Val samples: {len(test_data)}")

    def dict_collate(batch):
        """Collate tuples from MPDataset into dicts for LightningGenerator."""
        collated = collate_fn(batch)
        if isinstance(collated, (list, tuple)):
            return {"data": collated[0], "label": collated[1]}
        return collated

    train_loader = DataLoader(
        train_data,
        batch_size=cfg.nbatch,
        shuffle=True,
        num_workers=cfg.num_workers,
        generator=torch.Generator().manual_seed(seed),
        collate_fn=dict_collate,
    )
    test_loader = DataLoader(
        test_data,
        batch_size=cfg.nbatch,
        shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=dict_collate,
    )

    # Model
    model = LightningGenerator(cfg)

    # Loss tracking callback
    class EpochLossCallback(Callback):
        def __init__(self):
            self.epoch_train_losses: list[float] = []
            self.epoch_val_losses: list[float] = []

        def on_train_epoch_end(self, trainer, pl_module):
            train_loss = trainer.callback_metrics.get("train_loss_epoch")
            if train_loss is not None:
                self.epoch_train_losses.append(float(train_loss))

        def on_validation_epoch_end(self, trainer, pl_module):
            val_loss = trainer.callback_metrics.get("val_loss_epoch")
            if val_loss is None:
                val_loss = trainer.callback_metrics.get("val_loss")
            if val_loss is not None:
                self.epoch_val_losses.append(float(val_loss))

    loss_callback = EpochLossCallback()

    # WandB logger (optional)
    logger = False
    if wandb_project:
        from lightning.pytorch.loggers import WandbLogger

        # Collect sample IDs from train+val datasets
        sample_ids = sorted({
            Path(inp).stem
            for dataset in [train_data, test_data]
            for inp, _lbl in dataset.data
        })

        # Auto-compute dataset version from sample IDs if not provided
        dataset_version = os.environ.get("DATASET_VERSION", "")
        if not dataset_version:
            dataset_version = hashlib.md5("|".join(sample_ids).encode()).hexdigest()[:8]

        wandb_tags = [platform, f"ch{channels}", f"blk{residual_blocks}"]
        if gpu:
            wandb_tags.append("gpu")
        if dataset_version:
            wandb_tags.append(f"ds:{dataset_version}")
        # Pick up CI metadata from env
        git_sha = os.environ.get("GITHUB_SHA", "")[:8]
        if git_sha:
            wandb_tags.append(f"sha:{git_sha}")
        run_id = os.environ.get("GITHUB_RUN_ID", "")
        run_number = os.environ.get("GITHUB_RUN_NUMBER", "")
        workflow_name = os.environ.get("GITHUB_WORKFLOW", "")
        if workflow_name and run_number:
            wandb_run_name = f"{workflow_name}#{run_number}"
        elif git_sha:
            wandb_run_name = f"ci-{git_sha}"
        else:
            wandb_run_name = None

        repo = os.environ.get("GITHUB_REPOSITORY", "")
        logger = WandbLogger(
            project=wandb_project,
            entity=os.environ.get("WANDB_ENTITY", "PrinceOA"),
            name=wandb_run_name,
            tags=wandb_tags,
            config={
                "channels": channels,
                "residual_blocks": residual_blocks,
                "epochs": epochs,
                "gradient_checkpoint": gradient_checkpoint,
                "max_file_size_mb": max_file_size,
                "train_samples": len(train_data),
                "val_samples": len(test_data),
                "sample_ids": sample_ids,
                "dataset_version": dataset_version,
                "seed": seed,
                "instance_type": os.environ.get("INSTANCE_TYPE", ""),
                "github_run_id": run_id,
                "github_sha": os.environ.get("GITHUB_SHA", ""),
                "github_ref": os.environ.get("GITHUB_REF", ""),
            },
        )
        # Add GHA link as run notes (markdown, visible in WandB Overview tab)
        if run_id and repo:
            gha_url = f"https://github.com/{repo}/actions/runs/{run_id}"
            logger.experiment.notes = f"[GHA run {run_id}]({gha_url})"

    # Trainer
    trainer = Trainer(
        max_epochs=cfg.epochs,
        logger=logger,
        enable_checkpointing=False,
        enable_progress_bar=verbose,
        accelerator=accelerator,
        devices=1,
        precision=cfg.model_precision,
        deterministic=not wandb_project,
        gradient_clip_val=cfg.gradient_clip_value,
        callbacks=[loss_callback],
        log_every_n_steps=1,
    )

    # Train
    t0 = monotonic()
    trainer.fit(model, train_loader, test_loader)
    train_wallclock = monotonic() - t0

    # Log summary metrics + wallclock to WandB
    if wandb_project and logger.experiment:
        logger.experiment.summary["wallclock_s"] = train_wallclock
        run_url = logger.experiment.get_url()
        if run_url:
            echo(f"WANDB_RUN_URL={run_url}")

    # Get final losses
    final_val_loss = trainer.callback_metrics.get("val_loss_epoch")
    if final_val_loss is None:
        final_val_loss = trainer.callback_metrics.get("val_loss")
    if final_val_loss is None:
        echo(
            "Error: final validation loss not found in trainer.callback_metrics "
            "(expected 'val_loss_epoch' or 'val_loss')",
            err=True,
        )
        sys.exit(1)
    final_val_loss = float(final_val_loss)

    final_train_loss = trainer.callback_metrics.get("train_loss_epoch")
    if final_train_loss is not None:
        final_train_loss = float(final_train_loss)

    # Results
    results = {
        "final_val_loss": final_val_loss,
        "final_train_loss": final_train_loss,
        "epoch_val_losses": loss_callback.epoch_val_losses,
        "epoch_train_losses": loss_callback.epoch_train_losses,
    }

    if verbose:
        echo(f"\nResults for {platform}:")
        echo(f"  Final val_loss: {final_val_loss:.6f}")
        if final_train_loss:
            echo(f"  Final train_loss: {final_train_loss:.6f}")
        echo(f"  Epoch val_losses: {[f'{v:.6f}' for v in loss_callback.epoch_val_losses]}")
        echo(f"  Epoch train_losses: {[f'{v:.6f}' for v in loss_callback.epoch_train_losses]}")

    echo(f"Final val_loss: {final_val_loss:.6f}")

    # Update expected values file if requested
    if update_expected:
        if expected_values_file.exists():
            expected_values = json.loads(expected_values_file.read_text())
        else:
            expected_values = {}

        expected_values[platform] = results
        expected_values_file.write_text(json.dumps(expected_values, indent=2) + "\n")
        echo(f"Updated {expected_values_file} for platform '{platform}'")
        return

    # Check against expected
    if check:
        if not expected_values_file.exists():
            echo(f"No expected values file at {expected_values_file}", err=True)
            echo("Run with --update-expected to create it", err=True)
            sys.exit(1)

        expected_values = json.loads(expected_values_file.read_text())

        if platform not in expected_values:
            echo(f"No expected values for platform '{platform}'", err=True)
            echo(f"Available platforms: {list(expected_values.keys())}", err=True)
            echo("Run with --update-expected to add values for this platform", err=True)
            sys.exit(1)

        expected = expected_values[platform]

        if expected.get("final_val_loss") is None:
            echo(f"Expected values for '{platform}' are null (not yet generated)", err=True)
            echo("Run with --update-expected to generate values for this platform", err=True)
            sys.exit(1)

        expected_val_loss = expected["final_val_loss"]
        diff = abs(final_val_loss - expected_val_loss)

        if diff > tolerance:
            echo(
                f"FAIL: val_loss {final_val_loss:.6f} differs from expected "
                f"{expected_val_loss:.6f} by {diff:.6f} (tolerance: {tolerance})",
                err=True,
            )

            # Show per-epoch comparison if available
            if expected.get("epoch_val_losses") and loss_callback.epoch_val_losses:
                echo("\nPer-epoch val_loss comparison:", err=True)
                for i, (actual, exp) in enumerate(zip(
                    loss_callback.epoch_val_losses,
                    expected["epoch_val_losses"], strict=False,
                )):
                    epoch_diff = abs(actual - exp)
                    marker = " <-- DIVERGED" if epoch_diff > tolerance else ""
                    echo(f"  Epoch {i}: actual={actual:.6f}, expected={exp:.6f}, diff={epoch_diff:.6f}{marker}", err=True)

            sys.exit(1)
        else:
            echo(
                f"PASS: val_loss matches expected within tolerance ({diff:.6f} <= {tolerance})"
            )


if __name__ == "__main__":
    main()
