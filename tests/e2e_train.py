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
    ./tests/e2e_train.py

    # Run more epochs, update expected values for current platform
    ./tests/e2e_train.py --epochs 10 --update-expected

    # Just train, don't check (for exploration)
    ./tests/e2e_train.py --no-check
"""

from __future__ import annotations

import json
import os
import platform as platform_mod
import sys
from pathlib import Path

import click

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


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


class LossTracker:
    """Callback to track per-epoch losses."""

    def __init__(self):
        self.epoch_train_losses: list[float] = []
        self.epoch_val_losses: list[float] = []


@click.command()
@click.option('-B', '--residual-blocks', default=2, help="Number of residual blocks (default: 2, production: 16)")
@click.option('-C', '--channels', default=8, help="Number of model channels (default: 8, production: 32-64)")
@click.option('-c', '--check/--no-check', default=True, help="Check val_loss against expected value")
@click.option('-d', '--data-path', default=None, help="Path to input data (default: data/MP/chgcars/input)")
@click.option('-e', '--epochs', default=5, help="Number of training epochs")
@click.option('-g', '--gpu', is_flag=True, help="Use GPU acceleration (if available)")
@click.option('-l', '--label-path', default=None, help="Path to label data (default: data/MP/chgcars/label)")
@click.option('-m', '--map-path', default=None, help="Path to map file (default: data/MP/map/map_sample.json.gz)")
@click.option('-s', '--seed', default=42, help="Random seed for reproducibility")
@click.option('-t', '--tolerance', default=0.001, help="Tolerance for val_loss comparison (absolute)")
@click.option('-U', '--update-expected', is_flag=True, help="Update expected_values.json for current platform")
@click.option('-v', '--verbose', is_flag=True, help="Verbose output")
def main(
    residual_blocks: int,
    channels: int,
    check: bool,
    data_path: str | None,
    epochs: int,
    gpu: bool,
    label_path: str | None,
    map_path: str | None,
    seed: int,
    tolerance: float,
    update_expected: bool,
    verbose: bool,
):
    """Run deterministic e2e training test."""
    import torch
    from lightning.pytorch import Callback, Trainer, seed_everything
    from src.electrai.dataloader.registry import get_data
    from src.electrai.entrypoints.train import collate_fn
    from src.electrai.lightning import LightningGenerator
    from torch.utils.data import DataLoader

    # Paths
    repo_root = Path(__file__).parent.parent
    expected_values_file = Path(__file__).parent / "expected_values.json"

    # Platform detection (includes GPU suffix for linux-gpu)
    platform = get_platform(gpu=gpu)

    # Force deterministic behavior
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
    cfg.use_checkpoint = False

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
            click.echo(
                "Warning: --gpu requested but no GPU available, falling back to CPU",
                err=True,
            )
            accelerator = "cpu"
    else:
        accelerator = "cpu"

    if verbose:
        click.echo(f"Platform: {platform}")
        click.echo(f"Config: epochs={cfg.epochs}, seed={seed}, channels={cfg.n_channels}, blocks={cfg.n_residual_blocks}")
        click.echo(f"Accelerator: {accelerator}")
        click.echo(f"Data: {cfg.data_path}")

    # Load data
    train_data, test_data = get_data(cfg)
    if verbose:
        click.echo(f"Train samples: {len(train_data)}, Val samples: {len(test_data)}")

    train_loader = DataLoader(
        train_data,
        batch_size=cfg.nbatch,
        shuffle=True,
        num_workers=cfg.num_workers,
        generator=torch.Generator().manual_seed(seed),
    )
    test_loader = DataLoader(
        test_data,
        batch_size=cfg.nbatch,
        shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=collate_fn,
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

    # Trainer (minimal, no logging)
    trainer = Trainer(
        max_epochs=cfg.epochs,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=verbose,
        accelerator=accelerator,
        devices=1,
        precision=cfg.model_precision,
        deterministic=True,
        gradient_clip_val=cfg.gradient_clip_value,
        callbacks=[loss_callback],
    )

    # Train
    trainer.fit(model, train_loader, test_loader)

    # Get final losses
    final_val_loss = trainer.callback_metrics.get("val_loss_epoch")
    if final_val_loss is None:
        final_val_loss = trainer.callback_metrics.get("val_loss")
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
        click.echo(f"\nResults for {platform}:")
        click.echo(f"  Final val_loss: {final_val_loss:.6f}")
        if final_train_loss:
            click.echo(f"  Final train_loss: {final_train_loss:.6f}")
        click.echo(f"  Epoch val_losses: {[f'{v:.6f}' for v in loss_callback.epoch_val_losses]}")
        click.echo(f"  Epoch train_losses: {[f'{v:.6f}' for v in loss_callback.epoch_train_losses]}")

    click.echo(f"Final val_loss: {final_val_loss:.6f}")

    # Update expected values file if requested
    if update_expected:
        if expected_values_file.exists():
            expected_values = json.loads(expected_values_file.read_text())
        else:
            expected_values = {}

        expected_values[platform] = results
        expected_values_file.write_text(json.dumps(expected_values, indent=2) + "\n")
        click.echo(f"Updated {expected_values_file} for platform '{platform}'")
        return

    # Check against expected
    if check:
        if not expected_values_file.exists():
            click.echo(f"No expected values file at {expected_values_file}", err=True)
            click.echo("Run with --update-expected to create it", err=True)
            sys.exit(1)

        expected_values = json.loads(expected_values_file.read_text())

        if platform not in expected_values:
            click.echo(f"No expected values for platform '{platform}'", err=True)
            click.echo(f"Available platforms: {list(expected_values.keys())}", err=True)
            click.echo("Run with --update-expected to add values for this platform", err=True)
            sys.exit(1)

        expected = expected_values[platform]

        if expected.get("final_val_loss") is None:
            click.echo(f"Expected values for '{platform}' are null (not yet generated)", err=True)
            click.echo("Run with --update-expected to generate values for this platform", err=True)
            sys.exit(1)

        expected_val_loss = expected["final_val_loss"]
        diff = abs(final_val_loss - expected_val_loss)

        if diff > tolerance:
            click.echo(
                f"FAIL: val_loss {final_val_loss:.6f} differs from expected "
                f"{expected_val_loss:.6f} by {diff:.6f} (tolerance: {tolerance})",
                err=True,
            )

            # Show per-epoch comparison if available
            if expected.get("epoch_val_losses") and loss_callback.epoch_val_losses:
                click.echo("\nPer-epoch val_loss comparison:", err=True)
                for i, (actual, exp) in enumerate(zip(
                    loss_callback.epoch_val_losses,
                    expected["epoch_val_losses"]
                )):
                    epoch_diff = abs(actual - exp)
                    marker = " <-- DIVERGED" if epoch_diff > tolerance else ""
                    click.echo(f"  Epoch {i}: actual={actual:.6f}, expected={exp:.6f}, diff={epoch_diff:.6f}{marker}", err=True)

            sys.exit(1)
        else:
            click.echo(
                f"PASS: val_loss matches expected within tolerance ({diff:.6f} <= {tolerance})"
            )


if __name__ == "__main__":
    main()
