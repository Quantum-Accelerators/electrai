#!/usr/bin/env python
"""
End-to-end training test with deterministic seeding.

Runs a minimal training loop on the sample data in data/MP/ and verifies
that the final validation loss matches an expected value (within tolerance).

Usage:
    # Run with defaults (5 epochs, checks val_loss)
    ./tests/e2e_train.py

    # Run more epochs, update expected loss
    ./tests/e2e_train.py --epochs 10 --update-expected

    # Just train, don't check (for exploration)
    ./tests/e2e_train.py --no-check
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import click

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


@click.command()
@click.option('-B', '--residual-blocks', default=2, help="Number of residual blocks (default: 2, production: 16)")
@click.option('-C', '--channels', default=8, help="Number of model channels (default: 8, production: 32-64)")
@click.option('-c', '--check/--no-check', default=True, help="Check val_loss against expected value")
@click.option('-e', '--epochs', default=5, help="Number of training epochs")
@click.option('-g', '--gpu', is_flag=True, help="Use GPU acceleration (if available)")
@click.option('-s', '--seed', default=42, help="Random seed for reproducibility")
@click.option('-t', '--tolerance', default=0.001, help="Tolerance for val_loss comparison (absolute)")
@click.option('-U', '--update-expected', is_flag=True, help="Update expected_loss.txt with final loss")
@click.option('-v', '--verbose', is_flag=True, help="Verbose output")
def main(
    residual_blocks: int,
    channels: int,
    check: bool,
    epochs: int,
    gpu: bool,
    seed: int,
    tolerance: float,
    update_expected: bool,
    verbose: bool,
):
    """Run deterministic e2e training test."""
    import torch
    from lightning.pytorch import Trainer, seed_everything
    from src.electrai.dataloader.registry import get_data
    from src.electrai.entrypoints.train import collate_fn
    from src.electrai.lightning import LightningGenerator
    from torch.utils.data import DataLoader

    # Paths
    repo_root = Path(__file__).parent.parent
    expected_loss_file = Path(__file__).parent / "expected_loss.txt"

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
    cfg.data_path = str(repo_root / "data/MP/chgcars/input")
    cfg.label_path = str(repo_root / "data/MP/chgcars/label")
    cfg.map_path = str(repo_root / "data/MP/map/map_sample.json.gz")
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
        import torch

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
    )

    # Train
    trainer.fit(model, train_loader, test_loader)

    # Get final validation loss
    final_val_loss = trainer.callback_metrics.get("val_loss_epoch")
    if final_val_loss is None:
        final_val_loss = trainer.callback_metrics.get("val_loss")
    final_val_loss = float(final_val_loss)

    click.echo(f"Final val_loss: {final_val_loss:.6f}")

    # Update expected loss file if requested
    if update_expected:
        expected_loss_file.write_text(f"{final_val_loss:.6f}\n")
        click.echo(f"Updated {expected_loss_file}")
        return

    # Check against expected
    if check:
        if not expected_loss_file.exists():
            click.echo(f"No expected loss file at {expected_loss_file}", err=True)
            click.echo("Run with --update-expected to create it", err=True)
            sys.exit(1)

        expected_loss = float(expected_loss_file.read_text().strip())
        diff = abs(final_val_loss - expected_loss)

        if diff > tolerance:
            click.echo(
                f"FAIL: val_loss {final_val_loss:.6f} differs from expected "
                f"{expected_loss:.6f} by {diff:.6f} (tolerance: {tolerance})",
                err=True,
            )
            sys.exit(1)
        else:
            click.echo(
                f"PASS: val_loss matches expected within tolerance ({diff:.6f} <= {tolerance})"
            )


if __name__ == "__main__":
    main()
