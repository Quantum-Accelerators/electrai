#!/usr/bin/env python
"""
End-to-end training CLI with deterministic seeding.

Runs a minimal training loop on the sample data in data/MP/ and verifies
that the final validation loss matches an expected value (within tolerance).

Expected values are platform-specific (darwin-arm64 vs darwin-x86_64 vs linux)
since floating-point operations can produce slightly different results across
platforms and CPU architectures.

Usage:
    # Run with defaults (5 epochs, checks val_loss)
    uv run python scripts/e2e_train.py

    # Run more epochs, update expected values for current platform
    uv run python scripts/e2e_train.py --epochs 10 --update-expected

    # Just train, don't check (for exploration)
    uv run python scripts/e2e_train.py --no-check
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
        if machine == "arm64":
            return "darwin-arm64"
        else:
            return "darwin-x86_64"
    elif sys.platform.startswith("linux"):
        return "linux-gpu" if gpu else "linux"
    else:
        return f"{sys.platform}-{machine}"


def run_training(
    channels: int = 8,
    residual_blocks: int = 2,
    epochs: int = 5,
    seed: int = 42,
    gpu: bool = False,
    data_root: str | None = None,
    gradient_checkpoint: bool = False,
    max_file_size: float = 0,
    wandb_project: str | None = None,
    verbose: bool = False,
) -> dict:
    """Run e2e training and return results.

    Returns dict with keys:
    - platform: str (e.g. "darwin-arm64")
    - final_val_loss: float
    - final_train_loss: float | None
    - epoch_val_losses: list[float]
    - epoch_train_losses: list[float]
    - wallclock_s: float
    - wandb_run_url: str | None
    """
    from time import monotonic

    import torch
    from lightning.pytorch import Callback, Trainer, seed_everything

    from electrai.dataloader.dataset import RhoRead
    from electrai.lightning import LightningGenerator

    repo_root = Path(__file__).parent.parent

    # Force deterministic behavior (skip if WandB logging, which implies benchmark mode)
    if not wandb_project:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    seed_everything(seed, workers=True)

    class Config:
        pass

    cfg = Config()

    # Dataset (RhoRead datamodule)
    data_root_path = Path(data_root) if data_root else repo_root / "data/MP/chgcars"
    filelist = data_root_path / "mp_filelist.txt"
    if not filelist.exists():
        raise FileNotFoundError(f"filelist not found: {filelist}")

    # Model
    cfg.n_channels = channels
    cfg.n_residual_blocks = residual_blocks
    cfg.use_checkpoint = gradient_checkpoint
    cfg.model = {
        "_target_": "electrai.model.resunet.ResUNet3D",
        "in_channels": 1,
        "out_channels": 1,
        "n_channels": channels,
        "n_residual_blocks": residual_blocks,
        "kernel_size": 3,
        "depth": 2,
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
            if verbose:
                echo(
                    "Warning: gpu requested but no GPU available, falling back to CPU",
                    err=True,
                )
            accelerator = "cpu"
    else:
        accelerator = "cpu"

    platform = get_platform(gpu=(accelerator == "cuda"))

    if verbose:
        echo(f"Platform: {platform}", err=True)
        echo(
            f"Config: epochs={epochs}, seed={seed}, channels={channels}, blocks={residual_blocks}",
            err=True,
        )
        echo(f"Accelerator: {accelerator}", err=True)
        echo(f"Data: {filelist}", err=True)

    # Filter filelist by file size if requested (avoid OOM on large grids)
    if max_file_size > 0:
        max_bytes = int(max_file_size * 1024 * 1024)
        all_ids = filelist.read_text().strip().splitlines()
        kept = []
        for sample_id in all_ids:
            chgcar = data_root_path / "data" / f"{sample_id}.CHGCAR"
            if chgcar.exists() and chgcar.stat().st_size > max_bytes:
                if verbose:
                    echo(
                        f"  skip {sample_id} ({chgcar.stat().st_size / 1048576:.1f}MB > {max_file_size}MB)",
                        err=True,
                    )
            else:
                kept.append(sample_id)
        if len(kept) < len(all_ids):
            echo(
                f"Filtered {len(all_ids) - len(kept)}/{len(all_ids)} samples > {max_file_size}MB",
                err=True,
            )
            filelist = data_root_path / "mp_filelist_filtered.txt"
            filelist.write_text("\n".join(kept) + "\n")
        if not kept:
            raise ValueError("no samples remain after filtering")

    datamodule = RhoRead(
        root=str(filelist),
        precision="f32",
        batch_size=cfg.nbatch,
        train_workers=0,
        val_workers=0,
        val_frac=0.4,
        augmentation=False,
        random_seed=seed,
    )

    model = LightningGenerator(cfg)

    class EpochLossCallback(Callback):
        def __init__(self):
            self.epoch_train_losses: list[float] = []
            self.epoch_val_losses: list[float] = []
            self.epoch_times: list[float] = []
            self._epoch_start: float = 0

        def on_train_epoch_start(self, _trainer, _pl_module):
            from time import monotonic

            self._epoch_start = monotonic()

        def on_train_epoch_end(self, trainer, _pl_module):
            from time import monotonic

            if self._epoch_start:
                self.epoch_times.append(monotonic() - self._epoch_start)
            train_loss = trainer.callback_metrics.get("train_loss_epoch")
            if train_loss is not None:
                self.epoch_train_losses.append(float(train_loss))

        def on_validation_epoch_end(self, trainer, _pl_module):
            val_loss = trainer.callback_metrics.get("val_loss_epoch")
            if val_loss is None:
                val_loss = trainer.callback_metrics.get("val_loss")
            if val_loss is not None:
                self.epoch_val_losses.append(float(val_loss))

    loss_callback = EpochLossCallback()

    # WandB logger (optional)
    logger = False
    wandb_run_url = None
    if wandb_project:
        from lightning.pytorch.loggers import WandbLogger

        sample_ids = sorted(filelist.read_text().strip().splitlines())
        dataset_version = os.environ.get("DATASET_VERSION", "")
        if not dataset_version:
            dataset_version = hashlib.md5("|".join(sample_ids).encode()).hexdigest()[:8]

        wandb_tags = [platform, f"ch{channels}", f"blk{residual_blocks}"]
        if gpu:
            wandb_tags.append("gpu")
        if dataset_version:
            wandb_tags.append(f"ds:{dataset_version}")
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
            group=dataset_version,
            tags=wandb_tags,
            config={
                "channels": channels,
                "residual_blocks": residual_blocks,
                "epochs": epochs,
                "gradient_checkpoint": gradient_checkpoint,
                "max_file_size_mb": max_file_size,
                "total_samples": len(sample_ids),
                "sample_ids": sample_ids,
                "dataset_version": dataset_version,
                "seed": seed,
                "instance_type": os.environ.get("INSTANCE_TYPE", ""),
                "github_run_id": run_id,
                "github_sha": os.environ.get("GITHUB_SHA", ""),
                "github_ref": os.environ.get("GITHUB_REF", ""),
            },
        )
        if run_id and repo:
            gha_url = f"https://github.com/{repo}/actions/runs/{run_id}"
            logger.experiment.notes = f"[GHA run {run_id}]({gha_url})"

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

    t0 = monotonic()
    trainer.fit(model, datamodule=datamodule)
    train_wallclock = monotonic() - t0

    if wandb_project and logger.experiment:
        logger.experiment.summary["wallclock_s"] = train_wallclock
        wandb_run_url = logger.experiment.get_url()

    # Get final losses
    final_val_loss = trainer.callback_metrics.get("val_loss_epoch")
    if final_val_loss is None:
        final_val_loss = trainer.callback_metrics.get("val_loss")
    if final_val_loss is None:
        raise RuntimeError(
            "final validation loss not found in trainer.callback_metrics "
            "(expected 'val_loss_epoch' or 'val_loss')"
        )
    final_val_loss = float(final_val_loss)

    final_train_loss = trainer.callback_metrics.get("train_loss_epoch")
    if final_train_loss is not None:
        final_train_loss = float(final_train_loss)

    return {
        "platform": platform,
        "final_val_loss": final_val_loss,
        "final_train_loss": final_train_loss,
        "epoch_val_losses": loss_callback.epoch_val_losses,
        "epoch_train_losses": loss_callback.epoch_train_losses,
        "epoch_times": loss_callback.epoch_times,
        "wallclock_s": train_wallclock,
        "wandb_run_url": wandb_run_url,
    }


# fmt: off
@command()
@option("-B", "--residual-blocks", default=2, help="Number of residual blocks (default: 2, production: 16)")
@option("-C", "--channels", default=8, help="Number of model channels (default: 8, production: 32-64)")
@option("-c", "--check/--no-check", default=True, help="Check val_loss against expected value")
@option("-d", "--data-root", default=None, help="Path to data root dir containing mp_filelist.txt (default: data/MP/chgcars)")
@option("-e", "--epochs", default=5, help="Number of training epochs")
@option("-G", "--gradient-checkpoint", is_flag=True, help="Enable gradient checkpointing (saves VRAM, slower)")
@option("-g", "--gpu", is_flag=True, help="Use GPU acceleration (if available)")
@option("-M", "--max-file-size", default=0, type=float, help="Skip input files larger than N MB (0=no limit)")
@option("-s", "--seed", default=42, help="Random seed for reproducibility")
@option("-t", "--tolerance", default=0.001, help="Tolerance for val_loss comparison (absolute)")
@option("-U", "--update-expected", is_flag=True, help="Update expected_values.json for current platform")
@option("-v", "--verbose", is_flag=True, help="Verbose output")
@option("-W", "--wandb-project", default=None, help="Enable WandB logging with this project name")
def main(
    residual_blocks: int,
    channels: int,
    check: bool,
    data_root: str | None,
    epochs: int,
    gradient_checkpoint: bool,
    gpu: bool,
    max_file_size: float,
    seed: int,
    tolerance: float,
    update_expected: bool,
    verbose: bool,
    wandb_project: str | None,
):
    """Run deterministic e2e training test."""
    expected_values_file = Path(__file__).parent.parent / "tests" / "expected_values.json"

    results = run_training(
        channels=channels,
        residual_blocks=residual_blocks,
        epochs=epochs,
        seed=seed,
        gpu=gpu,
        data_root=data_root,
        gradient_checkpoint=gradient_checkpoint,
        max_file_size=max_file_size,
        wandb_project=wandb_project,
        verbose=verbose,
    )

    platform = results["platform"]
    final_val_loss = results["final_val_loss"]

    if verbose:
        echo(f"\nResults for {platform}:")
        echo(f"  Final val_loss: {final_val_loss:.6f}")
        if results["final_train_loss"] is not None:
            echo(f"  Final train_loss: {results['final_train_loss']:.6f}")
        echo(f"  Epoch val_losses: {[f'{v:.6f}' for v in results['epoch_val_losses']]}")
        echo(f"  Epoch train_losses: {[f'{v:.6f}' for v in results['epoch_train_losses']]}")
        epoch_times = results.get("epoch_times", [])
        if epoch_times:
            echo(f"  Epoch times: {[f'{t:.1f}s' for t in epoch_times]}")
            echo(f"  Mean epoch: {sum(epoch_times) / len(epoch_times):.1f}s")
            echo(f"  Overhead (wallclock - sum epochs): {results['wallclock_s'] - sum(epoch_times):.1f}s")

    echo(f"Final val_loss: {final_val_loss:.6f}")

    if results["wandb_run_url"]:
        echo(f"WANDB_RUN_URL={results['wandb_run_url']}")

    # Update expected values file if requested
    if update_expected:
        if expected_values_file.exists():
            expected_values = json.loads(expected_values_file.read_text())
        else:
            expected_values = {}

        expected_values[platform] = {
            "final_val_loss": results["final_val_loss"],
            "final_train_loss": results["final_train_loss"],
            "epoch_val_losses": results["epoch_val_losses"],
            "epoch_train_losses": results["epoch_train_losses"],
        }
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

            if expected.get("epoch_val_losses") and results["epoch_val_losses"]:
                echo("\nPer-epoch val_loss comparison:", err=True)
                for i, (actual, exp) in enumerate(zip(
                    results["epoch_val_losses"],
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
