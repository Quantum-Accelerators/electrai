from __future__ import annotations

import os
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import torch
import yaml
from hydra.utils import instantiate
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.strategies import DDPStrategy

from electrai.lightning import LightningGenerator


def train(args):
    # -----------------------------
    # Load YAML config
    # -----------------------------
    config_path = Path(args.config)
    with Path.open(config_path) as f:
        cfg_dict = yaml.safe_load(f)
    cfg = SimpleNamespace(**cfg_dict)

    # -----------------------------
    # Data
    # -----------------------------
    datamodule = instantiate(cfg.data)

    # -----------------------------
    # Model (LightningModule handles architecture + loss + optimizer)
    # -----------------------------
    lit_model = LightningGenerator(cfg)

    # -----------------------------
    # Logging and callbacks
    # -----------------------------
    wandb_mode = getattr(cfg, "wandb_mode", "disabled").lower()
    os.environ["WANDB_MODE"] = wandb_mode
    if wandb_mode != "disabled":
        from lightning.pytorch.loggers import WandbLogger

        # Short width-first run name (w128_0729-1912): auto-generated names
        # made restart segments of different-width runs indistinguishable,
        # and full config-derived names were too long for the project view.
        model_cfg = getattr(cfg, "model", None) or {}
        n_ch = model_cfg.get("n_channels") if isinstance(model_cfg, dict) else None
        stamp = datetime.now(UTC).strftime("%m%d-%H%M")
        run_name = f"w{n_ch}_{stamp}" if n_ch else getattr(cfg, "run_name", None)
        wandb_logger = WandbLogger(
            project=cfg.wb_pname, entity=cfg.entity, name=run_name, config=vars(cfg)
        )
    else:
        wandb_logger = None

    ckpt_path = Path(getattr(cfg, "ckpt_path", "./checkpoints"))
    checkpoint_cb = ModelCheckpoint(
        dirpath=ckpt_path,
        monitor="val_loss",
        save_top_k=2,
        mode="min",
        filename="ckpt_{epoch:02d}_{val_loss:.6f}",
    )

    # Frequent resume checkpoint. An epoch here is tens of thousands of steps
    # and can crash mid-way (e.g. a large sample tripping the NCCL watchdog), so
    # save `last.ckpt` periodically; a restart then resumes instead of redoing
    # the epoch from scratch. `val_loss` only exists at epoch end, so this is a
    # separate monitor-less callback (save_top_k=0 -> last.ckpt only).
    # Step-based, NOT train_time_interval: the wall-clock trigger needs a DDP
    # broadcast to align ranks and their clocks can disagree about when the
    # interval fired, which deadlocked a 4-rank run mid-epoch (ranks in
    # mismatched collectives; NCCL watchdog never fires). Step counts are
    # identical on every rank, so no alignment collective is needed.
    last_ckpt_cb = ModelCheckpoint(
        dirpath=ckpt_path, every_n_train_steps=2000, save_top_k=0, save_last=True
    )

    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    callbacks = [checkpoint_cb, last_ckpt_cb, lr_monitor]

    hf_cfg = getattr(cfg, "hf", None)
    if hf_cfg and hf_cfg.get("repo_id"):
        from electrai.callbacks.hf_upload import HuggingFaceCallback

        callbacks.append(HuggingFaceCallback(cfg))

    # -----------------------------
    # Trainer
    # -----------------------------
    local_world_size = int(
        os.environ.get("LOCAL_WORLD_SIZE", torch.cuda.device_count())
    )
    world_size = int(os.environ.get("WORLD_SIZE", local_world_size))
    num_nodes = max(1, world_size // local_world_size)
    trainer = Trainer(
        max_epochs=int(cfg.epochs),
        logger=wandb_logger,
        callbacks=callbacks,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        precision=cfg.precision,
        devices="auto",
        num_nodes=num_nodes,
        # Raise the NCCL collective timeout from the 30-min default: a single
        # large sample's forward/backward can stall the all-reduce on the other
        # ranks past 30 min and abort the whole group. The grid-size cap keeps
        # steps short; this is a safety net for the occasional slow one.
        strategy=DDPStrategy(timeout=timedelta(hours=2)),
        log_every_n_steps=1,
        gradient_clip_val=getattr(cfg, "gradient_clip_value", 1.0),
    )

    # -----------------------------
    # Train
    # -----------------------------
    ckpt = ckpt_path / "last.ckpt"
    trainer.fit(
        lit_model, datamodule=datamodule, ckpt_path=ckpt if ckpt.exists() else None
    )
