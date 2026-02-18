from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import torch
import yaml
from hydra.utils import instantiate
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint

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

        wandb_logger = WandbLogger(
            project=cfg.wb_pname, entity=cfg.entity, config=vars(cfg)
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
        save_last=True,
    )

    lr_monitor = LearningRateMonitor(logging_interval="epoch")

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
        callbacks=[checkpoint_cb, lr_monitor],
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        precision=cfg.precision,
        devices="auto",
        num_nodes=num_nodes,
        strategy="ddp",
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
