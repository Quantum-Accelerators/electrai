from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import torch
import yaml
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from src.electrai.dataloader.registry import get_data
from src.electrai.lightning import LightningGenerator
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate

torch.backends.cudnn.conv.fp32_precision = "tf32"


def collate_fn(batch):
    try:
        return default_collate(batch)
    except Exception:
        # Separate and return as lists of tensors
        x, y = zip(*batch, strict=False)
        return list(x), list(y)


def train(args):
    # -----------------------------
    # Load YAML config
    # -----------------------------
    config_path = Path(args.config)
    with Path.open(config_path) as f:
        cfg_dict = yaml.safe_load(f)
    cfg = SimpleNamespace(**cfg_dict)

    assert 0 < cfg.train_fraction < 1, "train_fraction must be between 0 and 1."

    # -----------------------------
    # Data
    # -----------------------------
    train_data, test_data = get_data(cfg)
    train_loader = DataLoader(
        train_data,
        batch_size=int(cfg.nbatch),
        shuffle=True,
        num_workers=cfg.num_workers,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        test_data,
        batch_size=int(cfg.nbatch),
        shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=collate_fn,
    )

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

    checkpoint_cb = ModelCheckpoint(
        monitor="val_loss",
        save_top_k=2,
        mode="min",
        filename=f"{cfg.model_prefix}" + "_{epoch:02d}_{val_loss:.6f}",
        save_last=True,
    )

    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    # -----------------------------
    # Trainer
    # -----------------------------
    trainer = Trainer(
        max_epochs=int(cfg.epochs),
        logger=wandb_logger,
        callbacks=[checkpoint_cb, lr_monitor],
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        precision=cfg.model_precision,
        log_every_n_steps=1,
    )

    # -----------------------------
    # Train
    # -----------------------------
    trainer.fit(lit_model, train_loader, test_loader)
