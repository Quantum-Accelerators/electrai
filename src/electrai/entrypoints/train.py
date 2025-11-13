from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import torch
import yaml
from lightning.pytorch.profilers import PyTorchProfiler
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from src.electrai.dataloader.registry import get_data
from src.electrai.lightning import LightningGenerator
from torch.utils.data import DataLoader

os.environ["WANDB_MODE"] = "online"
torch.set_float32_matmul_precision("medium")


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
    )
    test_loader = DataLoader(
        test_data,
        batch_size=int(cfg.nbatch),
        shuffle=False,
        num_workers=cfg.num_workers,
    )

    # -----------------------------
    # Model (LightningModule handles architecture + loss + optimizer)
    # -----------------------------
    lit_model = LightningGenerator(cfg)

    # -----------------------------
    # Logging and callbacks
    # -----------------------------
    wandb_logger = WandbLogger(
        project=cfg.wb_pname, name=cfg.wb_ename, entity=cfg.entity, config=vars(cfg)
    )

    checkpoint_cb = ModelCheckpoint(
        monitor="val_loss",
        save_top_k=2,
        mode="min",
        filename=f"{cfg.model_prefix}" + "_{epoch:02d}_{val_loss:.6f}",
        save_last=True,
    )

    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    # -----------------------------
    # Profiler
    # -----------------------------
    profiler = PyTorchProfiler(
        dirpath=cfg.profile_dir,
        filename="pytorch_profile",
        schedule=torch.profiler.schedule(wait=0, warmup=0, active=float("inf")),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(cfg.profile_dir),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    )

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
        log_every_n_steps=10,
        profiler=profiler,
    )

    # -----------------------------
    # Train
    # -----------------------------
    trainer.fit(lit_model, train_loader, test_loader)
