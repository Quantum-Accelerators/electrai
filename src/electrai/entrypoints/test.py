from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import torch
import yaml
from hydra.utils import instantiate
from lightning.pytorch import Trainer

from electrai.lightning import LightningGenerator


def test(args):
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
    lit_model.test_cfg = SimpleNamespace(log_dir=cfg.log_dir, out_dir=cfg.out_dir)

    # -----------------------------
    # Callback
    # -----------------------------
    ckpt_path = Path(getattr(cfg, "ckpt_path", "./checkpoints"))

    # -----------------------------
    # Trainer
    # -----------------------------
    if cfg.save_pred:
        out_dir = Path(getattr(cfg, "out_dir", "predictions"))
        out_dir.mkdir(exist_ok=True, parents=True)
    else:
        out_dir = None
    log_dir = Path(getattr(cfg, "log_dir", "logs"))
    tmp_dir = log_dir / "tmp"
    for directory in [log_dir, tmp_dir]:
        directory.mkdir(exist_ok=True, parents=True)
    local_world_size = int(
        os.environ.get("LOCAL_WORLD_SIZE", torch.cuda.device_count())
    )
    world_size = int(os.environ.get("WORLD_SIZE", local_world_size))
    num_nodes = max(1, world_size // local_world_size)
    trainer = Trainer(
        logger=None,
        callbacks=None,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices="auto",
        num_nodes=num_nodes,
        precision=cfg.precision,
        strategy="ddp",
    )

    lit_model.test_cfg = SimpleNamespace(
        log_dir=log_dir, out_dir=out_dir, tmp_dir=tmp_dir, save_pred=cfg.save_pred
    )

    # -----------------------------
    # Train
    # -----------------------------
    # If ckpt_path points to a specific checkpoint file, use it directly;
    # otherwise treat it as a directory and fall back to last.ckpt.
    ckpt = ckpt_path if ckpt_path.is_file() else ckpt_path / "last.ckpt"
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

    trainer.test(model=lit_model, datamodule=datamodule, ckpt_path=ckpt)
