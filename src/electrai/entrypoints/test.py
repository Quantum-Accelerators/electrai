from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import torch
import yaml
from hydra.utils import instantiate
from lightning.pytorch import Trainer
from src.electrai.lightning import LightningGenerator

torch.backends.cudnn.conv.fp32_precision = "tf32"


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
    test_loader = datamodule.test_dataloader()

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
    trainer = Trainer(
        logger=None,
        callbacks=None,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        precision=cfg.model_precision,
    )

    # -----------------------------
    # Train
    # -----------------------------
    ckpt = ckpt_path / "last.ckpt"
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

    trainer.test(model=lit_model, dataloaders=test_loader, ckpt_path=ckpt)
