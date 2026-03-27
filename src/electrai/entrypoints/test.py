from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace

import torch
import yaml
from hydra.utils import instantiate
from lightning.pytorch import Trainer

from electrai.lightning import LightningGenerator

logger = logging.getLogger(__name__)


def _resolve_checkpoint(cfg) -> Path:
    """Find the best available checkpoint from config.

    Resolution order:
    1. cfg.ckpt_file — explicit path to a specific .ckpt file
    2. cfg.ckpt_path / "last.ckpt"
    3. cfg.ckpt_path / "best.ckpt"
    4. Latest ckpt_*.ckpt in cfg.ckpt_path (highest epoch by lexicographic sort)
    """
    ckpt_file = getattr(cfg, "ckpt_file", None)
    if ckpt_file is not None:
        ckpt = Path(ckpt_file)
        if ckpt.exists():
            return ckpt
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

    ckpt_path = Path(getattr(cfg, "ckpt_path", "./checkpoints"))

    # If ckpt_path is itself a file, use it directly
    if ckpt_path.is_file():
        return ckpt_path

    for name in ("last.ckpt", "best.ckpt"):
        candidate = ckpt_path / name
        if candidate.exists():
            return candidate

    # Glob for ckpt_*.ckpt and pick the latest epoch by lexicographic sort
    candidates = sorted(ckpt_path.glob("ckpt_*.ckpt"))
    if candidates:
        return candidates[-1]

    raise FileNotFoundError(
        f"No checkpoint found in {ckpt_path}. "
        "Set ckpt_file to an explicit path, or ensure ckpt_path contains "
        "last.ckpt, best.ckpt, or ckpt_*.ckpt files."
    )


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
    trainer = Trainer(
        logger=None,
        callbacks=None,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        precision=cfg.model_precision,
    )

    lit_model.test_cfg = SimpleNamespace(
        log_dir=log_dir, out_dir=out_dir, tmp_dir=tmp_dir, save_pred=cfg.save_pred
    )

    # -----------------------------
    # Resolve checkpoint and run test
    # -----------------------------
    ckpt = _resolve_checkpoint(cfg)
    logger.info("Using checkpoint: %s", ckpt)

    trainer.test(model=lit_model, datamodule=datamodule, ckpt_path=ckpt)

    # -----------------------------
    # Post-test analysis
    # -----------------------------
    metrics_csv = log_dir / "metrics.csv"
    if metrics_csv.exists():
        from electrai.scripts.analyze.summarize import plot_distribution, summarize

        summary_text = summarize(metrics_csv, output_dir=log_dir)
        logger.info("\n%s", summary_text)
        plot_distribution(metrics_csv, output_dir=log_dir)

        # Optional: saturation analysis (always possible with enriched CSV)
        analyze_cfg = getattr(cfg, "analyze", None)
        run_analysis = analyze_cfg is None or getattr(analyze_cfg, "enabled", True)

        if run_analysis:
            from electrai.scripts.analyze.analyze_saturation import analyze_metrics

            saturation_dir = log_dir / "saturation"
            try:
                analyze_metrics(metrics_csv, saturation_dir)
            except (KeyError, ValueError) as e:
                logger.warning("Saturation analysis skipped: %s", e)

            # Tail analysis requires metadata CSV
            metadata_path = (
                getattr(analyze_cfg, "metadata", None) if analyze_cfg else None
            )
            if metadata_path is not None:
                from electrai.scripts.analyze.analyze_tail import main as tail_main

                tail_dir = log_dir / "tail"
                tail_main(
                    [
                        "--metrics",
                        str(metrics_csv),
                        "--metadata",
                        str(metadata_path),
                        "--output-dir",
                        str(tail_dir),
                    ]
                )
