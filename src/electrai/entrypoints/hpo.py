"""Hyperparameter optimization entrypoint using Optuna.

Run with regular python (NOT torchrun):
    uv run python src/electrai/entrypoints/hpo.py --config path/to/config.yaml

For multi-GPU training within each trial, set devices in config or use --devices flag:
    uv run python src/electrai/entrypoints/hpo.py --config path/to/config.yaml --devices 8
"""

from __future__ import annotations

import copy
import logging
import os
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING

import optuna
import torch
import yaml
from hydra.utils import instantiate
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from optuna.integration import PyTorchLightningPruningCallback
from src.electrai.lightning import LightningGenerator

if TYPE_CHECKING:
    from optuna import Trial

logger = logging.getLogger(__name__)


def get_gpu_config(hpo_cfg: dict, args) -> tuple[int, str]:
    """Determine number of devices and strategy for training.

    Returns (num_devices, strategy) tuple.

    Note: We use ddp_spawn instead of ddp because the regular ddp strategy
    uses subprocess launching which re-runs the entire script on each GPU.
    This causes each process to create different Optuna trials with different
    hyperparameters, leading to model mismatch errors across ranks.
    ddp_spawn uses torch.multiprocessing.spawn which creates child processes
    from within the training call, ensuring all ranks use the same model.
    """
    # Priority: CLI args > config > auto-detect
    if hasattr(args, "devices") and args.devices is not None:
        num_devices = args.devices
    else:
        num_devices = hpo_cfg.get("devices", "auto")

    if num_devices == "auto":
        num_devices = torch.cuda.device_count() if torch.cuda.is_available() else 1

    # Use ddp_spawn for multi-GPU (not ddp which uses subprocess launcher)
    strategy = (
        "ddp_spawn" if isinstance(num_devices, int) and num_devices > 1 else "auto"
    )

    return num_devices, strategy


def suggest_hyperparameters(trial: Trial, search_space: dict) -> dict:
    """Suggest hyperparameters from the search space using Optuna trial."""
    params = {}
    for name, spec in search_space.items():
        param_type = spec["type"]
        if param_type == "float":
            params[name] = trial.suggest_float(
                name, spec["low"], spec["high"], log=spec.get("log", False)
            )
        elif param_type == "int":
            params[name] = trial.suggest_int(name, spec["low"], spec["high"])
        elif param_type == "categorical":
            params[name] = trial.suggest_categorical(name, spec["choices"])
        else:
            msg = f"Unknown parameter type: {param_type}"
            raise ValueError(msg)
    return params


def apply_hyperparameters(cfg_dict: dict, params: dict) -> dict:
    """Apply suggested hyperparameters to the config dictionary."""
    cfg = copy.deepcopy(cfg_dict)

    # Model parameters
    if "depth" in params:
        cfg["model"]["depth"] = params["depth"]
    if "n_channels" in params:
        cfg["model"]["n_channels"] = params["n_channels"]
    if "n_residual_blocks" in params:
        cfg["model"]["n_residual_blocks"] = params["n_residual_blocks"]
    if "kernel_size" in params:
        cfg["model"]["kernel_size"] = params["kernel_size"]

    # Training parameters
    if "lr" in params:
        cfg["lr"] = params["lr"]
    if "weight_decay" in params:
        cfg["weight_decay"] = params["weight_decay"]
    if "warmup_length" in params:
        cfg["warmup_length"] = params["warmup_length"]
    if "gradient_clip_value" in params:
        cfg["gradient_clip_value"] = params["gradient_clip_value"]

    # Data parameters
    if "batch_size" in params:
        cfg["data"]["batch_size"] = params["batch_size"]
    if "augmentation" in params:
        cfg["data"]["augmentation"] = params["augmentation"]

    return cfg


def create_objective(cfg_dict: dict, hpo_cfg: dict, args):
    """Create the Optuna objective function."""
    # Determine GPU configuration once for all trials
    num_devices, strategy = get_gpu_config(hpo_cfg, args)
    logger.info(f"Training config: {num_devices} device(s), strategy={strategy}")

    def objective(trial: Trial) -> float:
        # Suggest hyperparameters
        params = suggest_hyperparameters(trial, hpo_cfg["search_space"])

        # Apply to config
        trial_cfg_dict = apply_hyperparameters(cfg_dict, params)
        cfg = SimpleNamespace(**trial_cfg_dict)

        # Check memory constraints: depth vs n_channels
        depth = trial_cfg_dict["model"]["depth"]
        n_channels = trial_cfg_dict["model"]["n_channels"]
        max_channels_at_bottleneck = n_channels * (2**depth)
        if max_channels_at_bottleneck > 512:
            # Skip configurations likely to OOM
            logger.warning(
                f"Pruning trial {trial.number}: depth={depth}, n_channels={n_channels} "
                f"would create {max_channels_at_bottleneck} channels at bottleneck"
            )
            raise optuna.TrialPruned

        # Create data module
        datamodule = instantiate(cfg.data)

        # Create model
        lit_model = LightningGenerator(cfg)

        # Set up W&B logging if enabled
        wandb_mode = getattr(cfg, "wandb_mode", "disabled").lower()
        os.environ["WANDB_MODE"] = wandb_mode
        if wandb_mode != "disabled":
            wandb_logger = WandbLogger(
                project=getattr(cfg, "wb_pname", "mp-hpo"),
                entity=getattr(cfg, "entity", None),
                name=f"trial_{trial.number}",
                group=hpo_cfg.get("study_name", "resunet_hpo"),
                config={
                    **params,
                    "trial_number": trial.number,
                    "num_devices": num_devices,
                },
                reinit=True,
            )
        else:
            wandb_logger = None

        # Callbacks - only use pruning callback on single GPU (DDP has issues with it)
        callbacks = [
            ModelCheckpoint(
                dirpath=Path(cfg.ckpt_path) / f"trial_{trial.number}",
                monitor="val_loss",
                save_top_k=1,
                mode="min",
            ),
            EarlyStopping(monitor="val_loss", patience=5, mode="min"),
        ]
        # Optuna pruning callback doesn't work well with DDP strategies
        if strategy not in ("ddp", "ddp_spawn"):
            callbacks.insert(
                0, PyTorchLightningPruningCallback(trial, monitor="val_loss")
            )

        # Trainer with multi-GPU support
        trainer = Trainer(
            max_epochs=int(cfg.epochs),
            callbacks=callbacks,
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            devices=num_devices,
            num_nodes=1,
            strategy=strategy,
            precision=cfg.precision,
            enable_progress_bar=True,
            enable_model_summary=trial.number == 0,  # Only show summary for first trial
            logger=wandb_logger,
            gradient_clip_val=getattr(cfg, "gradient_clip_value", 1.0),
        )

        try:
            trainer.fit(lit_model, datamodule=datamodule)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.warning(f"Trial {trial.number} OOM: {e}")
                torch.cuda.empty_cache()
                if wandb_logger is not None:
                    wandb_logger.experiment.finish(exit_code=1)
                raise optuna.TrialPruned from e
            raise

        # Return best validation loss
        val_loss = trainer.callback_metrics.get("val_loss")

        # Log final metrics and finish W&B run
        if wandb_logger is not None:
            if val_loss is not None:
                wandb_logger.experiment.summary["best_val_loss"] = val_loss.item()
            wandb_logger.experiment.finish()

        if val_loss is None:
            raise optuna.TrialPruned

        return val_loss.item()

    return objective


def create_sampler(sampler_cfg: dict) -> optuna.samplers.BaseSampler:
    """Create Optuna sampler from config."""
    sampler_type = sampler_cfg.get("type", "tpe").lower()
    if sampler_type == "tpe":
        return optuna.samplers.TPESampler()
    if sampler_type == "random":
        return optuna.samplers.RandomSampler()
    if sampler_type == "grid":
        msg = "Grid sampler requires explicit search_space definition"
        raise ValueError(msg)
    msg = f"Unknown sampler type: {sampler_type}"
    raise ValueError(msg)


def create_pruner(pruner_cfg: dict) -> optuna.pruners.BasePruner:
    """Create Optuna pruner from config."""
    pruner_type = pruner_cfg.get("type", "median").lower()
    n_startup = pruner_cfg.get("n_startup_trials", 5)
    n_warmup = pruner_cfg.get("n_warmup_steps", 3)

    if pruner_type == "median":
        return optuna.pruners.MedianPruner(
            n_startup_trials=n_startup, n_warmup_steps=n_warmup
        )
    if pruner_type == "hyperband":
        return optuna.pruners.HyperbandPruner()
    if pruner_type == "none":
        return optuna.pruners.NopPruner()
    msg = f"Unknown pruner type: {pruner_type}"
    raise ValueError(msg)


def run_hpo(args):
    """Run hyperparameter optimization study."""
    # Load config
    config_path = Path(args.config)
    with Path.open(config_path) as f:
        cfg_dict = yaml.safe_load(f)

    hpo_cfg = cfg_dict.get("hpo", {})

    # Create study
    sampler = create_sampler(hpo_cfg.get("sampler", {}))
    pruner = create_pruner(hpo_cfg.get("pruner", {}))

    study = optuna.create_study(
        study_name=hpo_cfg.get("study_name", "resunet_hpo"),
        storage=hpo_cfg.get("storage"),
        direction=hpo_cfg.get("direction", "minimize"),
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True,
    )

    # Create objective
    objective = create_objective(cfg_dict, hpo_cfg, args)

    # Run optimization
    study.optimize(
        objective,
        n_trials=hpo_cfg.get("n_trials", 50),
        timeout=hpo_cfg.get("timeout"),
        gc_after_trial=True,
        show_progress_bar=True,
    )

    # Report results
    logger.info("=" * 60)
    logger.info("HPO Study Complete")
    logger.info("=" * 60)
    logger.info(f"Best trial: {study.best_trial.number}")
    logger.info(f"Best value: {study.best_value:.6f}")
    logger.info("Best hyperparameters:")
    for key, value in study.best_params.items():
        logger.info(f"  {key}: {value}")

    # Save best config
    best_cfg = apply_hyperparameters(cfg_dict, study.best_params)
    del best_cfg["hpo"]  # Remove HPO section from final config
    best_config_path = config_path.parent / f"{config_path.stem}_best.yaml"
    with Path.open(best_config_path, "w") as f:
        yaml.dump(best_cfg, f, default_flow_style=False)
    logger.info(f"Best config saved to: {best_config_path}")

    return study


if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    parser = argparse.ArgumentParser(description="Run HPO for ElectrAI ResUNet")
    parser.add_argument(
        "--config",
        type=str,
        default="src/electrai/configs/MP/config_hpo.yaml",
        help="Path to HPO config file",
    )
    parser.add_argument(
        "--devices",
        type=int,
        default=None,
        help="Number of GPUs to use per trial (default: auto-detect all available)",
    )
    args = parser.parse_args()

    run_hpo(args)
