from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path
from types import SimpleNamespace

# Required for deterministic CUBLAS (cumulative/matmul-heavy ops) on CUDA >=10.2. Must be set
# before the first CUDA context is created, so this has to happen before `import torch`.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
import yaml
from hydra.utils import instantiate
from lightning.pytorch import Callback, Trainer, seed_everything
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint

from electrai.lightning import LightningGenerator

logger = logging.getLogger(__name__)


class BestCheckpointMirror(Callback):
    """Mirror the primary ModelCheckpoint's best checkpoint to ``best_{value}.ckpt``.

    The primary ModelCheckpoint tracks the global best across chained resume-from-last.ckpt
    jobs, but the best file lives under a rotating ``ckpt_{epoch}_{val_loss}.ckpt`` name. This
    callback copies the current best to ``best_{score}.ckpt`` on rank 0 whenever it improves and
    removes any prior ``best_*.ckpt``, so exactly one global-best file (named by its monitored
    value) is kept, even across job restarts.
    """

    def __init__(self, model_checkpoint, prefix="best"):
        self._mc = model_checkpoint
        self._prefix = prefix
        self._mirrored = None

    def on_validation_end(self, trainer, _pl_module):
        self._mirror(trainer)

    def on_fit_end(self, trainer, _pl_module):
        # Lightning runs ModelCheckpoint last, so the final validation's best is
        # only saved after our on_validation_end. Mirror once more at fit end so
        # the last improvement is never missed.
        self._mirror(trainer)

    def _mirror(self, trainer):
        if not trainer.is_global_zero:
            return
        src = self._mc.best_model_path
        score = self._mc.best_model_score
        if not (src and score is not None and src != self._mirrored):
            return
        src_path = Path(src)
        if not src_path.exists():
            return
        out_dir = Path(self._mc.dirpath or trainer.default_root_dir)
        dest = out_dir / f"{self._prefix}_{float(score):.6f}.ckpt"
        # copy (not symlink) so it survives rotation of the source ckpt file
        shutil.copyfile(src_path, dest)
        # keep exactly one best_*.ckpt: drop older ones (incl. leftovers from prior jobs)
        for old in out_dir.glob(f"{self._prefix}_*.ckpt"):
            if old.resolve() != dest.resolve():
                old.unlink()
        self._mirrored = src


def train(args):
    # -----------------------------
    # Load YAML config
    # -----------------------------
    config_path = Path(args.config)
    with Path.open(config_path) as f:
        cfg_dict = yaml.safe_load(f)
    cfg = SimpleNamespace(**cfg_dict)

    # -----------------------------
    # Seed (reproducibility)
    # -----------------------------
    # Top-level `seed` controls model weight init + data split/shuffling/worker order via
    # Lightning's seed_everything (python/numpy/torch, incl. CUDA). Nothing seeded this before,
    # so weight init came from whatever random state the process happened to start with --
    # fine for an ensemble (each job gets a different init "for free") but not reproducible.
    # Falls back to data.random_seed (legacy field, split-only) if `seed` isn't set, so existing
    # configs keep working; for a true multi-init ensemble, give each run its own `seed` value.
    seed = getattr(cfg, "seed", None)
    if seed is None:
        seed = cfg_dict.get("data", {}).get("random_seed", 42)
    seed_everything(seed, workers=True)

    # cuDNN's autotuner (benchmark=True) picks conv algorithms by timing, which varies run to
    # run; deterministic=True forces the same (slower) algorithm every time given the same
    # input shape. use_deterministic_algorithms extends this to non-cudnn ops (e.g. index_add,
    # scatter_add) that are otherwise nondeterministic on CUDA due to atomic-add ordering.
    # Verified on an A100 (della-l09g4) with strict mode (warn_only=False): a full ResUNet3D
    # forward+backward (incl. the trilinear nn.Upsample in PeriodicUpsampleConv3d) produces
    # bit-identical params/grads/output across reruns of the same seed on this torch/CUDA build
    # (2.10.0+cu128) -- no RuntimeError, despite older torch versions lacking a deterministic
    # CUDA kernel for trilinear-mode interpolate backward.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=False)

    # -----------------------------
    # Data
    # -----------------------------
    datamodule = instantiate(cfg.data)

    # -----------------------------
    # Model (LightningModule handles architecture + loss + optimizer)
    # -----------------------------
    lit_model = LightningGenerator(cfg)

    # Weights-only fine-tune start. Distinct from the `ckpt_path / "last.ckpt"` resume below,
    # which restores optimizer, LR schedule and epoch counters -- with that, cfg.lr and
    # cfg.epochs would be interpreted against the pretraining run's progress. warm_start_path
    # loads just the weights and trains as a fresh run. A same-run resume still wins: if this
    # run's own last.ckpt exists, fit() restores over these weights, which is correct.
    warm_start = getattr(cfg, "warm_start_path", None) or None
    if warm_start:
        import torch as _torch

        _ckpt = _torch.load(warm_start, map_location="cpu", weights_only=False)
        _state = _ckpt.get("state_dict", _ckpt)
        _missing, _unexpected = lit_model.load_state_dict(_state, strict=False)
        if _missing or _unexpected:
            raise RuntimeError(
                f"warm start {warm_start}: {len(_missing)} missing / "
                f"{len(_unexpected)} unexpected keys -- architecture mismatch"
            )
        logger.info("warm start (weights only) from %s", warm_start)

    # -----------------------------
    # Logging and callbacks
    # -----------------------------
    wandb_mode = getattr(cfg, "wandb_mode", "disabled").lower()
    os.environ["WANDB_MODE"] = wandb_mode
    if wandb_mode != "disabled":
        from lightning.pytorch.loggers import WandbLogger

        wandb_logger = WandbLogger(
            project=cfg.wb_pname,
            entity=cfg.entity,
            name=getattr(cfg, "run_name", None),
            config=vars(cfg),
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

    callbacks = [checkpoint_cb, BestCheckpointMirror(checkpoint_cb), lr_monitor]

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
        strategy="ddp",
        log_every_n_steps=1,
        gradient_clip_val=getattr(cfg, "gradient_clip_value", 1.0),
        # validate every N epochs (default 1 = every epoch, as before). For fine-tunes with a few training
        # structures, where an epoch is a few steps and validation over many structures would dominate.
        check_val_every_n_epoch=int(getattr(cfg, "check_val_every_n_epoch", 1)),
        # size_balanced shards the data itself; Lightning must not wrap its sampler
        use_distributed_sampler=getattr(datamodule, "train_sampler", "random")
        == "random",
    )

    # -----------------------------
    # Train
    # -----------------------------
    ckpt = ckpt_path / "last.ckpt"
    trainer.fit(
        lit_model, datamodule=datamodule, ckpt_path=ckpt if ckpt.exists() else None
    )
