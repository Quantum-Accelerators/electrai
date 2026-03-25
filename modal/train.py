"""Modal training entrypoint for electrai.

Run real training experiments on Modal GPUs with data from the
electrai-data Volume and checkpoints persisted to electrai-checkpoints.

Usage:
    # Default config (ResUNet, dataset_4, 50 epochs, L4)
    modal run modal/train.py

    # Custom config file
    modal run modal/train.py --config examples/MP/experiments/experiment_0/config.yaml

    # Override GPU, epochs, channels
    modal run modal/train.py --gpu A100 --epochs 10 --channels 64

    # Resume from last checkpoint
    modal run modal/train.py --gpu A100 --resume
"""

from __future__ import annotations

from pathlib import Path

import modal

ROOT = Path(__file__).parent.parent

# Persistent volumes
data_volume = modal.Volume.from_name("electrai-data")
ckpt_volume = modal.Volume.from_name("electrai-checkpoints", create_if_missing=True)

# Dependencies read from pyproject.toml (shared with ci.py, populate_volume.py)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install_from_pyproject(
        str(ROOT / "pyproject.toml"), optional_dependencies=["dev"]
    )
    .add_local_dir(str(ROOT / "src"), remote_path="/root/electrai/src", copy=True)
    .add_local_dir(
        str(ROOT / "examples"), remote_path="/root/electrai/examples", copy=True
    )
    .add_local_file(
        str(ROOT / "pyproject.toml"),
        remote_path="/root/electrai/pyproject.toml",
        copy=True,
    )
    .run_commands("cd /root/electrai && pip install --no-deps -e .")
)

app = modal.App("electrai-train", image=image)

DATA_ROOT = "/data/mp/chg_datasets/dataset_4"
CKPT_ROOT = "/checkpoints"


@app.function(
    gpu="L4",
    volumes={"/data": data_volume, CKPT_ROOT: ckpt_volume},
    secrets=[modal.Secret.from_name("wandb-credentials")],
    timeout=14400,  # 4 hours
    retries=0,
)
def train(config_json: str, gpu_type: str = "L4"):
    """Run training with the given config (as JSON, converted to YAML remotely)."""
    import json
    import logging
    import subprocess
    import sys

    import yaml

    log = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)

    cfg = json.loads(config_json)

    # Write config as YAML for the training entrypoint
    config_path = Path("/tmp/config.yaml")
    with config_path.open("w") as f:
        yaml.dump(cfg, f, default_flow_style=False)

    log.info("GPU: %s", gpu_type)
    log.info("Data root: %s", DATA_ROOT)
    log.info("Checkpoint dir: %s", CKPT_ROOT)

    # Verify data exists
    filelist = Path(DATA_ROOT) / "mp_filelist.txt"
    if not filelist.exists():
        raise FileNotFoundError(f"Filelist not found: {filelist}")
    n_samples = len(filelist.read_text().strip().splitlines())
    log.info("Dataset: %d samples", n_samples)

    # Check for existing checkpoint to resume from
    ckpt_path = Path(cfg.get("ckpt_path", CKPT_ROOT))
    last_ckpt = ckpt_path / "last.ckpt"
    if last_ckpt.exists():
        log.info("Found checkpoint: %s", last_ckpt)
    else:
        log.info("No checkpoint found, starting from scratch")

    # Run training
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "electrai.entrypoints.main",
            "train",
            "--config",
            str(config_path),
        ],
        cwd="/root/electrai",
        check=False,
    )

    # Persist checkpoints
    ckpt_volume.commit()

    if result.returncode != 0:
        raise RuntimeError(f"Training failed with exit code {result.returncode}")

    log.info("Training complete. Checkpoints saved to electrai-checkpoints volume.")


@app.local_entrypoint()
def main(
    config: str = "",
    gpu: str = "L4",
    epochs: int = 50,
    channels: int = 32,
    residual_blocks: int = 1,
    depth: int = 2,
    kernel_size: int = 5,
    lr: float = 0.01,
    batch_size: int = 1,
    val_frac: float = 0.005,
    wandb_project: str = "mp-experiment",
    resume: bool = False,
):
    import json
    import logging

    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger(__name__)

    if config:
        import subprocess

        result = subprocess.run(
            [
                "python3",
                "-c",
                f"import yaml, json; print(json.dumps(yaml.safe_load(open('{config}'))))",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        cfg = json.loads(result.stdout)
    else:
        cfg = {
            "data": {
                "_target_": "electrai.dataloader.dataset.RhoRead",
                "root": f"{DATA_ROOT}/mp_filelist.txt",
                "split_file": None,
                "precision": "f32",
                "batch_size": batch_size,
                "train_workers": 4,
                "val_workers": 2,
                "pin_memory": False,
                "val_frac": val_frac,
                "drop_last": False,
                "augmentation": False,
                "random_seed": 42,
            },
            "model": {
                "_target_": "electrai.model.resunet.ResUNet3D",
                "in_channels": 1,
                "out_channels": 1,
                "n_channels": channels,
                "n_residual_blocks": residual_blocks,
                "kernel_size": kernel_size,
                "depth": depth,
                "use_checkpoint": False,
            },
            "precision": 32,
            "epochs": epochs,
            "lr": lr,
            "weight_decay": 0.0,
            "warmup_length": 1,
            "beta1": 0.9,
            "beta2": 0.99,
            "wandb_mode": "online",
            "entity": "PrinceOA",
            "wb_pname": wandb_project,
            "ckpt_path": CKPT_ROOT,
        }

    # Always override data root and checkpoint path for Modal
    if "data" in cfg:
        cfg["data"]["root"] = f"{DATA_ROOT}/mp_filelist.txt"
    cfg["ckpt_path"] = CKPT_ROOT

    if not resume:
        cfg.pop("resume_from_checkpoint", None)

    config_json = json.dumps(cfg, indent=2)
    log.info("Config:\n%s", config_json)

    train_fn = train
    if gpu != "L4":
        train_fn = train.with_options(gpu=gpu)

    train_fn.remote(config_json=config_json, gpu_type=gpu)
    log.info("Done.")
