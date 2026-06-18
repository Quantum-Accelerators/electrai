"""Modal training entrypoint for electrai.

Run real training experiments on Modal GPUs with data from the
electrai-data Volume and checkpoints persisted to electrai-checkpoints.

Handles both single-dataset and multi-dataset (`datasets:` list) configs.
Della absolute paths in the config are remapped onto the Volume mount, and
checkpoints are namespaced by `run_name` so the entrypoint auto-resumes from
`<ckpt>/last.ckpt` across successive (24h-capped) invocations.

Usage:
    # Multi-dataset experiment config on a single A100 (80GB)
    modal run modal/train.py --config src/electrai/configs/MP/config_gga_gga+u_f32.yaml

    # Short subset smoke run
    modal run modal/train.py --config src/electrai/configs/MP/config_gga_gga+u_f32_smoke.yaml

    # Pick a different GPU (e.g. multi-GPU DDP once throughput is known)
    modal run modal/train.py --config <cfg> --gpu "A100-80GB:8"
"""

from __future__ import annotations

from pathlib import Path

import modal

ROOT = Path(__file__).parent.parent

# Persistent volumes
data_volume = modal.Volume.from_name("electrai-data", create_if_missing=True)
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

# Della ROSENGROUP share root and where it is mounted on Modal. Config paths are
# written as della absolute paths; they are rewritten onto the Volume here.
DELLA_SHARE_PREFIX = "/scratch/gpfs/ROSENGROUP/common/globus_share_OA"
VOLUME_ROOT = "/data"
CKPT_ROOT = "/checkpoints"
DEFAULT_GPU = "A100-80GB"


def _remap_path(path: str | None) -> str | None:
    """Rewrite a della share path onto the Volume mount. Idempotent."""
    if not path:
        return path
    if path.startswith(VOLUME_ROOT):
        return path
    if path.startswith(DELLA_SHARE_PREFIX):
        rel = path[len(DELLA_SHARE_PREFIX) :].lstrip("/")
        return str(Path(VOLUME_ROOT) / rel)
    return path


def _remap_data_paths(cfg: dict) -> dict:
    """Remap every dataset root/split_file in a config onto the Volume."""
    data = cfg.get("data", {})
    for ds in data.get("datasets") or []:
        ds["root"] = _remap_path(ds.get("root"))
        ds["split_file"] = _remap_path(ds.get("split_file"))
    if data.get("root"):
        data["root"] = _remap_path(data["root"])
    if data.get("split_file"):
        data["split_file"] = _remap_path(data["split_file"])
    return cfg


def _dataset_roots(cfg: dict) -> list[str]:
    data = cfg.get("data", {})
    if data.get("datasets"):
        return [ds["root"] for ds in data["datasets"]]
    if data.get("root"):
        return [data["root"]]
    return []


@app.function(
    gpu=DEFAULT_GPU,
    volumes={VOLUME_ROOT: data_volume, CKPT_ROOT: ckpt_volume},
    secrets=[modal.Secret.from_name("wandb-credentials")],
    timeout=86400,  # 24h (Modal max); resume across runs via <ckpt>/last.ckpt
    retries=0,
)
def train(config_json: str, gpu_type: str = DEFAULT_GPU):
    """Run training with the given config (as JSON, converted to YAML remotely)."""
    import json
    import logging
    import subprocess
    import sys

    import yaml

    log = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)

    cfg = json.loads(config_json)

    config_path = Path("/tmp/config.yaml")
    with config_path.open("w") as f:
        yaml.dump(cfg, f, default_flow_style=False)

    log.info("GPU: %s", gpu_type)
    log.info("Checkpoint dir: %s", cfg.get("ckpt_path"))

    # Verify each dataset filelist exists on the Volume and log sample counts.
    total = 0
    for root in _dataset_roots(cfg):
        fp = Path(root)
        if not fp.exists():
            raise FileNotFoundError(f"Filelist not found on Volume: {fp}")
        n = len(fp.read_text().strip().splitlines())
        total += n
        log.info("Dataset %s: %d samples", root, n)
    log.info("Total samples: %d", total)

    # Auto-resume: the entrypoint loads <ckpt_path>/last.ckpt if present.
    ckpt_dir = Path(cfg.get("ckpt_path", CKPT_ROOT))
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    last_ckpt = ckpt_dir / "last.ckpt"
    if last_ckpt.exists():
        log.info("Resuming from %s", last_ckpt)
    else:
        log.info("No checkpoint found, starting from scratch")

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
def main(config: str, gpu: str = DEFAULT_GPU):
    import json
    import logging
    import subprocess

    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger(__name__)

    # Load the YAML locally without importing electrai (yaml may not be in the
    # modal CLI env). Pass the path as argv to avoid shell/string interpolation.
    loaded = subprocess.run(
        [
            "python3",
            "-c",
            "import sys, yaml, json; "
            "print(json.dumps(yaml.safe_load(open(sys.argv[1]))))",
            config,
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    cfg = json.loads(loaded.stdout)

    # Remap della paths onto the Volume and pin checkpoints to the ckpt Volume,
    # namespaced by run so multiple experiments don't collide on last.ckpt.
    cfg = _remap_data_paths(cfg)
    run_name = cfg.get("run_name", "run")
    cfg["ckpt_path"] = f"{CKPT_ROOT}/{run_name}"

    config_json = json.dumps(cfg, indent=2)
    log.info("Config:\n%s", config_json)

    # Fire-and-forget so multi-day training survives any local CLI disconnect.
    # Pair with `modal run --detach`; monitor via `modal app logs <app-id>` or
    # the Modal web UI. See [[modal-long-running-detach-spawn]].
    fc = train.with_options(gpu=gpu).spawn(config_json=config_json, gpu_type=gpu)
    log.info("Spawned train FunctionCall id=%s on %s", fc.object_id, gpu)
    log.info("Monitor at https://modal.com/apps (look for electrai-train)")
