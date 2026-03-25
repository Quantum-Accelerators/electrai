# Modal GPU CI for electrai

## Context

electrai currently runs GPU e2e tests on EC2 via `ec2-gha` (`gpu-e2e.yml`). This spec adds a parallel Modal-based GPU CI workflow, following the helico pattern: run tests directly inside a Modal `@app.function` rather than provisioning a self-hosted runner.

## Approach

Like helico's `modal/ci.py`: define a Modal app with the project's deps baked into the image, copy source/tests in, run `e2e_train.py` inside the function. The GHA workflow just calls `modal run` from `ubuntu-latest`.

## Files to Create

### 1. `modal/ci.py`

```python
from pathlib import Path
import modal

ROOT = Path(__file__).parent.parent

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install(
        "torch>=2.9",
        "torchvision>=0.24",
        "lightning~=2.5",
        "numpy~=2.3",
        "scikit-learn>=1.7",
        "pymatgen>=2025.10",
        "pyyaml>=6.0",
        "zarr>=3.1",
        "hydra-core>=1.3",
        "wandb>=0.12",
        "click",
    )
    # Source code + tests + data (changes most, last layer)
    .add_local_dir(str(ROOT / "src"), remote_path="/root/electrai/src")
    .add_local_dir(str(ROOT / "tests"), remote_path="/root/electrai/tests")
    .add_local_dir(str(ROOT / "scripts"), remote_path="/root/electrai/scripts")
    .add_local_dir(str(ROOT / "data"), remote_path="/root/electrai/data")
    .add_local_file(
        str(ROOT / "pyproject.toml"), remote_path="/root/electrai/pyproject.toml"
    )
)

app = modal.App("electrai-ci", image=image)


@app.function(gpu="L4", timeout=600)
def run_e2e_test(epochs: int = 5, check: bool = True):
    """Run e2e training test on GPU."""
    import subprocess

    cmd = [
        "python",
        "scripts/e2e_train.py",
        "--gpu",
        "--verbose",
        "--epochs",
        str(epochs),
    ]
    if not check:
        cmd.append("--no-check")
    result = subprocess.run(cmd, cwd="/root/electrai", check=True)
    return result.returncode


@app.local_entrypoint()
def main(epochs: int = 5, check: bool = True):
    run_e2e_test.remote(epochs=epochs, check=check)
```

Notes:
- GPU type: `L4` matches current EC2 `g6.xlarge` (also L4). Could use fallback list `gpu=["L4", "A10G"]`.
- `data/MP/chgcars/` has the small test dataset (~5 samples) checked into the repo; bake it in.
- `scripts/e2e_train.py` handles GPU detection, deterministic seeding, and expected value checking internally.
- No `uv` needed inside the container — deps are pre-installed by `.pip_install()`.
- `PYTHONPATH` needs to include `src/` for the `electrai` package. Either `pip install -e .` inside the function, or set env. The simplest: add `.run_commands("cd /root/electrai && pip install -e .")` as the last image layer.

### 2. `.github/workflows/gpu-e2e-modal.yml`

```yaml
name: GPU E2E (Modal)

on:
  pull_request:
    branches: [main]
  workflow_dispatch:
    inputs:
      epochs:
        description: 'Number of training epochs'
        default: '5'
        type: string

jobs:
  test:
    runs-on: ubuntu-latest
    env:
      MODAL_TOKEN_ID: ${{ secrets.MODAL_TOKEN_ID }}
      MODAL_TOKEN_SECRET: ${{ secrets.MODAL_TOKEN_SECRET }}
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v5
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - run: uv pip install --system modal
      - run: modal run modal/ci.py --epochs ${{ inputs.epochs || '5' }}
```

## What this does NOT replicate from `gpu-e2e.yml`

- **CPU baseline test** — could add a second `@app.function(gpu=None)` call, but CPU tests already run in `gen-expected.yml`
- **`update_expected` mode** — would need to get the file back out of Modal (print to stdout and capture, or use a Modal Volume). Defer for now.
- **Artifact upload** — same challenge; not needed for the basic CI pass/fail

## Training on Modal (`modal/train.py`)

Full training entrypoint for running real experiments on Modal, replacing Lambda Labs.

### Data: `electrai-data` Volume

`dataset_4` (2,885 samples, ~205 GiB) synced from Della (Globus source of truth) → S3 → Modal Volume:
- S3: `s3://openathena/electrai/mp/chg_datasets/dataset_4/`
- Volume mount: `/data/mp/chg_datasets/dataset_4/{data,label}/`

Populate script: `modal/populate_volume.py` (S3 → Volume via `boto3`).

### Checkpoints: `electrai-checkpoints` Volume

Persists across runs. Mounted at `/checkpoints`.

### Usage

```bash
# Default: ResUNet, dataset_4, 50 epochs, L4
modal run modal/train.py

# A100, custom hyperparams
modal run modal/train.py --gpu A100 --channels 64 --epochs 50

# Use existing config file
modal run modal/train.py --config path/to/config.yaml --gpu A100
```

### Data provenance

```
Globus (ROSENGROUP share)
  └── /mp/chg_datasets/dataset_4/   (canonical, on Della)
        ├── Della → S3 (aws s3 sync, one-time)
        │     └── s3://openathena/electrai/mp/chg_datasets/dataset_4/
        │           └── S3 → Modal Volume (modal/populate_volume.py)
        └── Della → Lambda LLFS (Globus transfer, Betsy's prior setup)
              └── /home/ubuntu/betsy-electrai-2/dataset2/
```

## Secrets required

- `MODAL_TOKEN_ID` / `MODAL_TOKEN_SECRET` — repo secrets for GHA workflow
- `wandb-credentials` — Modal secret with `WANDB_API_KEY` (for training)
- `aws-credentials` — Modal secret with AWS creds (for `populate_volume.py`)

No `GH_SA_TOKEN` needed (no runner registration).

## Comparison: EC2 vs Modal for this workload

| | EC2 (`gpu-e2e.yml`) | Modal (`gpu-e2e-modal.yml`) |
|---|---|---|
| Setup time | ~3-5 min (instance boot) | ~30s (image cached) |
| GPU | L4 (g6.xlarge) | L4 |
| Deps install | `uv sync` on every run | Baked into image (cached) |
| Checkout | `actions/checkout` | `.add_local_dir()` |
| Artifacts | Native GHA | Not supported (print to stdout) |
| Cost | EC2 on-demand pricing | Modal per-second billing |
| Extra secrets | `GH_SA_TOKEN`, AWS OIDC | Modal tokens only |
