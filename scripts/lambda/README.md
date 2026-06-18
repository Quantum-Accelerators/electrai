# Lambda training runbook (`scripts/lambda/`)

End-to-end runbook for training `config_gga_gga+u_f32` on a Lambda Cloud GPU
instance (4× or 8× H100). Parallel to the Modal pipeline under `modal/` but
adapted for a reserved VM: data lives on local NVMe, training runs in tmux,
checkpoints back up to S3 hourly.

The four scripts are idempotent and can be re-run safely.

## Prerequisites

- Lambda H100 (or A100) instance, SSH'able as `ubuntu@…`.
- AWS credentials with **read** access to `s3://oa-electrai/mp/chg_datasets/*`
  (the `electrai-modal-reader` IAM user works; or a Lambda-specific read-only
  user in the same AWS account).
- Optional but recommended: AWS credentials with **write** to a checkpoint
  prefix like `s3://oa-electrai/checkpoints/lambda/` so training checkpoints
  back up off-instance.
- A `WANDB_API_KEY` for `PrinceOA` (any teammate's key works).

## Step 0 — Boot the instance and SSH in

```bash
ssh ubuntu@<instance-ip>      # or `ssh lambda` if you set up an SSH config
```

## Step 1 — One-time env setup

```bash
# inside the Lambda instance
curl -fsSL https://raw.githubusercontent.com/Quantum-Accelerators/electrai/betsy/gga-gga+u-f32/scripts/lambda/setup.sh | bash
#   or, if you've already cloned the repo:
# cd ~/electrai && bash scripts/lambda/setup.sh
```

The script installs `uv`, `aws`, and `tmux` (if missing), clones the repo at
the right branch, runs `uv sync`, and validates credentials. If AWS or wandb
isn't configured, it prints the exact next command.

After setup:
- `aws configure` (or export `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY`).
- `export WANDB_API_KEY='...'` in `~/.bashrc`.

## Step 2 — Sync data from S3 (~1–3 h)

```bash
bash ~/electrai/scripts/lambda/data_sync.sh
```

Pulls the packed `.zarr.zip` tree (~1.2 TiB, ~226K objects) from
`s3://oa-electrai/mp/chg_datasets/` to `~/data/mp/chg_datasets/`. Idempotent;
safe to re-run if interrupted.

## Step 3 — Wire data/label symlinks + smoke filelists

```bash
bash ~/electrai/scripts/lambda/prep_data.sh
```

Recreates the `functionals/{gga,gga+u}/{data,label}` symlinks into the
`rho_*` real dirs, writes `mp_filelist_smoke.txt` (first 200 ids) per
functional, and sanity-checks that the first id resolves to a real
`.zarr.zip`.

## Step 4 — Smoke training

```bash
bash ~/electrai/scripts/lambda/run_training.sh smoke
```

Launches under tmux session `electrai-train`. Validates the end-to-end
pipeline (DDP launches, ZipStore loads, wandb logs, checkpoints write) on
400 samples × 2 epochs in ~10 min. Throughput on Lambda's local NVMe is
the actual number we use to predict the full-run cost.

Watch:
```bash
tmux attach -t electrai-train
# or
tail -f ~/checkpoints/train.log
```

## Step 5 — Full training run

```bash
bash ~/electrai/scripts/lambda/run_training.sh full
```

Same tmux pattern, but the *full* config (113K samples × 100 epochs). The
training loop auto-resumes from `last.ckpt` if a previous run crashed, and a
sibling tmux window backs the checkpoint dir up to S3 every 10 min so even
a catastrophic instance loss only sets us back at most one window.

ETA on 4× H100, given the Modal smoke (1.28 it/s at H100:8 → ~5 samples/sec
on H100:4), is ~22 days for 100 epochs — over the 2-week ceiling. Plan for
either fewer epochs, more GPUs, or accepting a longer wall-clock. The
Lambda smoke result decides.

## Tunables (env vars)

| var | default | what |
|---|---|---|
| `REPO_DIR` | `~/electrai` | repo location |
| `DATA_ROOT` | `~/data` | local data root |
| `CKPT_ROOT` | `~/checkpoints` | local checkpoint dir |
| `BRANCH` | `betsy/gga-gga+u-f32` | branch to check out |
| `S3_BUCKET` | `oa-electrai` | S3 source bucket |
| `S3_PREFIX` | `mp/chg_datasets` | prefix on S3 |
| `S3_CKPT_BUCKET` | `oa-electrai` | S3 bucket for ckpt backups |
| `S3_CKPT_PREFIX` | `checkpoints/lambda` | ckpt backup prefix |
| `CKPT_BACKUP_S` | `600` | seconds between backups |
| `SMOKE_N` | `200` | ids per functional in smoke filelists |
| `TMUX_SESSION` | `electrai-train` | tmux session name |

## Recovery

If the instance dies mid-run:
1. Boot a fresh Lambda H100, run `setup.sh`.
2. **Skip data_sync.sh and prep_data.sh** if the data is already on this
   instance's NVMe (re-runs are idempotent but cost time). If on a new
   instance, run them.
3. Pull the latest checkpoint from S3:
   `aws s3 sync s3://oa-electrai/checkpoints/lambda/ ~/checkpoints/`
4. Re-run `run_training.sh full` — Lightning resumes from `last.ckpt`.
