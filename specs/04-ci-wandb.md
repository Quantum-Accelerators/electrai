# GHA CI → WandB Dashboard

## Overview

A GitHub Actions workflow that runs significant training jobs on a schedule (e.g. weekly), logs metrics to **Weights & Biases**, and tracks regressions in val loss, wallclock time, and GPU utilization.

## Goals

1. **Automated regression detection**: catch model quality regressions before they reach production
2. **Performance tracking**: wallclock time and GPU memory trends over time
3. **Reproducibility**: every CI training run is logged and comparable
4. **Dashboard**: WandB project page shows trends, team can inspect any run

## Current State

Two GPU workflows exist:
- `gpu-e2e.yml`: deterministic test (5 repo samples, 5 epochs, checks exact loss values)
- `gpu-benchmark.yml`: configurable benchmark (S3 data, variable model size, optional CPU comparison)

Neither currently logs to WandB.

## Architecture

### WandB Integration

The existing codebase already supports WandB via `wandb_mode` config:
```yaml
wandb_mode: "online"   # or "offline", "disabled"
```

For CI, we need:
- `WANDB_API_KEY` secret in GitHub
- `WANDB_PROJECT` set to a CI-specific project (e.g. `elf-net-ci`)
- Tags: `ci`, `weekly`, git SHA, branch name
- Config logged: model size, data size, instance type, epochs

### New Workflow: `gpu-weekly.yml`

```yaml
name: Weekly GPU Training
on:
  schedule:
    - cron: '0 6 * * 1'   # Monday 6am UTC
  workflow_dispatch:
    inputs:
      instance_type: ...
      s3_samples: ...
      epochs: ...

jobs:
  ec2:
    uses: Open-Athena/ec2-gha/.github/workflows/runner.yml@v2
    ...

  train:
    needs: ec2
    runs-on: ${{ needs.ec2.outputs.id }}
    steps:
      - # checkout, install, sync S3 data ...

      - name: Train with WandB logging
        env:
          WANDB_API_KEY: ${{ secrets.WANDB_API_KEY }}
          WANDB_PROJECT: elf-net-ci
        run: |
          uv run python src/electrai/entrypoints/main.py train \
            --config src/electrai/configs/ci/weekly.yaml

      - name: Check regression
        run: |
          # Compare final val_loss against historical baseline
          # Fail workflow if regression detected
```

### Metrics to Track

| Metric | Source | Regression = |
|--------|--------|-------------|
| `val_loss` (NMAE) | Training | Increase > 5% from baseline |
| `train_loss` | Training | Increase > 10% from baseline |
| Wallclock time | `time` wrapper | Increase > 20% (hardware-dependent) |
| Peak GPU VRAM | `nvidia-smi` | Increase (memory leak indicator) |
| GPU utilization | `nvidia-smi --query-gpu` | Decrease (data loading bottleneck?) |

### VRAM Budget & Sample Filtering

The 50-sample benchmark OOMed because some samples have grids too large for L4 (22GB VRAM). Need a strategy:

- **Option A**: Filter samples by file size (proxy for grid size). `< 20MB` fits comfortably on L4.
- **Option B**: Use a larger GPU (A100 40GB via Lambda Labs or EC2 `p4d.24xlarge`).
- **Option C**: Skip samples that OOM, log which ones were skipped.

For weekly CI, **Option A** is simplest and most reliable.

## Tasks

- [ ] Add `WANDB_API_KEY` to GitHub repo/org secrets
- [ ] Create `src/electrai/configs/ci/weekly.yaml` with CI-appropriate settings
- [ ] Create `gpu-weekly.yml` workflow
- [ ] Add WandB tags/config logging to training script
- [ ] Implement regression check step (compare against baseline)
- [ ] Add sample size filtering (skip samples > N MB) to `e2e_train.py`
- [ ] Set up WandB project page with dashboard panels
- [ ] Add Slack notification on regression (optional, via GHA Slack action)

## Open Questions

- What's the right training config for weekly CI? (channels, blocks, epochs, data size)
- Should we use the PrinceOA WandB org or create a new one for CI?
- How to handle baseline updates? Manual, or auto-update on `main` merges?
- Should weekly runs use Lambda Labs (cheaper, but less reliable availability) or EC2?
