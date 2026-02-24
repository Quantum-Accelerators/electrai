# Runner Abstraction: EC2 / Lambda Labs / Modal

## Overview

Abstract over multiple GPU cloud providers for GitHub Actions workflows. Currently using EC2 via `ec2-gha`; want to also support **Lambda Labs** (via `lambda-gha`) and **Modal** for flexibility, cost optimization, and availability.

## Motivation

- **EC2** (`ec2-gha`): reliable but expensive ($0.80+/hr for L4, $3+/hr for A100)
- **Lambda Labs**: OA has credits; ~free for us; cheaper per-GPU-hr; but availability can be spotty, and instance launch adds latency
- **Modal**: OA has credits; serverless GPU (no instance management); fast cold starts; pay-per-second
- No single provider always has the GPU type you need, when you need it

## Current State

- `ec2-gha` (`Open-Athena/ec2-gha`): mature, v2 release, used in `gpu-e2e.yml` and `gpu-benchmark.yml`
- `lambda-gha` (`Open-Athena/lambda-gha`): exists at `/Users/ryan/c/oa/lambda-gha`, partially built, has `runner.yml` reusable workflow
- Modal: no GHA integration yet

## Design Options

### Option 1: Provider-specific GHA workflows

Separate workflows per provider, dispatch the right one manually or via wrapper script:

```
.github/workflows/
  gpu-e2e-ec2.yml       # uses ec2-gha
  gpu-e2e-lambda.yml    # uses lambda-gha
  gpu-e2e-modal.yml     # uses modal
```

**Pro**: simple, no abstraction needed, each workflow is self-contained
**Con**: duplication of training steps across 3 files; maintenance burden

### Option 2: Parameterized single workflow

One workflow with a `provider` input:

```yaml
inputs:
  provider:
    description: 'GPU provider'
    default: 'ec2'
    type: choice
    options: [ec2, lambda, modal]
```

Then conditionally call different reusable workflows. Problem: GHA `uses:` can't be dynamic — you can't conditionally pick a reusable workflow at runtime.

**Verdict**: Not directly feasible in GHA YAML.

### Option 3: Wrapper CLI + thin GHA shims (recommended)

A Python CLI that abstracts provider selection, with thin GHA workflows that just call it:

```
src/electrai/ci/
  runner.py              # CLI: `runner launch --provider ec2 --gpu a100 ...`
  providers/
    ec2.py               # EC2 via boto3 / ec2-gha internals
    lambda_labs.py        # Lambda Labs API
    modal_runner.py       # Modal API
```

Each GHA workflow is a thin shim:
```yaml
# gpu-train.yml
jobs:
  train:
    runs-on: ubuntu-latest
    steps:
      - run: |
          uv run python -m electrai.ci.runner launch \
            --provider ${{ inputs.provider || 'ec2' }} \
            --instance-type ${{ inputs.instance_type }} \
            -- python tests/e2e_train.py --gpu ...
```

**Pro**: single source of truth for provider logic; CLI usable outside GHA too
**Con**: more upfront work; the runner CLI needs to handle instance lifecycle (launch → run → terminate)

### Option 4: Hybrid — keep ec2-gha/lambda-gha, add Modal

Keep the existing GHA reusable workflow pattern for EC2 and Lambda, add a Modal equivalent. Use a dispatch script (bash or python) to pick and trigger the right workflow:

```bash
# Local CLI
./scripts/gpu-run.sh ec2 gpu-e2e     # triggers gpu-e2e-ec2.yml
./scripts/gpu-run.sh lambda gpu-e2e  # triggers gpu-e2e-lambda.yml
./scripts/gpu-run.sh modal gpu-e2e   # triggers gpu-e2e-modal.yml
```

**Pro**: builds on existing infrastructure; each provider workflow is independent
**Con**: still some duplication in training steps (mitigated by shared composite actions)

## Lambda Labs Integration

`lambda-gha` already exists with a reusable `runner.yml` workflow. Key differences from `ec2-gha`:
- API key auth (not AWS OIDC)
- Instance types: `gpu_1x_a10`, `gpu_1x_a100_sxm4`, etc.
- SSH-based setup (no userdata/cloud-init like EC2)
- Availability can be limited; need retry/fallback logic

See `/Users/ryan/c/oa/lambda-gha/` for current implementation.

## Modal Integration

Modal is serverless — no instance management. A Modal "app" would:
1. Define a GPU function with dependencies
2. Run the training script inside it
3. Return results/artifacts

```python
import modal

app = modal.App("elf-net-ci")

@app.function(
    gpu="A100",
    image=modal.Image.from_registry("nvcr.io/nvidia/pytorch:24.12-py3")
        .pip_install("uv")
        .run_commands("cd /app && uv sync"),
    timeout=3600,
)
def train(config: dict):
    import subprocess
    subprocess.run(["uv", "run", "python", "tests/e2e_train.py", "--gpu", ...])
```

For GHA integration, the workflow just runs `modal run`:
```yaml
- run: modal run src/electrai/ci/modal_train.py --config ...
  env:
    MODAL_TOKEN_ID: ${{ secrets.MODAL_TOKEN_ID }}
    MODAL_TOKEN_SECRET: ${{ secrets.MODAL_TOKEN_SECRET }}
```

## Tasks

- [ ] Get `lambda-gha` to feature parity with `ec2-gha` (see its own roadmap)
- [ ] Create `gpu-e2e-lambda.yml` using `lambda-gha` reusable workflow
- [ ] Test Lambda Labs availability for `gpu_1x_a10` and `gpu_1x_a100_sxm4`
- [ ] Prototype Modal integration (`modal_train.py`)
- [ ] Add Modal secrets to GitHub
- [ ] Create dispatch script for provider selection
- [ ] Document GPU pricing comparison and when to use each provider
- [ ] Consider: automatic fallback (try Lambda → EC2 → Modal)?

## GPU Instance Mapping

| Provider | Instance | GPU | VRAM | $/hr |
|----------|----------|-----|------|------|
| EC2 | `g6.xlarge` | L4 | 24GB | ~$0.80 |
| EC2 | `g5.xlarge` | A10G | 24GB | ~$1.01 |
| EC2 | `p4d.24xlarge` | 8×A100 | 320GB | ~$32.77 |
| Lambda | `gpu_1x_a10` | A10 | 24GB | $0.75 |
| Lambda | `gpu_1x_a100_sxm4` | A100 | 80GB | $1.79 |
| Lambda | `gpu_1x_h100_sxm5` | H100 | 80GB | $2.49 |
| Modal | A10G | A10G | 24GB | ~$0.53 |
| Modal | A100 | A100 | 40/80GB | ~$1.10-2.78 |
| Modal | H100 | H100 | 80GB | ~$3.70 |

## Open Questions

- Is Option 3 (wrapper CLI) or Option 4 (parallel workflows + dispatch script) preferred?
- Should fallback be automatic or manual? (Auto-fallback adds complexity)
- Lambda Labs availability: how often are `a100_sxm4` instances available?
- Modal cold start time: acceptable for CI? (Usually 30-60s)
- Cost tracking: how to monitor spend across providers?
