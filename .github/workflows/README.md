# GitHub Actions Workflows

## gpu-e2e.yml - GPU E2E Training Test

Verifies deterministic training on both GPU and CPU.

**Trigger:** Manual (`workflow_dispatch`)

**Runs on:** EC2 GPU instance (via [ec2-gha])

**What it does:**
1. Runs GPU training, checks against `linux-gpu` expected values
2. Runs CPU baseline, checks against `linux` expected values

**Inputs:**
- `instance_type`: EC2 instance type (default: `g6.xlarge`)
- `epochs`: Training epochs (default: `5`)
- `update_expected`: Generate new expected values instead of checking (default: `false`)
- `debug`: Debug mode for SSH access (default: `false`)

**First run / after changes:**
```bash
# Generate expected values
gh workflow run gpu-e2e.yml -f update_expected=true

# Download and commit
gh run download <run-id> -n expected-linux-gpu
cp expected_values.json tests/
```

## gpu-benchmark.yml - GPU Benchmark

Compares GPU vs CPU training time with production-size model.

**Trigger:** Manual (`workflow_dispatch`)

**Runs on:** EC2 GPU instance (via [ec2-gha])

**What it does:**
1. Runs training on GPU, measures time
2. Runs training on CPU, measures time
3. Reports speedup in workflow summary

**Inputs:**
- `instance_type`: EC2 instance type (default: `g6.xlarge`)
- `epochs`: Training epochs (default: `5`)
- `channels`: Model channels - 8=tiny, 32=prod, 64=large (default: `32`)
- `residual_blocks`: Residual blocks - 2=tiny, 16=prod (default: `16`)
- `debug`: Debug mode (default: `false`)

**Note:** Currently uses the 5-sample test dataset. For meaningful GPU speedup, would need larger dataset (e.g., S3 data).

## gen-expected.yml - Generate Expected Values

Generates platform-specific expected values for deterministic e2e tests.

**Trigger:** Manual (`workflow_dispatch`) or push to `gen-expected` branch

**Runs on:** `macos-latest` (arm64) and `ubuntu-latest`

**What it does:**
1. Runs e2e test with `--update-expected` on both platforms
2. Merges results into single `expected_values.json`
3. Uploads as artifact (download with `gh run download <run-id> -n expected-values`)

**Platform expected values:**
- `darwin-arm64`: macOS Apple Silicon (M1 on GHA, tolerant of M2/M3/M4 locally)
- `linux`: Linux x86_64, CPU
- `linux-gpu`: Linux x86_64, CUDA GPU (generated via gpu-e2e.yml)

To update expected values:
```bash
# Run workflow
gh workflow run gen-expected.yml

# Download artifact after completion
gh run download <run-id> -n expected-values
cp expected_values.json tests/
```

## Required Secrets & Variables

For EC2 workflows (`gpu-e2e.yml`, `gpu-benchmark.yml`):
- **Secret:** `GH_SA_TOKEN` - GitHub PAT for runner registration
- IAM role trust configured in [Open-Athena/ops] for OIDC authentication

[ec2-gha]: https://github.com/Open-Athena/ec2-gha
[Open-Athena/ops]: https://github.com/Open-Athena/ops
