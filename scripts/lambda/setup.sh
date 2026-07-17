#!/usr/bin/env bash
# One-time setup for a fresh Lambda 4x/8x H100 (or A100) instance.
# Idempotent; safe to re-run.

set -euo pipefail

REPO_URL="${REPO_URL:-git@github.com:Quantum-Accelerators/electrai.git}"
REPO_DIR="${REPO_DIR:-$HOME/electrai}"
BRANCH="${BRANCH:-betsy/gga-gga+u-f32}"

echo "=== installing uv ==="
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
uv --version

echo "=== aws cli ==="
if ! command -v aws >/dev/null 2>&1; then
  sudo apt-get update -qq
  sudo apt-get install -y -qq awscli
fi
aws --version

echo "=== tmux ==="
if ! command -v tmux >/dev/null 2>&1; then
  sudo apt-get install -y -qq tmux
fi
tmux -V

echo "=== repo ==="
if [ ! -d "$REPO_DIR/.git" ]; then
  git clone "$REPO_URL" "$REPO_DIR"
fi
cd "$REPO_DIR"
git fetch origin
git checkout "$BRANCH"
git pull --ff-only origin "$BRANCH"

echo "=== uv sync ==="
uv sync

echo "=== checking credentials ==="
if ! aws sts get-caller-identity >/dev/null 2>&1; then
  echo "AWS creds not configured. Run: aws configure"
  echo "  (need read access to s3://oa-electrai/mp/chg_datasets/*)"
  exit 1
fi
echo "AWS identity:"
aws sts get-caller-identity --output text --query 'Arn'

if [ -z "${WANDB_API_KEY:-}" ]; then
  echo "WARNING: WANDB_API_KEY env var not set."
  echo "  Add to ~/.bashrc:  export WANDB_API_KEY='...'"
  echo "  (Or pass inline before run_training.sh.)"
fi

echo
echo "=== setup OK ==="
echo "Repo at  : $REPO_DIR"
echo "Branch   : $(git rev-parse --abbrev-ref HEAD)"
echo "Commit   : $(git rev-parse --short HEAD)"
echo
echo "Next:"
echo "  bash scripts/lambda/data_sync.sh       # ~1-3h, 1.2 TiB from S3 to local NVMe"
echo "  bash scripts/lambda/prep_data.sh       # relink data/label dirs, build smoke filelists"
echo "  bash scripts/lambda/run_training.sh smoke      # validate end-to-end"
echo "  bash scripts/lambda/run_training.sh full       # full 100-epoch campaign"
