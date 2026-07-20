#!/bin/bash
# Submit a CoreWeave training job through Iris. Run from the LAPTOP, with the
# repo checkout you want bundled as the current directory (use a clean
# worktree: the bundle is a zip of cwd, and stray large files bloat it).
#
# Usage:
#   scripts/coreweave/submit.sh <config-path> <job-name> [extra iris args...]
# Example:
#   scripts/coreweave/submit.sh src/electrai/configs/MP/config_gga_gga+u_w128.yaml \
#       electrai-w128-gga-ggau
#
# Env overrides:
#   MARIN_REPO   marin checkout providing the iris client  [~/code/marin]
#   GPUS         iris --gpu spec                           [GB200x4]
#   CPUS/MEMORY/DISK                                       [64 / 200GB / 60GB]
#
# Credentials are read locally (CAIOS keys from the `coreweave` AWS profile,
# W&B key from ~/.netrc) and passed as job env vars; nothing is printed.
set -euo pipefail

CONFIG=${1:?usage: submit.sh <config> <job-name> [extra iris args...]}
JOB_NAME=${2:?usage: submit.sh <config> <job-name> [extra iris args...]}
shift 2

MARIN_REPO=${MARIN_REPO:-$HOME/code/marin}
GPUS=${GPUS:-GB200x4}
CPUS=${CPUS:-64}
MEMORY=${MEMORY:-200GB}
DISK=${DISK:-60GB}

[[ -f $CONFIG ]] || {
    echo "submit: config not found in cwd bundle: $CONFIG" >&2
    exit 1
}

KUBECONFIG=${KUBECONFIG:-$HOME/.kube/config-coreweave} \
    uv run --project "$MARIN_REPO" --package marin-iris \
    iris --cluster=cw-us-east-08a job run \
    --enable-extra-resources --gpu "$GPUS" --cpu "$CPUS" --memory "$MEMORY" --disk "$DISK" \
    --priority batch --max-retries 3 --job-name "$JOB_NAME" --no-wait \
    -e AWS_ACCESS_KEY_ID "$(aws configure get aws_access_key_id --profile coreweave)" \
    -e AWS_SECRET_ACCESS_KEY "$(aws configure get aws_secret_access_key --profile coreweave)" \
    -e WANDB_API_KEY "$(awk '/machine api.wandb.ai/{f=1} f && /password/{print $2; exit}' ~/.netrc)" \
    -e CONFIG "$CONFIG" \
    "$@" \
    -- bash -lc 'cd "${IRIS_WORKDIR:-/app}" && bash scripts/coreweave/run_training.sh'
