#!/bin/bash
# Training wrapper for Iris jobs on the CoreWeave GB200 cluster.
#
# Sequence: install rclone -> stage dataset (idempotent) -> restore last.ckpt
# from CAIOS if the node has none -> start a background checkpoint sync loop
# -> torchrun. Lightning resumes from $CKPT_DIR/last.ckpt automatically when
# present, so preempted jobs continue from at most CKPT_SYNC_S + the 20-min
# checkpoint interval behind.
#
# Env (all optional):
#   CONFIG       training config                  [config_gga_gga+u_w96.yaml]
#   NPROC        GPUs on this node                [4]
#   CKPT_DIR     local checkpoint dir (must match ckpt_path in CONFIG)
#   CKPT_S3      CAIOS checkpoint prefix          [s3://rhoarnet-us-east-08a/checkpoints/gga_gga+u_w96]
#   CKPT_SYNC_S  sync interval seconds            [600]
#   STAGE_*      see stage_data.sh
#
# AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY must hold CAIOS credentials;
# WANDB_API_KEY is required for wandb_mode: online.
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$REPO_ROOT"

CONFIG=${CONFIG:-src/electrai/configs/MP/config_gga_gga+u_w96.yaml}
NPROC=${NPROC:-4}
# Default checkpoint locations derive from the config filename stem so that a
# different config (e.g. w128) can never accidentally resume another run's
# incompatible last.ckpt.
RUN_STEM=$(basename "$CONFIG" .yaml)
RUN_STEM=${RUN_STEM#config_}
CKPT_DIR=${CKPT_DIR:-/uv/cache/electrai/checkpoints/$RUN_STEM}
CKPT_S3=${CKPT_S3:-s3://rhoarnet-us-east-08a/checkpoints/$RUN_STEM}
CKPT_SYNC_S=${CKPT_SYNC_S:-600}
STAGE_ENDPOINT=${STAGE_ENDPOINT:-http://cwlota.com}
CKPT_REMOTE="cw:${CKPT_S3#s3://}"

if ! command -v rclone >/dev/null 2>&1; then
    RCDIR=$(mktemp -d)
    case "$(uname -m)" in
        x86_64) RCARCH=amd64 ;;
        aarch64) RCARCH=arm64 ;;
        *)
            echo "run_training: unsupported arch: $(uname -m)" >&2
            exit 1
            ;;
    esac
    python3 - "$RCARCH" "$RCDIR" <<'PYEOF'
import io
import sys
import urllib.request
import zipfile

url = f"https://downloads.rclone.org/rclone-current-linux-{sys.argv[1]}.zip"
zipfile.ZipFile(io.BytesIO(urllib.request.urlopen(url).read())).extractall(sys.argv[2])
PYEOF
    RCBIN=$(echo "$RCDIR"/rclone-*-linux-"$RCARCH")
    chmod +x "$RCBIN/rclone"
    export PATH="$RCBIN:$PATH"
fi

# CAIOS remote, configured via env: virtual-host addressing is mandatory
export RCLONE_CONFIG_CW_TYPE=s3
export RCLONE_CONFIG_CW_PROVIDER=Other
export RCLONE_CONFIG_CW_ENV_AUTH=true
export RCLONE_CONFIG_CW_ENDPOINT="$STAGE_ENDPOINT"
export RCLONE_CONFIG_CW_REGION=default
export RCLONE_CONFIG_CW_FORCE_PATH_STYLE=false

bash scripts/coreweave/stage_data.sh

# Restore for resume. Lightning versions the save_last file (last-v1.ckpt,
# last-v2.ckpt, ...) whenever an earlier run's last.ckpt already exists, but
# resume always reads last.ckpt — so fetch all last*.ckpt and promote the
# newest before launch. ALWAYS reconcile against the bucket with --update
# (copy only when the bucket file is newer): after several preemption
# bounces a retry can land on a node whose local last.ckpt is DAYS stale,
# and trusting it rolled a run back six epochs (2026-08-01, W160).
mkdir -p "$CKPT_DIR"
rclone copy "$CKPT_REMOTE" "$CKPT_DIR" --include 'last*.ckpt' --update --transfers 4 || true
NEWEST=$(ls -t "$CKPT_DIR"/last*.ckpt 2>/dev/null | head -1 || true)
if [[ -n "$NEWEST" && "$NEWEST" != "$CKPT_DIR/last.ckpt" ]]; then
    cp -f "$NEWEST" "$CKPT_DIR/.last_promote_tmp"
    mv -f "$CKPT_DIR/.last_promote_tmp" "$CKPT_DIR/last.ckpt"
    echo "run_training: promoted $(basename "$NEWEST") -> last.ckpt"
fi

# Guard: never resume from a checkpoint meaningfully older than the bucket's.
# A transient fetch failure above (swallowed by || true) can otherwise let a
# stale node copy win promotion — that silently rolled W128 back ~20 epochs
# on 2026-08-05. Failing hard makes the Iris retry refetch instead.
REMOTE_TS=$(rclone lsl "$CKPT_REMOTE" --include 'last.ckpt' 2>/dev/null | awk '{print $2" "$3}' | head -1)
if [[ -n "$REMOTE_TS" && -f "$CKPT_DIR/last.ckpt" ]]; then
    remote_s=$(date -d "$REMOTE_TS" +%s 2>/dev/null || echo 0)
    local_s=$(stat -c %Y "$CKPT_DIR/last.ckpt" 2>/dev/null || echo 0)
    if ((remote_s > 0 && local_s > 0 && remote_s - local_s > 600)); then
        echo "run_training: local last.ckpt is $((remote_s - local_s))s staler than bucket — refusing stale resume" >&2
        exit 1
    fi
fi
if [[ -f "$CKPT_DIR/last.ckpt" ]]; then
    echo "run_training: resuming from last.ckpt"
else
    echo "run_training: fresh start"
fi

ckpt_sync() {
    rclone copy "$CKPT_DIR" "$CKPT_REMOTE" --transfers 4 || true
}
# wandb runs offline (online init hits the viewer flags=null TypeError and a
# rank-0 crash deadlocks DDP); `wandb sync` uses a different API path and works
wandb_sync() {
    for d in "$REPO_ROOT"/wandb/offline-run-*; do
        [ -d "$d" ] && uv run --no-sync wandb sync "$d" >/dev/null 2>&1
    done
    return 0
}
(while true; do
    sleep "$CKPT_SYNC_S"
    ckpt_sync
    wandb_sync
done) &
SYNC_PID=$!

export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="$REPO_ROOT"

uv run --no-sync torchrun --standalone --nproc_per_node="$NPROC" \
    src/electrai/entrypoints/main.py train --config "$CONFIG" &
TRAIN_PID=$!
trap 'kill -TERM "$TRAIN_PID" 2>/dev/null || true' TERM INT

set +e
wait "$TRAIN_PID"
RC=$?
set -e

kill "$SYNC_PID" 2>/dev/null || true
ckpt_sync
wandb_sync
echo "run_training: exited rc=$RC (final checkpoint + wandb sync done)"
exit "$RC"
