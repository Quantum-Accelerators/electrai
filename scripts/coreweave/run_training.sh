#!/bin/bash
# Training wrapper for Iris jobs on the CoreWeave GB200 cluster.
#
# Sequence: install s5cmd -> stage dataset (idempotent) -> restore last.ckpt
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
CKPT_DIR=${CKPT_DIR:-/mnt/local/iris-cache/electrai/checkpoints/gga_gga+u_w96}
CKPT_S3=${CKPT_S3:-s3://rhoarnet-us-east-08a/checkpoints/gga_gga+u_w96}
CKPT_SYNC_S=${CKPT_SYNC_S:-600}
STAGE_ENDPOINT=${STAGE_ENDPOINT:-http://cwlota.com}

# s5cmd once, shared with stage_data.sh via PATH
S5ROOT=$(mktemp -d)
case "$(uname -m)" in
    x86_64) S5ARCH=Linux-64bit ;;
    aarch64) S5ARCH=Linux-arm64 ;;
    *)
        echo "run_training: unsupported arch: $(uname -m)" >&2
        exit 1
        ;;
esac
python3 - "$S5ARCH" "$S5ROOT" <<'PYEOF'
import io
import sys
import tarfile
import urllib.request

url = f"https://github.com/peak/s5cmd/releases/download/v2.3.0/s5cmd_2.3.0_{sys.argv[1]}.tar.gz"
tarfile.open(fileobj=io.BytesIO(urllib.request.urlopen(url).read()), mode="r:gz").extractall(sys.argv[2])
PYEOF
chmod +x "$S5ROOT/s5cmd"
export PATH="$S5ROOT:$PATH"

bash scripts/coreweave/stage_data.sh

# Restore for resume: only when this node has no local last.ckpt (a warm node's
# local copy is never older than the bucket's).
mkdir -p "$CKPT_DIR"
if [[ ! -f "$CKPT_DIR/last.ckpt" ]]; then
    if s5cmd --endpoint-url "$STAGE_ENDPOINT" cp "$CKPT_S3/last.ckpt" "$CKPT_DIR/last.ckpt"; then
        echo "run_training: restored last.ckpt from $CKPT_S3"
    else
        echo "run_training: no remote last.ckpt, fresh start"
    fi
fi

ckpt_sync() {
    s5cmd --endpoint-url "$STAGE_ENDPOINT" sync "$CKPT_DIR/" "$CKPT_S3/" || true
}
(while true; do
    sleep "$CKPT_SYNC_S"
    ckpt_sync
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
echo "run_training: exited rc=$RC (final checkpoint sync done)"
exit "$RC"
