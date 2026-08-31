#!/bin/bash
# Stage the charge-density dataset from CAIOS object storage to node-local NVMe.
#
# Runs as the first step of an Iris task on the CoreWeave GB200 cluster.
# Idempotent: a marker file written after a successful sync lets preemption
# retries that land on a warm node (the cache dir is a hostPath that survives
# pod restarts) skip the download entirely. The dataset is immutable, so the
# marker is trusted unless STAGE_FORCE=1.
#
# Uses rclone: CAIOS requires virtual-host addressing for list operations,
# which s5cmd cannot emit (it is path-style only against custom endpoints).
#
# STAGE_ROOT lives under /uv/cache because that is the ONLY host-persistent
# mount Iris task pods get (hostPath /mnt/local/iris-cache/uv-cache). Writing
# anywhere else lands on the container overlay and counts against the pod's
# ephemeral-storage limit, which kills the pod mid-stage (exit 137).
#
# Env (all optional):
#   STAGE_BUCKET    source bucket                  [rhoarnet-us-east-08a]
#   STAGE_PREFIX    bucket prefix to mirror        [mp/chg_datasets]
#   STAGE_ROOT      local destination root         [/uv/cache/electrai]
#   STAGE_ENDPOINT  S3 endpoint                    [http://cwlota.com]
#   STAGE_FORCE     1 = re-sync even if marker present
#
# AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY must hold CAIOS credentials
# (passed to the Iris job with -e; they override the cluster-injected ones).
set -euo pipefail

STAGE_BUCKET=${STAGE_BUCKET:-rhoarnet-us-east-08a}
STAGE_PREFIX=${STAGE_PREFIX:-mp/chg_datasets}
STAGE_ROOT=${STAGE_ROOT:-/uv/cache/electrai}
STAGE_ENDPOINT=${STAGE_ENDPOINT:-http://cwlota.com}
DEST="$STAGE_ROOT/$STAGE_PREFIX"
MARKER="$DEST/.staged.ok"

if [[ -f "$MARKER" && "${STAGE_FORCE:-0}" != "1" ]]; then
    echo "stage_data: marker present ($(cat "$MARKER")), skipping sync"
    exit 0
fi

if ! command -v rclone >/dev/null 2>&1; then
    RCDIR=$(mktemp -d)
    case "$(uname -m)" in
        x86_64) RCARCH=amd64 ;;
        aarch64) RCARCH=arm64 ;;
        *)
            echo "stage_data: unsupported arch: $(uname -m)" >&2
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

mkdir -p "$DEST"
echo "stage_data: syncing s3://$STAGE_BUCKET/$STAGE_PREFIX -> $DEST"
start=$(date +%s)
rclone copy "cw:$STAGE_BUCKET/$STAGE_PREFIX" "$DEST" \
    --transfers 96 --checkers 128 --size-only --fast-list \
    --stats 60s --stats-one-line --log-level NOTICE

# RhoRead resolves data/ and label/ as siblings of the filelist, so the
# functionals dirs need the same symlink shim prep_data.sh created on Lambda.
ln -sfn ../../rho_gga/data "$DEST/functionals/gga/data"
ln -sfn ../../rho_gga/label "$DEST/functionals/gga/label"
ln -sfn "../../rho_gga+u/data" "$DEST/functionals/gga+u/data"
ln -sfn "../../rho_gga+u/label" "$DEST/functionals/gga+u/label"

n_files=$(find "$DEST" -type f | wc -l)
echo "stage_data: staged $n_files files in $(($(date +%s) - start))s"
date -u +"%Y-%m-%dT%H:%M:%SZ $n_files files" >"$MARKER"
