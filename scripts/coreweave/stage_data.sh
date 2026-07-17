#!/bin/bash
# Stage the charge-density dataset from CAIOS object storage to node-local NVMe.
#
# Runs as the first step of an Iris task on the CoreWeave GB200 cluster.
# Idempotent: a marker file written after a successful sync lets preemption
# retries that land on a warm node (the cache dir is a hostPath that survives
# pod restarts) skip the download entirely. The dataset is immutable, so the
# marker is trusted unless STAGE_FORCE=1.
#
# Env (all optional):
#   STAGE_BUCKET    source bucket                  [rhoarnet-us-east-08a]
#   STAGE_PREFIX    bucket prefix to mirror        [mp/chg_datasets]
#   STAGE_ROOT      local destination root         [/mnt/local/iris-cache/electrai]
#   STAGE_ENDPOINT  S3 endpoint                    [http://cwlota.com]
#   STAGE_FORCE     1 = re-sync even if marker present
#
# AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY must hold CAIOS credentials
# (passed to the Iris job with -e; they override the cluster-injected ones).
set -euo pipefail

STAGE_BUCKET=${STAGE_BUCKET:-rhoarnet-us-east-08a}
STAGE_PREFIX=${STAGE_PREFIX:-mp/chg_datasets}
STAGE_ROOT=${STAGE_ROOT:-/mnt/local/iris-cache/electrai}
STAGE_ENDPOINT=${STAGE_ENDPOINT:-http://cwlota.com}
DEST="$STAGE_ROOT/$STAGE_PREFIX"
MARKER="$DEST/.staged.ok"

if [[ -f "$MARKER" && "${STAGE_FORCE:-0}" != "1" ]]; then
    echo "stage_data: marker present ($(cat "$MARKER")), skipping sync"
    exit 0
fi

S5DIR=$(mktemp -d)
case "$(uname -m)" in
    x86_64) S5ARCH=Linux-64bit ;;
    aarch64) S5ARCH=Linux-arm64 ;;
    *)
        echo "stage_data: unsupported arch: $(uname -m)" >&2
        exit 1
        ;;
esac

python3 - "$S5ARCH" "$S5DIR" <<'PYEOF'
import io
import sys
import tarfile
import urllib.request

url = f"https://github.com/peak/s5cmd/releases/download/v2.3.0/s5cmd_2.3.0_{sys.argv[1]}.tar.gz"
tarfile.open(fileobj=io.BytesIO(urllib.request.urlopen(url).read()), mode="r:gz").extractall(sys.argv[2])
PYEOF
S5="$S5DIR/s5cmd"
chmod +x "$S5"

mkdir -p "$DEST"
echo "stage_data: syncing s3://$STAGE_BUCKET/$STAGE_PREFIX -> $DEST"
start=$(date +%s)
"$S5" --endpoint-url "$STAGE_ENDPOINT" --numworkers 512 --stat \
    sync "s3://$STAGE_BUCKET/$STAGE_PREFIX/*" "$DEST/"
# RhoRead resolves data/ and label/ as siblings of the filelist, so the
# functionals dirs need the same symlink shim prep_data.sh created on Lambda.
ln -sfn ../../rho_gga/data "$DEST/functionals/gga/data"
ln -sfn ../../rho_gga/label "$DEST/functionals/gga/label"
ln -sfn "../../rho_gga+u/data" "$DEST/functionals/gga+u/data"
ln -sfn "../../rho_gga+u/label" "$DEST/functionals/gga+u/label"

n_files=$(find "$DEST" -type f | wc -l)
echo "stage_data: staged $n_files files in $(($(date +%s) - start))s"
date -u +"%Y-%m-%dT%H:%M:%SZ $n_files files" >"$MARKER"
