#!/usr/bin/env bash
# Launch 2-node (or more) DDP training across Lambda Cloud H100 instances.
# Drop-in companion to run_training.sh (single-node). The training entrypoint
# already reads LOCAL_WORLD_SIZE / WORLD_SIZE and derives num_nodes for the
# Lightning Trainer, so we just need torchrun to set the right env on both
# nodes and a NCCL setup that works over Lambda's 100 Gbps Ethernet (no IB).
#
# Usage (run on EACH node, simultaneously):
#   # head (rank 0):
#   NODE_RANK=0 MASTER_ADDR=<head-private-ip> \
#     bash scripts/lambda/run_training_multinode.sh full
#
#   # worker (rank 1):
#   NODE_RANK=1 MASTER_ADDR=<head-private-ip> \
#     bash scripts/lambda/run_training_multinode.sh full
#
# Required env:
#   NODE_RANK     0 for head, 1..NUM_NODES-1 for workers
#   MASTER_ADDR   IP of the head node, reachable from all workers (use the
#                 PRIVATE / internal interface IP, not the public NAT'd one)
#
# Optional env (with defaults):
#   NUM_NODES         2
#   NPROC_PER_NODE    8                              (1 rank per H100)
#   MASTER_PORT       29500
#   NCCL_SOCKET_IFNAME (autodetect via `ip route get $MASTER_ADDR`)
#   WANDB_MODE_OVERRIDE  (inherit pattern from run_training.sh; "offline"
#                         for online wandb, "disabled" for smokes)
#   See MULTINODE.md for the full operator runbook.

set -uo pipefail

MODE="${1:-smoke}"   # smoke | full

# -- required inputs -----------------------------------------------------------
: "${NODE_RANK:?NODE_RANK is required (0 = head, 1+ = worker)}"
: "${MASTER_ADDR:?MASTER_ADDR is required (head node private IP)}"

# -- defaults ------------------------------------------------------------------
NUM_NODES="${NUM_NODES:-2}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
MASTER_PORT="${MASTER_PORT:-29500}"

REPO_DIR="${REPO_DIR:-$HOME/electrai}"
NFS_ROOT="${NFS_ROOT:-$(ls -d /lambda/nfs/* 2>/dev/null | head -1)}"
[ -z "$NFS_ROOT" ] && { echo "ERROR: no /lambda/nfs/* mount found; pass NFS_ROOT=... explicitly"; exit 1; }
DATA_ROOT="${DATA_ROOT:-$NFS_ROOT/data}"
CKPT_ROOT="${CKPT_ROOT:-$NFS_ROOT/checkpoints}"
S3_CKPT_BUCKET="${S3_CKPT_BUCKET:-oa-electrai}"
S3_CKPT_PREFIX="${S3_CKPT_PREFIX:-checkpoints/lambda-multinode}"
CKPT_BACKUP_S="${CKPT_BACKUP_S:-600}"
TMUX_SESSION="${TMUX_SESSION:-electrai-train}"
UV_BIN="${UV_BIN:-$HOME/.local/bin/uv}"

# -- pick the right config -----------------------------------------------------
case "$MODE" in
  smoke) SRC_CFG="src/electrai/configs/MP/config_gga_gga+u_f32_smoke.yaml" ;;
  full)  SRC_CFG="src/electrai/configs/MP/config_gga_gga+u_f32.yaml" ;;
  *) echo "MODE must be 'smoke' or 'full'"; exit 1 ;;
esac

cd "$REPO_DIR"

# -- WANDB key (same trick as run_training.sh; Ubuntu's ~/.bashrc returns early
# for non-interactive shells, so we grep the export line directly) -------------
if [ -z "${WANDB_API_KEY:-}" ] && [ -f "$HOME/.bashrc" ]; then
  eval "$(grep -E '^[[:space:]]*export WANDB_API_KEY=' "$HOME/.bashrc" | tail -1)"
fi
if [ -z "${WANDB_API_KEY:-}" ]; then
  echo "ERROR: WANDB_API_KEY env var not set."
  echo "  Add to ~/.bashrc:  export WANDB_API_KEY='...'  then re-run."
  exit 1
fi

mkdir -p "$CKPT_ROOT"

# -- auto-detect the NIC that reaches MASTER_ADDR ------------------------------
# `ip route get <ip>` prints the interface used to reach that IP. On Lambda
# this is usually the second NIC (the 100 Gbps internal one) -- e.g. `enp...`
# or `ens...`. If autodetection picks the wrong NIC (e.g. the management
# interface), set NCCL_SOCKET_IFNAME explicitly.
if [ -z "${NCCL_SOCKET_IFNAME:-}" ]; then
  DETECTED_IF=$(ip -o route get "$MASTER_ADDR" 2>/dev/null | awk '{
    for (i=1;i<=NF;i++) if ($i=="dev") { print $(i+1); exit }
  }')
  if [ -n "$DETECTED_IF" ]; then
    NCCL_SOCKET_IFNAME="$DETECTED_IF"
    echo "NCCL_SOCKET_IFNAME auto-detected -> $NCCL_SOCKET_IFNAME (route to $MASTER_ADDR)"
  else
    echo "WARN: could not auto-detect NIC for $MASTER_ADDR; falling back to NCCL default."
    echo "  Set NCCL_SOCKET_IFNAME=<iface>  (see MULTINODE.md)."
  fi
fi

# -- rewrite the config so dataset paths and ckpt_path point at our local roots
# Same sed pipeline as run_training.sh; both nodes regenerate this independently
# (idempotent, same input -> same output). ------------------------------------
RUNTIME_CFG="$CKPT_ROOT/runtime-config.yaml"
WANDB_SED=()
if [ -n "${WANDB_MODE_OVERRIDE:-}" ]; then
  WANDB_SED=(-e 's|^wandb_mode: .*|wandb_mode: '"$WANDB_MODE_OVERRIDE"'|')
fi
sed \
  -e 's| /scratch/gpfs/ROSENGROUP/common/globus_share_OA/| '"$DATA_ROOT"'/|g' \
  -e 's| /data/mp/chg_datasets/| '"$DATA_ROOT"'/mp/chg_datasets/|g' \
  -e 's|^ckpt_path: .*|ckpt_path: '"$CKPT_ROOT"'/${MODE}_${RUN}|' \
  "${WANDB_SED[@]}" \
  "$SRC_CFG" \
  | python3 -c "
import sys, re
text = sys.stdin.read()
text = text.replace('\${MODE}', '$MODE')
m = re.search(r'^run_name:\s*(.+)$', text, re.M)
run = (m.group(1).strip() if m else 'run')
text = text.replace('\${RUN}', run)
print(text)
" > "$RUNTIME_CFG"

echo "=== runtime config datasets/ckpt ==="
grep -E '^\s*(root|split_file|ckpt_path):' "$RUNTIME_CFG"

# -- tmux ----------------------------------------------------------------------
if tmux has-session -t "$TMUX_SESSION" 2>/dev/null; then
  echo "killing existing tmux session $TMUX_SESSION"
  tmux kill-session -t "$TMUX_SESSION"
fi

PYTHONPATH="$REPO_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONPATH

# -- NCCL env  -----------------------------------------------------------------
# Why each var (all of these get passed through tmux into the training shell):
#   NCCL_IB_DISABLE=1      Lambda Ethernet-only fabric; force the socket
#                          transport so NCCL doesn't waste time probing IB.
#   NCCL_SOCKET_IFNAME=... Pin NCCL to the NIC that actually reaches
#                          MASTER_ADDR (auto-detected above). Without this,
#                          NCCL may pick docker0 / a vlan and hang.
#   NCCL_DEBUG=INFO        Keep this loud for the first few multi-node runs;
#                          drop to WARN once stable to cut log noise.
#   NCCL_ASYNC_ERROR_HANDLING=1
#                          Surface NCCL errors as Python exceptions instead
#                          of silent hangs. Strongly recommended on Ethernet.
#   NCCL_P2P_LEVEL=NVL     Intra-node: use NVLink only (H100 SXM has NVL).
#                          Inter-node always falls back to socket regardless.
NCCL_ENV=(
  "NCCL_IB_DISABLE=1"
  "NCCL_DEBUG=INFO"
  "NCCL_ASYNC_ERROR_HANDLING=1"
  "NCCL_P2P_LEVEL=NVL"
)
if [ -n "${NCCL_SOCKET_IFNAME:-}" ]; then
  NCCL_ENV+=("NCCL_SOCKET_IFNAME=$NCCL_SOCKET_IFNAME")
fi
NCCL_EXPORTS=$(printf 'export %s; ' "${NCCL_ENV[@]}")

# -- training command ----------------------------------------------------------
# Rendezvous backend choice:
#   We use torchrun's classic --master_addr / --master_port (a.k.a. "static"
#   rendezvous). For exactly 2 known-IP nodes this is simpler and friendlier
#   to firewall config than c10d, and matches the rest of the runbook's
#   "head node IP" mental model. Switch to `--rdzv_backend=c10d
#   --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT --rdzv_id=electrai` if/when we
#   want elastic restarts (e.g. workers can drop/rejoin) -- not needed today.
#
# torchrun sets LOCAL_RANK, RANK, WORLD_SIZE, LOCAL_WORLD_SIZE in each worker
# process. train.py already reads LOCAL_WORLD_SIZE / WORLD_SIZE to derive
# num_nodes for the Lightning Trainer.
#
# We wrap in a while-loop for the same crash-resume behaviour as the
# single-node script. Lightning auto-resumes from last.ckpt on restart.
TORCHRUN_CMD=(
  "$UV_BIN" run torchrun
  --nnodes="$NUM_NODES"
  --node_rank="$NODE_RANK"
  --nproc_per_node="$NPROC_PER_NODE"
  --master_addr="$MASTER_ADDR"
  --master_port="$MASTER_PORT"
  -m electrai.entrypoints.main train --config "$RUNTIME_CFG"
)
# expand to a single shell string for tmux's -d send-keys form
TORCHRUN_STR="${TORCHRUN_CMD[*]}"

TRAIN_CMD="set -o pipefail; cd '$REPO_DIR' && \
  export WANDB_API_KEY='$WANDB_API_KEY' && \
  $NCCL_EXPORTS \
  while true; do \
    echo \"\$(date -Iseconds) [node $NODE_RANK/$NUM_NODES] starting training\" | tee -a $CKPT_ROOT/train.log; \
    $TORCHRUN_STR 2>&1 | tee -a $CKPT_ROOT/train.log; \
    rc=\$?; \
    echo \"\$(date -Iseconds) [node $NODE_RANK/$NUM_NODES] training exited rc=\$rc\" | tee -a $CKPT_ROOT/train.log; \
    [ \$rc -eq 0 ] && break; \
    sleep 30; \
  done"

tmux new-session -d -s "$TMUX_SESSION" -n train "$TRAIN_CMD"

# -- head-node-only: ckpt backup + wandb sync windows --------------------------
# In Lightning DDP only global rank 0 writes checkpoints, and global rank 0
# is always on the head node (NODE_RANK=0). So workers don't need backup or
# wandb-sync windows.
if [ "$NODE_RANK" = "0" ]; then
  BACKUP_CMD="while true; do \
    echo \"\$(date -Iseconds) backing up checkpoints\" | tee -a $CKPT_ROOT/backup.log; \
    aws s3 sync '$CKPT_ROOT' 's3://$S3_CKPT_BUCKET/$S3_CKPT_PREFIX/' --only-show-errors 2>&1 | tee -a $CKPT_ROOT/backup.log; \
    sleep $CKPT_BACKUP_S; \
  done"
  tmux new-window -t "$TMUX_SESSION" -n backup "$BACKUP_CMD"

  # wandb-sync window: only useful when WANDB_MODE_OVERRIDE=offline (the
  # working pattern for PrinceOA right now -- online login crashes). For
  # smoke runs with WANDB_MODE_OVERRIDE=disabled there's nothing to sync,
  # but starting the window is cheap and idempotent.
  WANDB_SYNC_CMD="cd '$REPO_DIR' && export WANDB_API_KEY='$WANDB_API_KEY' && \
    while true; do \
      echo \"\$(date -Iseconds) wandb sync sweep\" | tee -a $CKPT_ROOT/wandb-sync.log; \
      find . -type d -name 'offline-run-*' -print 2>/dev/null \
        | xargs -r -n1 '$UV_BIN' run wandb sync 2>&1 \
        | tee -a $CKPT_ROOT/wandb-sync.log; \
      sleep 300; \
    done"
  tmux new-window -t "$TMUX_SESSION" -n wandb-sync "$WANDB_SYNC_CMD"
fi

tmux ls

cat <<EOF

Multi-node training launched (node $NODE_RANK of $NUM_NODES).
  tmux session : $TMUX_SESSION
  attach       : tmux attach -t $TMUX_SESSION
  train log    : tail -f $CKPT_ROOT/train.log
EOF

if [ "$NODE_RANK" = "0" ]; then
  cat <<EOF
  backup log   : tail -f $CKPT_ROOT/backup.log
  wandb-sync   : tail -f $CKPT_ROOT/wandb-sync.log
  S3 ckpt sink : s3://$S3_CKPT_BUCKET/$S3_CKPT_PREFIX/
EOF
fi

cat <<EOF

NCCL env on this node:
EOF
printf '  %s\n' "${NCCL_ENV[@]}"

cat <<EOF

If training hangs at "initializing process group", check:
  1. MASTER_ADDR ($MASTER_ADDR) is reachable from this node ($(hostname))
     (try:  nc -zv $MASTER_ADDR $MASTER_PORT)
  2. NCCL_SOCKET_IFNAME is the right NIC (see MULTINODE.md "diagnostics").
  3. Both nodes were started within ~60s of each other; static rendezvous
     does not wait forever for late joiners.
EOF
