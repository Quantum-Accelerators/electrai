#!/usr/bin/env bash
# Launch training in a detached tmux session. Auto-resume from last.ckpt; a
# parallel rclone-style aws sync backs checkpoints up to S3 every CKPT_BACKUP_S.
#
# Usage:
#   bash scripts/lambda/run_training.sh smoke   # subset + 2 epochs
#   bash scripts/lambda/run_training.sh full    # full 113K + 100 epochs

set -euo pipefail

MODE="${1:-smoke}"   # smoke | full
REPO_DIR="${REPO_DIR:-$HOME/electrai}"
DATA_ROOT="${DATA_ROOT:-$HOME/data}"
CKPT_ROOT="${CKPT_ROOT:-$HOME/checkpoints}"
S3_CKPT_BUCKET="${S3_CKPT_BUCKET:-oa-electrai}"
S3_CKPT_PREFIX="${S3_CKPT_PREFIX:-checkpoints/lambda}"
CKPT_BACKUP_S="${CKPT_BACKUP_S:-600}"   # 10 min
TMUX_SESSION="${TMUX_SESSION:-electrai-train}"

case "$MODE" in
  smoke) SRC_CFG="src/electrai/configs/MP/config_gga_gga+u_f32_smoke.yaml" ;;
  full)  SRC_CFG="src/electrai/configs/MP/config_gga_gga+u_f32.yaml" ;;
  *) echo "MODE must be 'smoke' or 'full'"; exit 1 ;;
esac

cd "$REPO_DIR"

# Pull WANDB_API_KEY out of ~/.bashrc if a non-interactive caller bypassed it.
if [ -z "${WANDB_API_KEY:-}" ] && [ -f "$HOME/.bashrc" ]; then
  set +u; source "$HOME/.bashrc"; set -u
fi
if [ -z "${WANDB_API_KEY:-}" ]; then
  echo "ERROR: WANDB_API_KEY env var not set."
  echo "  Add to ~/.bashrc:  export WANDB_API_KEY='...'  then re-run."
  exit 1
fi

mkdir -p "$CKPT_ROOT"

# Rewrite the config so the dataset paths and ckpt_path point at local NVMe.
RUNTIME_CFG="$CKPT_ROOT/runtime-config.yaml"
sed \
  -e 's|/scratch/gpfs/ROSENGROUP/common/globus_share_OA|'"$DATA_ROOT"'|g' \
  -e 's|^ckpt_path: .*|ckpt_path: '"$CKPT_ROOT"'/${MODE}_${RUN}|' \
  "$SRC_CFG" \
  | python3 -c "
import sys, re
text = sys.stdin.read()
# substitute MODE/RUN tokens from env
text = text.replace('\${MODE}', '$MODE')
# pull run_name from the config itself for the ckpt subdir
m = re.search(r'^run_name:\s*(.+)$', text, re.M)
run = (m.group(1).strip() if m else 'run')
text = text.replace('\${RUN}', run)
print(text)
" > "$RUNTIME_CFG"

# Sanity-print the rewritten paths
echo "=== runtime config datasets/ckpt ==="
grep -E '^\s*(root|split_file|ckpt_path):' "$RUNTIME_CFG"

# Kill any existing session of the same name
if tmux has-session -t "$TMUX_SESSION" 2>/dev/null; then
  echo "killing existing tmux session $TMUX_SESSION"
  tmux kill-session -t "$TMUX_SESSION"
fi

# Launch training + backup loop in one tmux session with two windows.
PYTHONPATH="$REPO_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONPATH

# Window 1: training. Re-exec on crash so a transient failure doesn't end
# the session; Lightning auto-resumes from last.ckpt on the next start.
TRAIN_CMD="cd '$REPO_DIR' && export WANDB_API_KEY='$WANDB_API_KEY' && \
  while true; do \
    echo \"\$(date -Iseconds) starting training\" | tee -a $CKPT_ROOT/train.log; \
    uv run python -m electrai.entrypoints.main train --config '$RUNTIME_CFG' 2>&1 | tee -a $CKPT_ROOT/train.log; \
    rc=\$?; \
    echo \"\$(date -Iseconds) training exited rc=\$rc\" | tee -a $CKPT_ROOT/train.log; \
    [ \$rc -eq 0 ] && break; \
    sleep 30; \
  done"

# Window 2: checkpoint backup loop, every CKPT_BACKUP_S seconds.
BACKUP_CMD="while true; do \
  echo \"\$(date -Iseconds) backing up checkpoints\" | tee -a $CKPT_ROOT/backup.log; \
  aws s3 sync '$CKPT_ROOT' 's3://$S3_CKPT_BUCKET/$S3_CKPT_PREFIX/' --only-show-errors 2>&1 | tee -a $CKPT_ROOT/backup.log; \
  sleep $CKPT_BACKUP_S; \
done"

tmux new-session -d -s "$TMUX_SESSION" -n train "$TRAIN_CMD"
tmux new-window  -t "$TMUX_SESSION" -n backup "$BACKUP_CMD"
tmux ls

cat <<EOF

Training launched as tmux session: $TMUX_SESSION
  attach:        tmux attach -t $TMUX_SESSION
  train log:     tail -f $CKPT_ROOT/train.log
  backup log:    tail -f $CKPT_ROOT/backup.log
  wandb:         https://wandb.ai/PrinceOA/mp-large-scale
  S3 ckpt sink:  s3://$S3_CKPT_BUCKET/$S3_CKPT_PREFIX/
EOF
