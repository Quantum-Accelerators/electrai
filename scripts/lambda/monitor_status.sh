#!/usr/bin/env bash
# One-shot status snapshot for an in-flight Lambda training run.
#
# Usage:
#   bash monitor_status.sh [host=lambda2] [run_name=full_gga_gga+u_f32]
#
# Designed to pair with the hourly LLM-driven monitor (see MONITOR.md). The
# output is intentionally machine-greppable plus human-readable.

set -uo pipefail

HOST="${1:-lambda2}"
RUN="${2:-full_gga_gga+u_f32}"
SESSION="${SESSION:-electrai-train}"

ssh "$HOST" "
  NFS_ROOT=\"\${NFS_ROOT:-\$(ls -d /lambda/nfs/* 2>/dev/null | head -1)}\"
  CKPT_DIR=\"\$NFS_ROOT/checkpoints\"
  RUN_DIR=\"\$CKPT_DIR/$RUN\"

  echo '=== tmux ==='
  tmux ls 2>&1 | head -5

  echo '=== now ==='
  date -Iseconds

  echo '=== mtimes (train.log within 60s of now, last.ckpt within 8h = live) ==='
  stat -c '%y %n' \"\$CKPT_DIR/train.log\" \"\$RUN_DIR/last.ckpt\" 2>&1

  echo '=== GPU util (all >= 80% = active) ==='
  nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader

  echo '=== current step (live tmux pane) ==='
  tmux capture-pane -t $SESSION:train -p -S -100 2>&1 \\
    | tr '\\r' '\\n' \\
    | grep -oE 'Epoch [0-9]+: *[0-9]+%.*it/s.*' | tail -2

  echo '=== checkpoints ==='
  ls -la \"\$RUN_DIR/\" 2>&1 | tail -8

  echo '=== backup log (S3 ckpt mirror, 10-min cadence) ==='
  tail -3 \"\$CKPT_DIR/backup.log\" 2>&1

  echo '=== wandb-sync log (10-min cadence; expect Syncing: ... done.) ==='
  tail -8 \"\$CKPT_DIR/wandb-sync.log\" 2>&1
"
