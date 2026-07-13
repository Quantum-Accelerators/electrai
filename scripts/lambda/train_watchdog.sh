#!/usr/bin/env bash
# One-shot training watchdog for the Lambda W64 run. Run from cron (~10 min).
#
# Catches the failure that silently burned ~2 days on 2026-07-13: training went
# to NaN but kept "running" (GPUs busy, session up), and save_top_k masked it so
# the best-checkpoint list looked frozen rather than alarming.
#
# Detects, on each tick:
#   NaN    - sustained nan train_loss (2 strikes, ~20 min) -> ALERT + stop run
#   STALL  - train.log not updated in STALL_S while session up -> ALERT
#   DOWN   - tmux session gone -> ALERT
#
# Emits: ALERT.txt in the checkpoint dir (mirrored to S3 by the backup loop),
# a line in ~/train_watchdog.log, and a Slack post if SLACK_WEBHOOK_URL is set.
# De-dups Slack so an unchanged condition doesn't ping every tick.
#
# Install:  (echo '*/10 * * * * bash $HOME/electrai/scripts/lambda/train_watchdog.sh') | crontab -
set -uo pipefail

ENV_FILE="${WATCHDOG_ENV:-$HOME/.config/electrai-monitor/monitor.env}"
[ -f "$ENV_FILE" ] && { set -a; . "$ENV_FILE"; set +a; }

SESSION="${WATCHDOG_SESSION:-electrai-train}"
NFS_ROOT="${NFS_ROOT:-$(ls -d /lambda/nfs/* 2>/dev/null | head -1)}"
CKPT_ROOT="${CKPT_ROOT:-$NFS_ROOT/checkpoints}"
LOG="${WATCHDOG_LOG:-$CKPT_ROOT/train.log}"
STALL_S="${WATCHDOG_STALL_S:-2700}"              # 45 min without a log write
NAN_STRIKES_MAX="${WATCHDOG_NAN_STRIKES:-2}"     # consecutive nan ticks before acting
STOP_ON_NAN="${WATCHDOG_STOP_ON_NAN:-1}"
ALERT_FILE="$CKPT_ROOT/ALERT.txt"
WLOG="$HOME/train_watchdog.log"
STATE_DIR="${WATCHDOG_STATE:-$HOME/.local/state/electrai-watchdog}"
mkdir -p "$STATE_DIR"
STRIKE_FILE="$STATE_DIR/nan_strikes"
LASTMSG_FILE="$STATE_DIR/last_alert"

ts() { date -Iseconds; }

alert() {
  local kind="$1"; shift
  local msg="[$kind] $*"
  printf '%s %s\n' "$(ts)" "$msg" | tee -a "$WLOG" > "$ALERT_FILE"
  # Slack, de-duplicated on (kind+msg)
  if [ -n "${SLACK_WEBHOOK_URL:-}" ] && [ "$msg" != "$(cat "$LASTMSG_FILE" 2>/dev/null)" ]; then
    local safe; safe=$(printf 'electrai W64 watchdog (%s): %s' "$(hostname)" "$msg" | tr -d '"\n\r' | cut -c1-400)
    curl -fsS -X POST -H 'Content-type: application/json' \
      --data "{\"text\":\":rotating_light: $safe\"}" "$SLACK_WEBHOOK_URL" >/dev/null 2>&1 || true
  fi
  printf '%s' "$msg" > "$LASTMSG_FILE"
}

ok() { printf '%s OK %s\n' "$(ts)" "$*" >> "$WLOG"; rm -f "$ALERT_FILE"; : > "$LASTMSG_FILE"; }

# --- DOWN: session gone ---
if ! tmux has-session -t "$SESSION" 2>/dev/null; then
  alert DOWN "tmux session '$SESSION' is gone -- training not running"
  exit 2
fi

# --- STALL: log not written recently ---
if [ -f "$LOG" ]; then
  age=$(( $(date +%s) - $(stat -c %Y "$LOG") ))
  if [ "$age" -gt "$STALL_S" ]; then
    alert STALL "train.log not updated in ${age}s (> ${STALL_S}s) -- stalled/hung"
    exit 3
  fi
else
  alert DOWN "train.log missing at $LOG"
  exit 2
fi

# --- NaN: sustained nan in the newest run's recent train_loss values ---
start=$(grep -n "starting training" "$LOG" 2>/dev/null | tail -1 | cut -d: -f1); start="${start:-1}"
recent=$(tail -n +"$start" "$LOG" 2>/dev/null | tr '\r' '\n' | grep -oE 'train_loss_step=[0-9na.]+' | tail -30)
last=$(printf '%s' "$recent" | tail -1)
nan_n=$(printf '%s\n' "$recent" | grep -c 'nan')

# Real divergence: latest value nan AND >=25/30 recent are nan (startup shows
# nan only briefly before numeric losses appear, so it won't hold across ticks).
if [ -n "$last" ] && printf '%s' "$last" | grep -q 'nan' && [ "$nan_n" -ge 25 ]; then
  strikes=$(( $(cat "$STRIKE_FILE" 2>/dev/null || echo 0) + 1 ))
  echo "$strikes" > "$STRIKE_FILE"
  printf '%s NaN-strike %s/%s (nan %s/30, last=%s)\n' "$(ts)" "$strikes" "$NAN_STRIKES_MAX" "$nan_n" "$last" >> "$WLOG"
  if [ "$strikes" -ge "$NAN_STRIKES_MAX" ]; then
    if [ "$STOP_ON_NAN" = "1" ]; then
      tmux kill-session -t "$SESSION" 2>/dev/null
      pkill -9 -f "[v]env/bin/python3 -m electrai" 2>/dev/null
      alert NAN-STOPPED "Sustained NaN over ${strikes} ticks -- STOPPED the run to stop compute waste. Recover: relaunch from newest good ckpt_epoch=* (NOT last.ckpt)."
    else
      alert NAN "Sustained NaN over ${strikes} ticks -- training is producing NaN."
    fi
    exit 4
  fi
  exit 0
fi

# healthy
echo 0 > "$STRIKE_FILE"
ok "last=${last:-<none>} nan=${nan_n}/30 logage=${age}s"
exit 0
