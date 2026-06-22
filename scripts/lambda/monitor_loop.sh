#!/usr/bin/env bash
# Resident monitor loop for the EC2 watcher box. Each tick injects a small live
# CONTEXT header in front of monitor_agent_prompt.md and runs Claude headless;
# Claude's reply is appended to the journal (its durable memory across ticks).
#
# Continuity is by JOURNAL, not by conversation buffer: a 26-day run would blow
# any context window, and the journal survives restarts and token refreshes.
#
# Run under systemd (see systemd/electrai-monitor.service). Config + secrets come
# from the EnvironmentFile (see monitor.env.example).
#
# Heartbeat: only a *successful* tick touches $HEARTBEAT. The watchdog checks that
# file (not the journal) so a silently failing Claude (expired token, quota,
# network) is detected even though the journal header still updates each tick.

set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${REPO_DIR:-$HOME/electrai}"

# Optional env file (systemd already loads it; this covers manual runs).
ENV_FILE="${MONITOR_ENV:-$HOME/.config/electrai-monitor/monitor.env}"
if [ -f "$ENV_FILE" ]; then set -a; . "$ENV_FILE"; set +a; fi

HOSTS="${HOSTS:-lambda2}"
RUN="${RUN:-full_gga_gga+u_f32}"
INTERVAL="${MONITOR_INTERVAL:-900}"        # seconds between ticks (15 min)
JOURNAL="${MONITOR_JOURNAL:-$HOME/.local/state/electrai-monitor/journal.md}"
HEARTBEAT="${MONITOR_HEARTBEAT:-$HOME/.local/state/electrai-monitor/heartbeat}"
MAINT_FLAG="${MONITOR_MAINT_FLAG:-$HOME/.config/electrai-monitor/MAINTENANCE}"
PROMPT_FILE="${MONITOR_PROMPT:-$HERE/monitor_agent_prompt.md}"
SETTINGS="${MONITOR_SETTINGS:-$HERE/monitor_settings.json}"
MODEL="${MONITOR_MODEL:-claude-opus-4-8}"
CLAUDE_BIN="${CLAUDE_BIN:-claude}"

mkdir -p "$(dirname "$JOURNAL")" "$(dirname "$HEARTBEAT")"
cd "$REPO_DIR" || { echo "FATAL: REPO_DIR=$REPO_DIR missing" >&2; exit 1; }

echo "$(date -Iseconds) monitor_loop start: HOSTS='$HOSTS' RUN='$RUN' interval=${INTERVAL}s model=$MODEL" >> "$JOURNAL"

while true; do
  ts="$(date -Iseconds)"
  maint="no"; [ -f "$MAINT_FLAG" ] && maint="yes"

  ctx="CONTEXT (injected $ts):
- Nodes to check (HOSTS): $HOSTS
- Run name (RUN): $RUN
- Journal path (tail it for prior state): $JOURNAL
- Maintenance mode: $maint   (if yes: observe + journal only, do NOT remediate)
- Snapshot command for this tick:
    HOSTS=\"$HOSTS\" bash scripts/lambda/monitor_status_all.sh \"$RUN\"
- Slack webhook is in env as \$SLACK_WEBHOOK_URL (do not print it).
"
  prompt="${ctx}
$(cat "$PROMPT_FILE")"

  echo "" >> "$JOURNAL"
  echo "=== $ts tick (maint=$maint) ===" >> "$JOURNAL"

  if "$CLAUDE_BIN" -p "$prompt" \
        --model "$MODEL" \
        --settings "$SETTINGS" \
        --permission-mode default \
        --add-dir "$REPO_DIR" \
        >> "$JOURNAL" 2>&1; then
    date -Iseconds > "$HEARTBEAT"
  else
    rc=$?
    echo "$(date -Iseconds) ERROR: claude tick failed rc=$rc (auth? quota? network?) — heartbeat NOT updated" >> "$JOURNAL"
  fi

  sleep "$INTERVAL"
done
