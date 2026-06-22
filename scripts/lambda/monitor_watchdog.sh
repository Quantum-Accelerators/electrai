#!/usr/bin/env bash
# Independent watchdog for the resident monitor — NO LLM, no cluster access.
# "Who watches the watcher": confirms the monitor service is up and that a tick
# actually SUCCEEDED recently (via the heartbeat file, which only a clean Claude
# run touches). Pings Slack if not. Run from a systemd timer or cron (~30 min).
#
# Catches the failure the monitor itself cannot report: expired Claude token,
# quota exhaustion, a wedged loop, or the EC2 box itself degrading.

set -uo pipefail

ENV_FILE="${MONITOR_ENV:-$HOME/.config/electrai-monitor/monitor.env}"
if [ -f "$ENV_FILE" ]; then set -a; . "$ENV_FILE"; set +a; fi

HEARTBEAT="${MONITOR_HEARTBEAT:-$HOME/.local/state/electrai-monitor/heartbeat}"
SERVICE="${MONITOR_SERVICE:-electrai-monitor.service}"
STALE_S="${WATCHDOG_STALE_S:-3600}"

problems=()

if command -v systemctl >/dev/null 2>&1; then
  systemctl is-active --quiet "$SERVICE" || problems+=("service $SERVICE not active")
fi

if [ -f "$HEARTBEAT" ]; then
  age=$(( $(date +%s) - $(stat -c %Y "$HEARTBEAT") ))
  [ "$age" -gt "$STALE_S" ] && problems+=("last successful tick ${age}s ago (> ${STALE_S}s) — Claude may be de-authed / out of quota / wedged")
else
  problems+=("heartbeat $HEARTBEAT missing — monitor has never completed a clean tick")
fi

[ ${#problems[@]} -eq 0 ] && exit 0

msg=":rotating_light: electrai-monitor watchdog on $(hostname): $(printf '%s; ' "${problems[@]}")"
if [ -n "${SLACK_WEBHOOK_URL:-}" ]; then
  curl -fsS -X POST -H 'Content-type: application/json' \
    --data "$(printf '{"text":%s}' "\"${msg//\"/\\\"}\"")" \
    "$SLACK_WEBHOOK_URL" >/dev/null 2>&1 || true
fi
echo "$(date -Iseconds) WATCHDOG: $msg" >&2
exit 1
