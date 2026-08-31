#!/usr/bin/env bash
# Multi-host wrapper around monitor_status.sh. Prints a labeled snapshot for
# every node in HOSTS so the EC2 monitor agent can evaluate head + worker in a
# single read.
#
# Usage:
#   bash monitor_status_all.sh                                  # HOSTS=lambda2, default run
#   HOSTS="lambda3-a lambda3-b" bash monitor_status_all.sh full_gga_gga+u_f32_16x
#
# Note on multi-node: only the HEAD node holds NFS/checkpoint/wandb-sync state.
# The worker's snapshot will legitimately show NFS/checkpoint errors — judge the
# worker on its `=== tmux ===` and `=== GPU util ===` sections only.

set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN="${1:-full_gga_gga+u_f32}"
HOSTS="${HOSTS:-lambda2}"

for h in $HOSTS; do
  echo "################## NODE: $h ##################"
  bash "$HERE/monitor_status.sh" "$h" "$RUN" || echo "(monitor_status.sh exited rc=$? for $h)"
  echo
done
