#!/usr/bin/env bash
# Generate GHA step summary for GPU benchmark results.
# Shared between EC2 (gpu-benchmark.yml) and Modal (gpu-benchmark-modal.yml).
#
# Required env vars:
#   PLATFORM         - "EC2 (g6.xlarge)" or "Modal (A100:2)" etc.
#   EPOCHS, CHANNELS, RESIDUAL_BLOCKS
#   DATA_DESC        - e.g. "50 samples (S3, ≤25MB)" or "dataset_4 (1000 samples)"
#
# Optional env vars:
#   VAL_LOSS, TRAIN_LOSS, WALLCLOCK - metrics (omitted if empty)
#   GPU_TIME, CPU_TIME              - for EC2 CPU comparison
#   WANDB_RUN_URL, WANDB_PROJECT, WANDB_ENTITY
#   GH_TOKEN                        - for PR link lookup

set -euo pipefail

: "${WANDB_ENTITY:=PrinceOA}"

echo "## GPU Benchmark Results" >> "$GITHUB_STEP_SUMMARY"
echo "" >> "$GITHUB_STEP_SUMMARY"

# ── Configuration table ──────────────────────────────────────────
echo "### Configuration" >> "$GITHUB_STEP_SUMMARY"
echo "| Parameter | Value |" >> "$GITHUB_STEP_SUMMARY"
echo "|-----------|-------|" >> "$GITHUB_STEP_SUMMARY"
echo "| Platform | $PLATFORM |" >> "$GITHUB_STEP_SUMMARY"
echo "| Epochs | $EPOCHS |" >> "$GITHUB_STEP_SUMMARY"
echo "| Channels | $CHANNELS |" >> "$GITHUB_STEP_SUMMARY"
echo "| Residual Blocks | $RESIDUAL_BLOCKS |" >> "$GITHUB_STEP_SUMMARY"
echo "| Data | $DATA_DESC |" >> "$GITHUB_STEP_SUMMARY"

# Commit link (with optional PR link)
COMMIT_LINK="[\`${GITHUB_SHA::7}\`](${GITHUB_SERVER_URL}/${GITHUB_REPOSITORY}/commit/${GITHUB_SHA})"
if [ -n "${GH_TOKEN:-}" ]; then
  PR_NUM=$(curl -sf -H "Authorization: token $GH_TOKEN" \
    "${GITHUB_API_URL}/repos/${GITHUB_REPOSITORY}/commits/${GITHUB_SHA}/pulls" \
    | python3 -c "import sys,json; ps=json.load(sys.stdin); print(ps[0]['number'] if ps else '')" 2>/dev/null || true)
  if [ -n "$PR_NUM" ]; then
    COMMIT_LINK="${COMMIT_LINK} ([#${PR_NUM}](${GITHUB_SERVER_URL}/${GITHUB_REPOSITORY}/pull/${PR_NUM}))"
  fi
fi
echo "| Commit | ${COMMIT_LINK} |" >> "$GITHUB_STEP_SUMMARY"

# WandB link
if [ -n "${WANDB_RUN_URL:-}" ]; then
  WANDB_RUN_ID="${WANDB_RUN_URL##*/}"
  echo "| WandB | [${WANDB_PROJECT}](https://wandb.ai/${WANDB_ENTITY}/${WANDB_PROJECT}) run [${WANDB_RUN_ID}](${WANDB_RUN_URL}) |" >> "$GITHUB_STEP_SUMMARY"
fi
echo "" >> "$GITHUB_STEP_SUMMARY"

# ── Results table ────────────────────────────────────────────────
echo "### Results" >> "$GITHUB_STEP_SUMMARY"

# EC2 style: GPU/CPU timing comparison
if [ -n "${GPU_TIME:-}" ]; then
  echo "| Device | Time | Speedup |" >> "$GITHUB_STEP_SUMMARY"
  echo "|--------|------|---------|" >> "$GITHUB_STEP_SUMMARY"
  if [ -n "${CPU_TIME:-}" ]; then
    SPEEDUP=$(echo "scale=2; $CPU_TIME / $GPU_TIME" | bc)
    echo "| GPU | ${GPU_TIME}s | **${SPEEDUP}x** |" >> "$GITHUB_STEP_SUMMARY"
    echo "| CPU | ${CPU_TIME}s | 1.0x |" >> "$GITHUB_STEP_SUMMARY"
  else
    echo "| GPU | ${GPU_TIME}s | - |" >> "$GITHUB_STEP_SUMMARY"
    echo "" >> "$GITHUB_STEP_SUMMARY"
    echo "*CPU comparison skipped. Run with \`compare_cpu=true\` to include.*" >> "$GITHUB_STEP_SUMMARY"
  fi
fi

# Modal style: val_loss / train_loss / wallclock
if [ -n "${VAL_LOSS:-}" ] || [ -n "${WALLCLOCK:-}" ]; then
  echo "| Metric | Value |" >> "$GITHUB_STEP_SUMMARY"
  echo "|--------|-------|" >> "$GITHUB_STEP_SUMMARY"
  [ -n "${VAL_LOSS:-}" ] && echo "| val_loss | $VAL_LOSS |" >> "$GITHUB_STEP_SUMMARY"
  [ -n "${TRAIN_LOSS:-}" ] && echo "| train_loss | $TRAIN_LOSS |" >> "$GITHUB_STEP_SUMMARY"
  [ -n "${WALLCLOCK:-}" ] && echo "| Wallclock | ${WALLCLOCK}s |" >> "$GITHUB_STEP_SUMMARY"
fi
