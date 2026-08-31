#!/usr/bin/env bash
# Sync packed zarr training data from S3 to local NVMe.
# Idempotent (aws s3 sync skips files that already exist with matching size).

set -euo pipefail

S3_BUCKET="${S3_BUCKET:-oa-electrai}"
S3_PREFIX="${S3_PREFIX:-mp/chg_datasets}"
NFS_ROOT="${NFS_ROOT:-$(ls -d /lambda/nfs/* 2>/dev/null | head -1)}"
[ -z "$NFS_ROOT" ] && { echo "ERROR: no /lambda/nfs/* mount found; pass NFS_ROOT=... explicitly"; exit 1; }
DATA_ROOT="${DATA_ROOT:-$NFS_ROOT/data}"
DEST="$DATA_ROOT/$S3_PREFIX"

mkdir -p "$DEST"

echo "=== syncing s3://$S3_BUCKET/$S3_PREFIX/ -> $DEST ==="
echo "(this is ~1.2 TiB / ~226K .zarr.zip files; expect 1-3h on Lambda NVMe)"
echo

# --no-progress keeps the log quiet; --only-show-errors keeps it informative
# without thousands of "upload: ... -> ..." lines.
aws s3 sync \
  "s3://$S3_BUCKET/$S3_PREFIX/" "$DEST/" \
  --only-show-errors --no-progress

echo
echo "=== verifying ==="
# Count files in each major subtree
for sub in rho_gga rho_gga+u functionals; do
  count=$(find "$DEST/$sub" -type f 2>/dev/null | wc -l)
  size=$(du -sh "$DEST/$sub" 2>/dev/null | awk '{print $1}')
  printf "  %-15s %8s files  %8s\n" "$sub" "$count" "$size"
done

total=$(find "$DEST" -type f | wc -l)
total_sz=$(du -sh "$DEST" 2>/dev/null | awk '{print $1}')
echo "  --------------------------------------"
printf "  %-15s %8s files  %8s\n" "TOTAL" "$total" "$total_sz"

# Sanity: S3 holds the unpacked zarr layout (~678K files: ~226K stores ×
# ~3 inner files + standalone files). The .zarr.zip packed form is
# Modal-Volume-specific; not present on S3.
if [ "$total" -lt 670000 ]; then
  echo "WARN: file count looks low; sync may not be complete."
  exit 1
fi

echo "OK."
