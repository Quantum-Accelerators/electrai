#!/usr/bin/env bash
# Recreate the data/label symlinks and build smoke filelists in the local data
# tree. Lambda equivalent of modal/prep_volume.py.

set -euo pipefail

NFS_ROOT="${NFS_ROOT:-$(ls -d /lambda/nfs/* 2>/dev/null | head -1)}"
[ -z "$NFS_ROOT" ] && { echo "ERROR: no /lambda/nfs/* mount found; pass NFS_ROOT=... explicitly"; exit 1; }
DATA_ROOT="${DATA_ROOT:-$NFS_ROOT/data}"
BASE="$DATA_ROOT/mp/chg_datasets"
SMOKE_N="${SMOKE_N:-200}"

echo "=== relinking functionals/{gga,gga+u}/{data,label} -> rho_* ==="
for entry in "gga:rho_gga" "gga+u:rho_gga+u"; do
  func="${entry%:*}"
  rho="${entry#*:}"
  fdir="$BASE/functionals/$func"
  mkdir -p "$fdir"
  for sub in data label; do
    link="$fdir/$sub"
    target="../../$rho/$sub"
    real="$BASE/$rho/$sub"
    if [ ! -d "$real" ]; then
      echo "  ERROR: $real missing -- data_sync.sh hasn't completed"
      exit 1
    fi
    if [ -L "$link" ]; then rm "$link"; fi
    if [ -d "$link" ] && [ ! -L "$link" ]; then
      echo "  $link is a real dir; leaving as-is"
      continue
    fi
    ln -s "$target" "$link"
    echo "  linked $link -> $target"
  done
done

echo
echo "=== writing smoke filelists (first $SMOKE_N ids) ==="
for func in gga 'gga+u'; do
  fdir="$BASE/functionals/$func"
  fl="$fdir/mp_filelist.txt"
  smoke="$fdir/mp_filelist_smoke.txt"
  if [ ! -f "$fl" ]; then
    echo "  ERROR: $fl missing"
    exit 1
  fi
  head -n "$SMOKE_N" "$fl" > "$smoke"
  n=$(wc -l < "$smoke")
  echo "  wrote $smoke ($n ids)"
done

echo
echo "=== sanity check (first id of each filelist resolves) ==="
# Loader auto-detects either packed (.zarr.zip) or unpacked (.zarr/ dir).
# S3 holds unpacked; Modal Volume held packed; Lambda gets whichever S3 has.
for func in gga 'gga+u'; do
  fdir="$BASE/functionals/$func"
  first=$(head -n 1 "$fdir/mp_filelist.txt")
  zip="$fdir/data/$first.zarr.zip"
  store="$fdir/data/$first.zarr"
  if [ -f "$zip" ]; then
    fmt="packed (.zarr.zip)"; resolved="$zip"
  elif [ -d "$store" ]; then
    fmt="unpacked (.zarr/)"; resolved="$store"
  else
    echo "  ERROR: neither $zip nor $store resolves -- transfer incomplete"
    exit 1
  fi
  n=$(wc -l < "$fdir/mp_filelist.txt")
  echo "  OK ($func, $fmt): $resolved ($n total ids)"
done

echo
echo "prep_data complete. DATA_ROOT=$DATA_ROOT"
