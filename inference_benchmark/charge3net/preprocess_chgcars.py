"""
Preprocess raw VASP CHGCAR files into the .npy + _atoms.pkl format that
ChargE3Net's DensityPickleDir expects.

Inputs:
    --input    directory containing <mpid>.CHGCAR files
    --output   directory to write .npy / _atoms.pkl / filelist.txt / etc.
    --limit    optional cap on number of materials (for a fast shakedown run)
    --workers  parallel workers for the read+save loop

Outputs in --output/:
    <mpid>.npy           density grid
    <mpid>_atoms.pkl     ASE Atoms object
    filelist.txt         one mpid per line (master list for DensityPickleDir)
    probe_counts.csv     id, Count, shape_x, shape_y, shape_z (for full-grid mode)
    split.json           {"train": [], "validation": [], "test": [<all indices>]}

Charge density only (no spin). Skips materials whose CHGCAR is unreadable.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from functools import partial
from multiprocessing.pool import Pool
from pathlib import Path

from tqdm import tqdm

# Reuse upstream conversion + probe-count helpers so the output is exactly
# what DensityPickleDir / DensityGraphDataset expect.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.convert_chgcar_to_pkl import convert as chgcar_to_npypkl
from scripts.write_mp_probe_count_file import count_elements_in_numpy_files


def _convert_one(chgcar_path: Path, output_dir: Path) -> tuple[str, bool, str]:
    mpid = chgcar_path.stem  # "mp-1774721.CHGCAR" -> "mp-1774721"
    npy_path = output_dir / f"{mpid}.npy"
    pkl_path = output_dir / f"{mpid}_atoms.pkl"
    try:
        chgcar_to_npypkl(
            chgcar_path,
            npy_path,
            pkl_path,
            filelist_file=None,  # We build the filelist ourselves at the end.
            overwrite=False,  # Idempotent: re-runs skip already-converted files.
            spin=False,
        )
        return (mpid, True, "")
    except Exception:
        return (mpid, False, traceback.format_exc())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=0, help="0 = no limit")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    chgcars = sorted(args.input.glob("mp-*.CHGCAR"))
    print(f"Found {len(chgcars)} CHGCAR files in {args.input}")
    if args.limit > 0:
        chgcars = chgcars[: args.limit]
        print(f"Capped to first {len(chgcars)} for this run")

    print(f"Converting with {args.workers} workers -> {args.output}")
    fn = partial(_convert_one, output_dir=args.output)
    successes: list[str] = []
    failures: list[tuple[str, str]] = []
    with Pool(args.workers) as p:
        for mpid, ok, err in tqdm(p.imap_unordered(fn, chgcars), total=len(chgcars)):
            if ok:
                successes.append(mpid)
            else:
                failures.append((mpid, err))

    successes.sort()
    print(f"\nConverted: {len(successes)}, failed: {len(failures)}")
    if failures:
        for mpid, err in failures[:5]:
            print(f"  failed {mpid}:\n{err.splitlines()[-1]}")

    # filelist.txt — one mpid per line, in deterministic order.
    filelist_path = args.output / "filelist.txt"
    filelist_path.write_text("\n".join(successes) + "\n")
    print(f"Wrote {filelist_path} ({len(successes)} entries)")

    # probe_counts.csv — needed for full-grid splitting_mode in DensityGraphDataset.
    print("Counting probes per material (probe_counts.csv)...")
    count_elements_in_numpy_files(
        file_list_path=str(filelist_path), workers=args.workers
    )

    # split.json — put everything in "test" since this is an inference benchmark.
    split = {"train": [], "validation": [], "test": list(range(len(successes)))}
    split_path = args.output / "split.json"
    split_path.write_text(json.dumps(split))
    print(f"Wrote {split_path} (test={len(successes)})")


if __name__ == "__main__":
    main()
