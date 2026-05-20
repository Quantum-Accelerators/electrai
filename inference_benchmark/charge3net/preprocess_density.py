"""
Sample N test-set materials from rho_gga and convert each to ChargE3Net's
.npy + _atoms.pkl format.

Source layout (rho_gga is the dataset ckpt_1 was trained on):
    <input>/<mpid>.zarr/
        charge_density_total      # (NX, NY, NZ) float32
        attrs.structure           # pymatgen Structure JSON
        attrs.metadata            # misc

Sampling: only mpids listed in <split_file>["test"] are considered, so the
benchmark stays on materials the ResUNet checkpoint was not trained on.

Outputs in --output/:
    <mpid>.npy           density grid
    <mpid>_atoms.pkl     ASE Atoms object
    filelist.txt         master list for DensityPickleDir
    probe_counts.csv     id, Count, shape_x, shape_y, shape_z
    split.json           {"train": [], "validation": [], "test": [<all indices>]}

Charge density only (no spin). Skips materials whose zarr is unreadable.
"""

from __future__ import annotations

import argparse
import json
import pickle
import random
import sys
import traceback
from functools import partial
from multiprocessing.pool import Pool
from pathlib import Path

import numpy as np
import zarr
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.write_mp_probe_count_file import count_elements_in_numpy_files


def _convert_one(mpid: str, input_dir: Path, output_dir: Path) -> tuple[str, bool, str]:
    npy_path = output_dir / f"{mpid}.npy"
    pkl_path = output_dir / f"{mpid}_atoms.pkl"
    try:
        z = zarr.open(store=str(input_dir / f"{mpid}.zarr"), mode="r")
        density = np.array(z["charge_density_total"], dtype=np.float32)

        struct_attr = z.attrs["structure"]
        if isinstance(struct_attr, str):
            struct_attr = json.loads(struct_attr)
        structure = Structure.from_dict(struct_attr)
        atoms = AseAtomsAdaptor.get_atoms(structure)

        np.save(str(npy_path), density)
        with open(pkl_path, "wb") as f:
            pickle.dump(atoms, f)
        return (mpid, True, "")
    except Exception:
        return (mpid, False, traceback.format_exc())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Directory of <mpid>.zarr files (e.g. rho_gga/label/)",
    )
    parser.add_argument(
        "--filelist",
        required=True,
        type=Path,
        help="rho_gga/mp_filelist.txt — master list, indexes the split",
    )
    parser.add_argument(
        "--split-file",
        required=True,
        type=Path,
        help="rho_gga/split_limit_22M.json — picks the test partition",
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--limit",
        type=int,
        default=1000,
        help="Number of test materials to sample (0 = all)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    master = [m.strip() for m in args.filelist.read_text().splitlines() if m.strip()]
    split = json.loads(args.split_file.read_text())
    if "test" not in split:
        raise ValueError(f"Split file has no 'test' key (keys: {list(split)})")
    test_indices = split["test"]
    test_mpids = [master[i] for i in test_indices]
    print(f"Master filelist: {len(master)}; test partition: {len(test_mpids)}")

    if args.limit > 0 and args.limit < len(test_mpids):
        rng = random.Random(args.seed)
        test_mpids = rng.sample(test_mpids, args.limit)
        print(f"Random sample (seed={args.seed}): {len(test_mpids)} mpids")

    test_mpids.sort()

    print(f"Converting with {args.workers} workers -> {args.output}")
    fn = partial(_convert_one, input_dir=args.input, output_dir=args.output)
    successes: list[str] = []
    failures: list[tuple[str, str]] = []
    with Pool(args.workers) as p:
        for mpid, ok, err in tqdm(
            p.imap_unordered(fn, test_mpids), total=len(test_mpids)
        ):
            if ok:
                successes.append(mpid)
            else:
                failures.append((mpid, err))

    successes.sort()
    print(f"\nConverted: {len(successes)}, failed: {len(failures)}")
    if failures:
        for mpid, err in failures[:5]:
            print(f"  failed {mpid}: {err.splitlines()[-1]}")

    filelist_path = args.output / "filelist.txt"
    filelist_path.write_text("\n".join(successes) + "\n")
    print(f"Wrote {filelist_path} ({len(successes)} entries)")

    print("Counting probes per material (probe_counts.csv)...")
    count_elements_in_numpy_files(
        file_list_path=str(filelist_path), workers=args.workers
    )

    split_out = {"train": [], "validation": [], "test": list(range(len(successes)))}
    split_path = args.output / "split.json"
    split_path.write_text(json.dumps(split_out))
    print(f"Wrote {split_path} (test={len(successes)})")


if __name__ == "__main__":
    main()
