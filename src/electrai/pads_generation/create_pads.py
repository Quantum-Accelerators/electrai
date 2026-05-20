from __future__ import annotations

import argparse
import json
import shutil
from itertools import product as iproduct
from pathlib import Path

import numpy as np
import yaml
from numba import njit, prange

from electrai.zarr_conversion.convert_to_zarr import load_chgcar
from electrai.zarr_conversion.zarr_writer import write_chgcar_to_zarr


# -------------------- ARGS --------------------
def _derive_task_id(path: Path) -> str:
    name = path.name
    if name.endswith(".json.gz"):
        return name[: -len(".json.gz")]
    if name.lower().endswith(".chgcar"):
        return name[: -len(".chgcar")]
    return path.stem


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).parent.parent / "configs" / "config_pads.yaml",
    )
    p.add_argument(
        "--input", type=Path, help="Path to .json.gz or .CHGCAR file (overrides config)"
    )
    return p.parse_args()


# -------------------- NUMBA CORE --------------------
@njit(parallel=True, fastmath=True, cache=True)
def _accumulate_images(
    grid_cart, atom_images_cart, r_valid, d_valid, r_max, r_start, dr
):
    nvox = grid_cart.shape[0]
    nimg = atom_images_cart.shape[0]
    r_max2 = r_max * r_max
    M = r_valid.shape[0]

    total = np.zeros(nvox, dtype=np.float32)

    for vi in prange(nvox):
        gx = grid_cart[vi, 0]
        gy = grid_cart[vi, 1]
        gz = grid_cart[vi, 2]
        acc = np.float32(0.0)

        for ii in range(nimg):
            dx = gx - atom_images_cart[ii, 0]
            dy = gy - atom_images_cart[ii, 1]
            dz = gz - atom_images_cart[ii, 2]
            d2 = dx * dx + dy * dy + dz * dz
            if d2 > r_max2:
                continue

            r = np.sqrt(d2)

            idx = int((r - r_start) / dr)
            if idx < 0:
                acc += d_valid[0]
            elif idx < M - 1:
                t = (r - (r_start + idx * dr)) / dr
                acc += d_valid[idx] + t * (d_valid[idx + 1] - d_valid[idx])

        total[vi] = acc

    return total


# -------------------- GRID --------------------
def build_voxel_grid(cell, grid_shape):
    Nx, Ny, Nz = grid_shape
    cell = np.asarray(cell, dtype=np.float32)

    fx = (np.arange(Nx, dtype=np.float32) + 0.5) / Nx
    fy = (np.arange(Ny, dtype=np.float32) + 0.5) / Ny
    fz = (np.arange(Nz, dtype=np.float32) + 0.5) / Nz

    gx, gy, gz = np.meshgrid(fx, fy, fz, indexing="ij")
    grid_frac = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)

    return grid_frac @ cell


# -------------------- RECONSTRUCTION --------------------
def reconstruct_from_radial_function(
    radial_r,
    radial_density,
    ref_point,
    cell,
    grid_shape,
    extra_shells=1,
    prebuilt_grid=None,
):
    r_valid = np.asarray(radial_r, dtype=np.float32)
    d_valid = np.asarray(radial_density, dtype=np.float32)
    cell = np.asarray(cell, dtype=np.float32)

    r_max = float(r_valid[-1])
    r_start = float(r_valid[0])
    dr = float(r_valid[1] - r_valid[0])

    grid_cart = (
        prebuilt_grid
        if prebuilt_grid is not None
        else build_voxel_grid(cell, grid_shape)
    )

    lat_lengths = np.linalg.norm(cell, axis=1)
    nmax = int(np.ceil(r_max / lat_lengths.min())) + extra_shells

    shifts = np.array(
        list(iproduct(range(-nmax, nmax + 1), repeat=3)), dtype=np.float32
    )
    atom0_frac = np.asarray(ref_point, dtype=np.float32)
    atom_images_cart = (atom0_frac[None, :] + shifts) @ cell

    corners = np.array(
        [[i, j, k] for i in [0, 1] for j in [0, 1] for k in [0, 1]], dtype=np.float32
    )
    corners_cart = corners @ cell

    lo = corners_cart.min(axis=0)
    hi = corners_cart.max(axis=0)

    clamped = np.clip(atom_images_cart, lo, hi)
    dist_to_box = np.linalg.norm(atom_images_cart - clamped, axis=1)
    atom_images_cart = atom_images_cart[dist_to_box <= r_max]

    total_flat = _accumulate_images(
        grid_cart, atom_images_cart, r_valid, d_valid, r_max, r_start, dr
    )
    return total_flat.reshape(grid_shape)


# -------------------- TASK --------------------
def process_task(
    input_path: Path,
    task_id: str,
    output_dir: Path,
    radial_cache: dict,
    zval_dict: dict,
    extra_shells: int,
):
    out_zarr = output_dir / f"{task_id}.zarr"

    cd = load_chgcar(input_path)

    structure = cd.structure
    grid_size = cd.data["total"].shape
    cell_matrix = structure.lattice.matrix

    chg_label = np.array(cd.data["total"], dtype=np.float64)

    if out_zarr.exists():
        shutil.rmtree(out_zarr)

    new_grid = np.zeros(grid_size, dtype=np.float64)
    grid_cart = build_voxel_grid(cell_matrix, grid_size)
    voxel_volume = structure.lattice.volume / np.prod(grid_size)

    for site in structure:
        specie = str(site.specie)
        radial_r, radial_density = radial_cache[specie]

        atom_contribution = reconstruct_from_radial_function(
            radial_r,
            radial_density,
            ref_point=site.frac_coords,
            cell=cell_matrix,
            grid_shape=grid_size,
            extra_shells=extra_shells,
            prebuilt_grid=grid_cart,
        )
        nelecs = zval_dict[specie]
        t = (nelecs / voxel_volume) / np.sum(atom_contribution)
        atom_contribution *= t
        new_grid += atom_contribution

    new_grid *= structure.lattice.volume
    cd.data["total"] = new_grid

    write_chgcar_to_zarr(cd, out_zarr, write_diff=False, chunks=None)

    n_vox = chg_label.size
    den = np.sum(chg_label)
    nmae = float(np.sum(np.abs(chg_label - new_grid)) / den) if den != 0 else np.nan
    nelec_diff = float((np.sum(new_grid) - np.sum(chg_label)) / n_vox)

    return nmae, nelec_diff


# -------------------- MAIN --------------------
def main():
    args = parse_args()

    with args.config.open() as f:
        cfg = yaml.safe_load(f)

    paths = cfg["paths"]
    input_path = (
        Path(args.input) if args.input is not None else Path(paths["input_file"])
    )
    output_dir = Path(paths["output_dir"])
    stats_dir = Path(paths["stats_dir"])
    radial_dir = Path(paths["radial_dir"])
    zval_path = Path(paths["zval_file"])
    extra_shells = cfg["pads"]["extra_shells"]

    task_id = _derive_task_id(input_path)

    output_dir.mkdir(parents=True, exist_ok=True)
    stats_dir.mkdir(parents=True, exist_ok=True)

    radial_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for fp in radial_dir.glob("*.npy"):
        specie = fp.stem
        arr = np.load(fp)
        r, d = arr[:, 0].astype(np.float32), arr[:, 1].astype(np.float32)
        valid = np.isfinite(r) & np.isfinite(d)
        r, d = r[valid], d[valid]
        order = np.argsort(r)
        r, d = r[order], d[order]
        # NaN removal leaves non-uniform spacing; resample to uniform grid
        # so the O(1) index in _accumulate_images stays correct.
        dr_raw = np.float32(0.0025)
        n_uniform = round((r[-1] - r[0]) / dr_raw) + 1
        r_uniform = np.linspace(r[0], r[-1], n_uniform, dtype=np.float32)
        d_uniform = np.interp(r_uniform, r, d).astype(np.float32)
        radial_cache[specie] = (r_uniform, d_uniform)

    with zval_path.open() as f:
        zval_dict: dict[str, float] = json.load(f)

    nmae, nelec_diff = process_task(
        input_path, task_id, output_dir, radial_cache, zval_dict, extra_shells
    )

    print(f"{task_id} nmae={nmae:.6e} nelec_diff={nelec_diff:.6e}", flush=True)  # noqa: T201

    result_path = stats_dir / f"{task_id}.json"
    with result_path.open("w") as f:
        json.dump({"nmae": nmae, "nelec_diff": nelec_diff}, f)


if __name__ == "__main__":
    main()
