"""
Create a point-charge version of a CHGCAR file.

This script reads a CHGCAR file, extracts atomic positions, and creates a new
CHGCAR where the charge density consists of point charges located at each
atomic position. The total charge is distributed equally among all atoms.
"""

from __future__ import annotations

import argparse
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from pymatgen.io.vasp.outputs import Chgcar

logger = logging.getLogger(__name__)


def fractional_to_grid_index(
    frac_coords: np.ndarray, grid_shape: tuple[int, int, int]
) -> tuple[int, int, int]:
    """
    Convert fractional coordinates to the nearest grid index.

    Parameters
    ----------
    frac_coords : np.ndarray
        Fractional coordinates (a, b, c) in range [0, 1).
    grid_shape : tuple[int, int, int]
        Shape of the charge density grid (ngx, ngy, ngz).

    Returns
    -------
    tuple[int, int, int]
        Grid indices (i, j, k) for the nearest grid point.
    """
    # Wrap to [0, 1) range
    frac_coords = frac_coords % 1.0

    # Convert to grid indices
    indices = np.round(frac_coords * np.array(grid_shape)).astype(int)

    # Handle edge case where rounding gives grid_shape
    indices = indices % np.array(grid_shape)

    return tuple(indices)


def create_point_charge_density(
    chgcar: Chgcar, spread_charge: bool = False
) -> np.ndarray:
    """
    Create a point-charge density array from atomic positions.

    Parameters
    ----------
    chgcar : Chgcar
        Original CHGCAR object containing structure and charge density.
    spread_charge : bool
        If True, spread the charge using trilinear interpolation to the 8
        neighboring grid points. If False, place all charge at the nearest
        grid point.

    Returns
    -------
    np.ndarray
        3D array of charge density with point charges at atomic positions.
    """
    structure = chgcar.structure
    original_data = chgcar.data["total"]
    grid_shape = original_data.shape

    # Calculate total number of electrons from original charge density
    # CHGCAR stores charge density * volume, so integrate to get total electrons
    total_electrons = original_data.sum() / np.prod(grid_shape)

    # Initialize new charge density array
    point_charge_data = np.zeros(grid_shape, dtype=original_data.dtype)

    num_atoms = len(structure)
    charge_per_atom = total_electrons / num_atoms

    for site in structure:
        frac_coords = site.frac_coords

        if spread_charge:
            # Trilinear interpolation to 8 neighboring grid points
            point_charge_data = _add_spread_charge(
                point_charge_data, frac_coords, grid_shape, charge_per_atom
            )
        else:
            # Place all charge at nearest grid point
            idx = fractional_to_grid_index(frac_coords, grid_shape)
            point_charge_data[idx] += charge_per_atom * np.prod(grid_shape)

    return point_charge_data


def _add_spread_charge(
    data: np.ndarray,
    frac_coords: np.ndarray,
    grid_shape: tuple[int, int, int],
    charge: float,
) -> np.ndarray:
    """
    Add charge spread across 8 neighboring grid points using trilinear interpolation.

    Parameters
    ----------
    data : np.ndarray
        Charge density array to modify in place.
    frac_coords : np.ndarray
        Fractional coordinates of the atom.
    grid_shape : tuple[int, int, int]
        Shape of the grid.
    charge : float
        Total charge to distribute.

    Returns
    -------
    np.ndarray
        Modified charge density array.
    """
    # Wrap to [0, 1) range
    frac_coords = frac_coords % 1.0

    # Convert to continuous grid coordinates
    grid_coords = frac_coords * np.array(grid_shape)

    # Get lower corner indices
    i0, j0, k0 = np.floor(grid_coords).astype(int)

    # Get fractional position within the cell
    di = grid_coords[0] - i0
    dj = grid_coords[1] - j0
    dk = grid_coords[2] - k0

    # Trilinear interpolation weights
    weights = [
        (1 - di) * (1 - dj) * (1 - dk),  # (0, 0, 0)
        di * (1 - dj) * (1 - dk),  # (1, 0, 0)
        (1 - di) * dj * (1 - dk),  # (0, 1, 0)
        (1 - di) * (1 - dj) * dk,  # (0, 0, 1)
        di * dj * (1 - dk),  # (1, 1, 0)
        di * (1 - dj) * dk,  # (1, 0, 1)
        (1 - di) * dj * dk,  # (0, 1, 1)
        di * dj * dk,  # (1, 1, 1)
    ]

    # Corner offsets
    offsets = [
        (0, 0, 0),
        (1, 0, 0),
        (0, 1, 0),
        (0, 0, 1),
        (1, 1, 0),
        (1, 0, 1),
        (0, 1, 1),
        (1, 1, 1),
    ]

    # Add charge to each corner with periodic boundary conditions
    for weight, (oi, oj, ok) in zip(weights, offsets, strict=False):
        ii = (i0 + oi) % grid_shape[0]
        jj = (j0 + oj) % grid_shape[1]
        kk = (k0 + ok) % grid_shape[2]
        data[ii, jj, kk] += weight * charge * np.prod(grid_shape)

    return data


def create_point_charge_chgcar(
    input_path: str | Path, output_path: str | Path, spread_charge: bool = False
) -> None:
    """
    Read a CHGCAR file and create a point-charge version.

    Parameters
    ----------
    input_path : str | Path
        Path to the input CHGCAR file.
    output_path : str | Path
        Path to write the output CHGCAR file.
    spread_charge : bool
        If True, spread the charge using trilinear interpolation.
        If False (default), place all charge at the nearest grid point.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    logger.info(f"Reading CHGCAR from {input_path}")
    chgcar = Chgcar.from_file(str(input_path))

    structure = chgcar.structure
    logger.info(f"Structure: {structure.composition.reduced_formula}")
    logger.info(f"Number of atoms: {len(structure)}")
    logger.info(f"Grid shape: {chgcar.data['total'].shape}")

    original_total = chgcar.data["total"].sum() / np.prod(chgcar.data["total"].shape)
    logger.info(f"Original total electrons: {original_total:.4f}")

    logger.info("Creating point charge density...")
    point_charge_data = create_point_charge_density(chgcar, spread_charge=spread_charge)

    new_total = point_charge_data.sum() / np.prod(point_charge_data.shape)
    logger.info(f"New total electrons: {new_total:.4f}")

    # Create new Chgcar with point charge data
    new_chgcar = Chgcar(chgcar.poscar, {"total": point_charge_data})

    logger.info(f"Writing point-charge CHGCAR to {output_path}")
    new_chgcar.write_file(str(output_path))
    logger.info("Done!")


def _process_single_file(args: tuple[Path, Path, bool]) -> tuple[Path, bool, str]:
    """
    Worker function for parallel processing.

    Parameters
    ----------
    args : tuple[Path, Path, bool]
        Tuple of (input_path, output_path, spread_charge).

    Returns
    -------
    tuple[Path, bool, str]
        Tuple of (input_path, success, error_message).
    """
    input_path, output_path, spread_charge = args
    try:
        create_point_charge_chgcar(input_path, output_path, spread_charge)
        return (input_path, True, "")
    except Exception as e:
        return (input_path, False, str(e))


def convert_directory(
    input_dir: str | Path,
    output_dir: str | Path,
    pattern: str = "*.CHGCAR",
    spread_charge: bool = False,
    max_workers: int | None = None,
) -> tuple[int, int]:
    """
    Convert all CHGCAR files in a directory to point-charge versions.

    Parameters
    ----------
    input_dir : str | Path
        Directory containing CHGCAR files.
    output_dir : str | Path
        Directory where output CHGCAR files will be created.
    pattern : str
        Glob pattern to match input files (default: "*.CHGCAR").
    spread_charge : bool
        If True, spread the charge using trilinear interpolation.
    max_workers : int | None
        Maximum number of parallel workers. If None, uses the number of CPU cores.

    Returns
    -------
    tuple[int, int]
        Number of successfully converted files and number of failed conversions.
    """
    input_dir = Path(input_dir).expanduser()
    output_dir = Path(output_dir).expanduser()

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all matching files
    input_files = list(input_dir.glob(pattern))
    logger.info(f"Found {len(input_files)} files matching '{pattern}' in {input_dir}")

    if not input_files:
        return 0, 0

    # Prepare arguments for parallel processing
    conversion_args = [
        (input_file, output_dir / input_file.name, spread_charge)
        for input_file in input_files
    ]

    success_count = 0
    failed_count = 0

    # Process files in parallel
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_file = {
            executor.submit(_process_single_file, args): args[0]
            for args in conversion_args
        }

        for future in as_completed(future_to_file):
            input_file, success, error_msg = future.result()
            if success:
                success_count += 1
            else:
                logger.error(f"Failed to convert {input_file}: {error_msg}")
                failed_count += 1

    logger.info(
        f"Conversion complete: {success_count} successful, {failed_count} failed"
    )
    return success_count, failed_count


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser(
        description="Create point-charge versions of CHGCAR files."
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Single file conversion
    file_parser = subparsers.add_parser("convert", help="Convert a single CHGCAR file.")
    file_parser.add_argument("input", type=str, help="Path to the input CHGCAR file.")
    file_parser.add_argument(
        "output", type=str, help="Path to write the output CHGCAR file."
    )
    file_parser.add_argument(
        "--spread",
        action="store_true",
        help="Spread charge across 8 neighboring grid points using trilinear interpolation.",
    )

    # Directory conversion
    dir_parser = subparsers.add_parser(
        "convert-dir", help="Convert all CHGCAR files in a directory."
    )
    dir_parser.add_argument(
        "input_dir", type=str, help="Directory containing CHGCAR files."
    )
    dir_parser.add_argument(
        "output_dir", type=str, help="Directory to write output CHGCAR files."
    )
    dir_parser.add_argument(
        "--pattern",
        type=str,
        default="*.CHGCAR",
        help="Glob pattern to match input files (default: '*.CHGCAR').",
    )
    dir_parser.add_argument(
        "--spread",
        action="store_true",
        help="Spread charge across 8 neighboring grid points using trilinear interpolation.",
    )
    dir_parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of parallel workers (default: number of CPU cores).",
    )

    args = parser.parse_args()

    if args.command == "convert":
        create_point_charge_chgcar(
            input_path=args.input, output_path=args.output, spread_charge=args.spread
        )
    elif args.command == "convert-dir":
        convert_directory(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            pattern=args.pattern,
            spread_charge=args.spread,
            max_workers=args.workers,
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
