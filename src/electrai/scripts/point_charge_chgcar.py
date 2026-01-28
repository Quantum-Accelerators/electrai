"""
Create a point-charge version of a CHGCAR file.

This script reads a CHGCAR file, extracts atomic positions, and creates a new
CHGCAR where the charge density consists of point charges located at each
atomic position. The total charge is distributed equally among all atoms.
"""

from __future__ import annotations

import argparse
import logging
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


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser(
        description="Create a point-charge version of a CHGCAR file."
    )
    parser.add_argument("input", type=str, help="Path to the input CHGCAR file.")
    parser.add_argument(
        "output", type=str, help="Path to write the output CHGCAR file."
    )
    parser.add_argument(
        "--spread",
        action="store_true",
        help="Spread charge across 8 neighboring grid points using trilinear interpolation.",
    )

    args = parser.parse_args()

    create_point_charge_chgcar(
        input_path=args.input, output_path=args.output, spread_charge=args.spread
    )


if __name__ == "__main__":
    main()
