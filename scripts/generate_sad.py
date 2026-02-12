# from pymatgen.io.ase import AseAtomsAdaptor
# from pymatgen.io.vasp import Chgcar
from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import yaml
from ase.geometry import cellpar_to_cell
from ase.io import read
from gpaw import GPAW, PW

if TYPE_CHECKING:
    from ase import Atoms


def read_structure(folder_path: Path) -> Atoms:
    poscar_path = folder_path / "POSCAR"
    xyz_path = folder_path / "coord.xyz"
    cell_path = folder_path / "cell.dat"

    if poscar_path.is_file():
        atoms = read(poscar_path)

    elif xyz_path.is_file() and cell_path.is_file():
        atoms = read(xyz_path)

        cell_data = np.loadtxt(cell_path)

        cell = cell_data if cell_data.shape == (3, 3) else cellpar_to_cell(cell_data)

        atoms.set_cell(cell)
        atoms.set_pbc(True)

    else:
        raise FileNotFoundError(f"Missing required structure files in {folder_path}")
    return atoms


def run_calculation(atoms: Atoms):
    cutoff = 200
    # Run GPAW calculation
    calc = GPAW(
        mode=PW(cutoff),
        xc="PBE",
        # h=0.2,
        txt="gpaw.log",
    )
    atoms.calc = calc
    atoms.get_potential_energy()
    # After running the calculation
    # nelectrons = calc.get_number_of_electrons()


if __name__ == "__main__":
    config_file = sys.argv[1]
    with Path.open(config_file) as f:
        cfg = yaml.safe_load(f)
    root = Path(cfg["root"])
    folders = root.iterdir()
    for folder in folders:
        folder_path = Path(root) / folder
        if folder_path.is_dir():
            atoms = read_structure(folder_path)
            run_calculation(atoms)
