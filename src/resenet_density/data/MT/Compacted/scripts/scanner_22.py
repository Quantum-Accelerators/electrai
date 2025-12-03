from __future__ import annotations

from sys import argv

import numpy as np
from pyscf import lib
from pyscf.dft import rks as molrks
from pyscf.pbc import dft, gto, tools
from pyscf.pbc.dft.multigrid.multigrid_pair import _eval_rhoG
from pyscf.pbc.scf.addons import smearing_

"""
argv[1]: input xyz
"""

basis1 = "gth-szv"
basis2 = "gth-tzv2p"
cut1 = 50
cut2 = 200
xcstr = "pbe"
ppstr = "gth-" + xcstr
conv_tol = 1e-7
conv_tol_grad = 1e-5
margin = 4
sigma = 0.01


class _RKS(dft.rks.RKS):
    # to get rid of pbc correction that is slow
    def _finalize(self):
        molrks.RKS._finalize(self)


def get_rho(mf, dm):
    # use mulitgrid to get rho; this is fast
    hermi = 1
    deriv = 0
    rhoG = _eval_rhoG(mf.with_df, dm, hermi, np.zeros((1, 3)), deriv)
    mesh = mf.with_df.mesh
    ngrids = np.prod(mesh)
    weight = mf.cell.vol / ngrids
    rhoR = tools.ifft(rhoG.reshape(-1, ngrids), mesh).real * (1.0 / weight)
    return rhoR[0]


fp = open(f"{argv[1]}")
natom = int(fp.readline())
fp.readline()
atoms = fp.readlines()[:natom]
fp.close()
coords = np.array([line.split()[1:4] for line in atoms], dtype=float)
charge = int(argv[2])
atoms = [" ".join(line.split()[:4]) for line in atoms]
geom_cen = np.mean(coords, axis=0)

# rotate 45 degree along y axis to have a smaller box
coords = coords - geom_cen
rot_matrix = np.array(
    [[0.70710678, 0.0, 0.70710678], [0.0, 1.0, 0.0], [-0.70710678, 0.0, 0.70710678]]
)
coords = coords @ rot_matrix
geom_cen = np.mean(coords, axis=0)

box = np.max(coords, axis=0) - np.min(coords, axis=0) + margin
box = np.ceil(box * np.sqrt(2 * cut1) / np.pi / lib.param.BOHR)
box = np.diag(box / np.sqrt(2 * cut1) * np.pi * lib.param.BOHR - 1e-4)
shift = np.diag(box) / 2 - geom_cen
coords = coords + shift
fp = open("centered.xyz", "w")
print(natom, file=fp)
print("", file=fp)
for line, c in zip(atoms, coords, strict=False):
    print(line.split()[0], *c, file=fp)
fp.close()
fp = open("centered.xyz")
fp.readline()
fp.readline()
atoms = fp.readlines()
fp.close()
np.savetxt("box.dat", box)

opt_cut1 = cut1
opt_cut2 = cut2

cell = gto.Cell()
cell.basis = basis1
cell.ke_cutoff = cut1
cell.a = box
cell.pseudo = ppstr
cell.atom = atoms
cell.max_memory = 10000
cell.precision = 1e-6
cell.rcut_by_shell_radius = True
cell.charge = charge
cell.build()
lattice_vectors = cell.lattice_vectors().copy()
# np.save("grid_coords_1.npy", cell11.get_uniform_grids())

cell11 = cell.copy()
cell11.ke_cutoff = opt_cut1
cell11.build()

cell12 = cell11.copy()
cell12.ke_cutoff = opt_cut2
cell12.build()
# np.save("grid_coords_2.npy", cell12.get_uniform_grids())

cell21 = cell11.copy()
cell21.basis = basis2
cell21.build()

cell22 = cell11.copy()
cell22.basis = basis2
cell22.ke_cutoff = opt_cut2
cell22.build()

cells = {"11": cell11, "12": cell12, "21": cell21, "22": cell22}


def make_mf(cell):
    df = dft.multigrid.MultiGridFFTDF2(cell)
    mf = _RKS(cell)
    mf.with_df = df
    mf.conv_tol = conv_tol
    mf.conv_tol_grad = conv_tol_grad
    mf.xc = xcstr
    mf.init_guess = "atom"
    mf.max_cycle = 200
    mf.verbose = 4
    return smearing_(mf, sigma=sigma)


mf11 = make_mf(cell11)
mf12 = make_mf(cell12)
mf21 = make_mf(cell21)
mf22 = make_mf(cell22)

mfs = {"11": mf11, "12": mf12, "21": mf21, "22": mf22}


def run_mf(mf, suffix):
    assert mf is mfs[suffix]
    dm = mf.get_init_guess()
    nelec = np.trace(dm @ mf.get_ovlp())
    dm *= mf.mol.nelectron / nelec
    ni = mf._numint
    rho = get_rho(mf, dm)
    np.save("rho_atom.npy", rho)
    E = mf.kernel(dm0=dm)
    if not mf.converged:
        print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
        print("@@@@@@@@@", suffix, "not converged @@@@@@@@@")
        print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
    np.savetxt(f"grid_sizes_{suffix}.dat", mf.grids.mesh, fmt="%d")
    dm = mf.make_rdm1()
    rho = get_rho(mf, dm)
    np.savetxt(f"energy_{suffix}.dat", [E])
    #    np.save(f"dm_{suffix}.npy", dm)
    np.save(f"rho_{suffix}.npy", rho)
    return dm


run_mf(mf22, "22")
