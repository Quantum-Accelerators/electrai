from __future__ import annotations

from sys import argv

import numpy as np
from pyscf import lib
from pyscf.data import elements
from pyscf.dft import rks as molrks
from pyscf.pbc import dft, gto, tools
from pyscf.pbc.dft.multigrid.multigrid_pair import _eval_rhoG
from pyscf.pbc.scf.addons import smearing_
from pyscf.scf import addons, atom_hf_pp, hf

atomic_configuration = elements.NRSRHF_CONFIGURATION

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
fp = open("check.xyz", "w")
print(natom, file=fp)
print("", file=fp)
for line, c in zip(atoms, coords, strict=False):
    print(line.split()[0], *c, file=fp)
fp.close()
fp = open("check.xyz")
fp.readline()
fp.readline()
atoms = fp.readlines()
fp.close()
# np.savetxt("box.dat", box)

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


mf22 = make_mf(cell22)

mfs = {"22": mf22}


def get_init_guess(mf):
    dm_results = {}
    for a in mf.cell.atom:
        ss = a.split()
        if ss[0] in dm_results:
            continue
        mol = gto.Cell()
        mol.atom = f"{ss[0]} 0  0  0"
        mol.charge = 0
        mol.enuc = 0
        mol.cart = False
        mol.basis = basis1
        mol.pseudo = ppstr
        mol.spin = elements.NUC[ss[0]] % 2
        mol.build()
        mol.a = None
        if mol.nelectron == 1:
            atm_hf = atom_hf_pp.AtomHF1ePP(mol)
            atm_hf.run()
            dm0 = hf.make_rdm1(atm_hf.mo_coeff, atm_hf.mo_occ)
        else:
            atm_hf = atom_hf_pp.AtomSCFPP(mol)
            atm_hf.atomic_configuration = atomic_configuration
            dm0 = atm_hf.get_init_guess(key="1e")
        mol2 = mol.copy()
        mol2.basis = basis2
        mol2.build()
        dm_results[ss[0]] = addons.project_dm_nr2nr(mol, dm0, mol2)
    slices = mf.cell.aoslice_by_atom()
    dm = np.zeros([mf.cell.nao] * 2)
    for i in range(mf.cell.natm):
        a = mf.cell.atom[i]
        ss = a.split()
        p0, p1 = slices[i][2:]
        dm[p0:p1, p0:p1] = dm_results[ss[0]]
    return dm


def run_mf(mf, suffix):
    assert mf is mfs[suffix]
    dm = get_init_guess(mf)
    nelec = lib.einsum("ij,ji->", dm, mf.get_ovlp())
    dm *= mf.mol.nelectron / nelec
    ni = mf._numint
    rho = get_rho(mf, dm)
    np.save("rho_atomh1e.npy", rho)
    #    np.savetxt(f"grid_sizes_{suffix}.dat", mf.grids.mesh, fmt="%d")
    rho = get_rho(mf, dm)
    return dm


run_mf(mf22, "22")
