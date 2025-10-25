from __future__ import annotations

from sys import argv

import numpy as np
from pyscf import lib
from pyscf.data import elements
from pyscf.pbc import dft, gto, tools
from pyscf.pbc.dft.multigrid.multigrid_pair import _eval_rhoG
from pyscf.scf import addons, atom_hf_pp, hf

atomic_configuration = elements.NRSRHF_CONFIGURATION

"""
argv[1]: directory to coordinates
argv[2]: system name (w/o .xyz)
"""

basis1 = "gth-szv"
basis2 = "gth-tzv2p"
cut1 = 50
cut2 = 200
xcstr = "pbe"
ppstr = "gth-" + xcstr
conv_tol = 1e-11
margin = 4

fp = open(f"{argv[1]}/{argv[2]}.xyz")
natom = int(fp.readline())
fp.readline()
atoms = fp.readlines()[:natom]
fp.close()
coords = np.array([line.split()[1:4] for line in atoms], dtype=float)
charge = np.round(sum([float(line.split()[4]) for line in atoms])).astype(int)
atoms = [" ".join(line.split()[:4]) for line in atoms]
geom_cen = np.mean(coords, axis=0)
box = np.max(coords, axis=0) - np.min(coords, axis=0) + margin
box = np.ceil(box * np.sqrt(2 * cut1) / np.pi / lib.param.BOHR)
box = np.diag(box / np.sqrt(2 * cut1) * np.pi * lib.param.BOHR - 1e-4)
shift = np.diag(box) / 2 - geom_cen
coords = coords + shift
fp = open("centered.xyz", "w")
print(natom, file=fp)
print("", file=fp)
for line, c in zip(atoms, coords):
    print(line.split()[0], *c, file=fp)
fp.close()
fp = open("centered.xyz")
fp.readline()
fp.readline()
atoms = fp.readlines()
fp.close()

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

cell22 = cell.copy()
cell22.basis = basis2
cell22.ke_cutoff = opt_cut2
cell22.build()


def make_mf(cell):
    df = dft.multigrid.MultiGridFFTDF2(cell)
    mf = dft.rks.RKS(cell)
    mf.with_df = df
    mf.conv_tol = conv_tol
    mf.xc = xcstr
    mf.init_guess = "atom"
    mf.max_cycle = 200
    return mf


mf22 = make_mf(cell22)

mfs = {"22": mf22}


def get_init_guess(mf):
    dm_results = {}
    for a in mf.cell.atom:
        if a[0] in dm_results:
            continue
        mol = gto.Cell()
        mol.atom = f"{a[0]} 0  0  0"
        mol.charge = 0
        mol.enuc = 0
        mol.cart = False
        mol.basis = basis1
        mol.pseudo = ppstr
        mol.spin = elements.NUC[a[0]] % 2
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
        dm_results[a[0]] = addons.project_dm_nr2nr(mol, dm0, mol2)
    slices = mf.cell.aoslice_by_atom()
    dm = np.zeros([mf.cell.nao] * 2)
    for i in range(mf.cell.natm):
        a = mf.cell.atom[i]
        p0, p1 = slices[i][2:]
        dm[p0:p1, p0:p1] = dm_results[a[0]]
    return dm


def run_mf(mf, suffix):
    assert mf is mfs[suffix]
    mol = mf.cell
    dm0 = get_init_guess(mf)
    nelec = np.trace(dm0 @ mf.get_ovlp())
    if abs(nelec - mol.nelectron) / mol.nelectron > 0.01:
        print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
        print("@@@@@@@@@", "nelectron error in dm0", nelec, mol.nelectron, "@@@@@@@@@")
        print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
    dm0 = dm0 / nelec * mol.nelectron
    # get rho
    hermi = 1
    deriv = 0
    rhoG = _eval_rhoG(mf.with_df, dm0, hermi, np.zeros((1, 3)), deriv)
    mesh = mf.with_df.mesh
    ngrids = np.prod(mesh)
    weight = mf.cell.vol / ngrids
    rhoR = tools.ifft(rhoG.reshape(-1, ngrids), mesh).real * (1.0 / weight)
    rho = rhoR[0]
    np.savetxt(f"grid_sizes_{suffix}.dat", mf.grids.mesh, fmt="%d")
    np.save(f"dm_{suffix}.npy", dm0)
    np.save(f"rho_{suffix}.npy", rho)
    return dm0


run_mf(mf22, "22")
