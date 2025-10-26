from __future__ import annotations

import pickle

from pyscf.data import elements
from pyscf.pbc import gto
from pyscf.scf import atom_hf_pp, hf

atomic_configuration = elements.NRSRHF_CONFIGURATION

basis1 = "gth-szv"
basis2 = "gth-tzv2p"
cut1 = 50
cut2 = 200
xcstr = "pbe"
ppstr = "gth-" + xcstr


dm_results_basis1 = dict()
dm_results_basis2 = dict()
for a in ["H"]:
    print(a)
    mol = gto.Cell()
    mol.atom = f"{a} 0  0  0"
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
    # mol2 = mol.copy()
    # mol2.basis = basis2
    # mol2.build()
    dm_results_basis1[a] = dm0
    # dm_results_basis2[a] = addons.project_dm_nr2nr(mol, dm0, mol2)

with open("atomic_dm_table_szv.pkl", "wb") as f:
    pickle.dump(dm_results_basis1, f)

# with open("atomic_dm_table_tzv2p.pkl", "wb") as f:
#     pickle.dump(dm_results_basis2, f)
