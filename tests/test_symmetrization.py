"""
symmetrization with the SAMB data of others/examples (MultiPie v2): the symmetrized Hamiltonian
seedname_hr_sym.dat.cw has the same eigenvalues as seedname_hr.dat.cw within the truncation of the model.
"""

import os
import shutil

import numpy as np
import pytest

from symclosestwannier.cw.cw_creator import cw_creator

from conftest import EXAMPLES_DIR, read_hr_dat

KET_AMN = {
    "ch4_sl": "[[C,1,0,s],[C,1,1,pz],[C,1,1,px],[C,1,1,py],[H,1,0,s],[H,2,0,s],[H,4,0,s],[H,3,0,s]]",
    "graphene_pz": "auto",
}


# ==================================================
def read_hr_sym_dat(filename, tmp_path):
    """
    read seedname_hr_sym.dat.cw, which has a header before the hr.dat format.
    """
    with open(filename) as fp:
        lines = fp.readlines()
    start = next(i for i, line in enumerate(lines) if line.startswith("# written"))
    body = tmp_path / "hr_sym.dat"
    body.write_text("".join(lines[start:]))

    return read_hr_dat(str(body))


# ==================================================
def eigenvalues(ndegen, irvec, Hr, k):
    phase = np.exp(2j * np.pi * k @ irvec.T) / ndegen
    return np.linalg.eigvalsh(np.einsum("kr,rmn->kmn", phase, Hr))


# ==================================================
@pytest.mark.parametrize("seedname, atol", [("ch4_sl", 1e-6), ("graphene_pz", 0.03)])
def test_symmetrized_hamiltonian(make_case, tmp_path, seedname, atol):
    """
    ch4_sl: H(0) is fully symmetric, so the eigenvalues agree.
    graphene_pz: the SAMBs cover up to the 20th neighbor, the rest of H(R) is dropped (max. deviation 18 meV).
    """
    extra = f"write_hr = true\nsymmetrization = true\nmp_outdir = ./{seedname}\nmp_seedname = {seedname}\nket_amn = {KET_AMN[seedname]}"
    workdir = make_case(seedname, extra=extra)
    (workdir / seedname).mkdir()
    shutil.copy(os.path.join(EXAMPLES_DIR, seedname, seedname, f"{seedname}.pkl"), workdir / seedname)

    cw_creator(seedname)

    hr = read_hr_dat(str(workdir / f"{seedname}_hr.dat.cw"))
    hr_sym = read_hr_sym_dat(workdir / seedname / f"{seedname}_hr_sym.dat.cw", tmp_path)

    k = np.random.default_rng(0).random((20, 3))
    np.testing.assert_allclose(eigenvalues(*hr_sym, k), eigenvalues(*hr, k), rtol=0, atol=atol)
