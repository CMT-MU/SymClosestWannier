"""
tests for seedname_tb.dat.cw (wannier90 _tb.dat format) written by CWModel.write_tb.
"""

import sys

import numpy as np
import pytest

from symclosestwannier.cw.cw_manager import CWManager
from symclosestwannier.cw.cw_model import CWModel
from symclosestwannier.cw.cwin import CWin
from symclosestwannier.util.exceptions import SymCWInputError

SEEDNAME = "model"


# ==================================================
def make_model(nrpts=31):
    """
    artificial model: non-orthogonal lattice, complex H(R) and A(R), ndegen != 1.
    ndegen is written 15 per line, so nrpts = 1, 15, 16, 31 cover the line boundaries.
    """
    rng = np.random.default_rng(nrpts)
    num_wann = 3
    irvec = np.array([[i, j, k] for i in range(-2, 3) for j in range(-2, 3) for k in range(-1, 2)])
    irvec = irvec[np.argsort(np.linalg.norm(irvec, axis=1), kind="stable")][:nrpts]
    ndegen = rng.integers(1, 5, size=nrpts)
    Hr = rng.normal(size=(nrpts, num_wann, num_wann)) + 1j * rng.normal(size=(nrpts, num_wann, num_wann))
    Ar = rng.normal(size=(3, nrpts, num_wann, num_wann)) + 1j * rng.normal(size=(3, nrpts, num_wann, num_wann))
    unit_cell_cart = np.array([[2.5, 0.0, 0.0], [-1.25, 2.165, 0.0], [0.3, 0.4, 7.0]])

    cwi = {
        "seedname": SEEDNAME,
        "num_wann": num_wann,
        "unit_cell_cart": unit_cell_cart.tolist(),
        "irvec": irvec.tolist(),
        "ndegen": ndegen.tolist(),
        "tb_gauge": False,
    }

    return cwi, Hr, Ar


# ==================================================
@pytest.fixture(autouse=True)
def restore_state(monkeypatch):
    """
    CWManager changes the current directory and appends it to sys.path.
    """
    monkeypatch.chdir(".")
    monkeypatch.setattr(sys, "path", list(sys.path))


# ==================================================
def write_tb(tmp_path, cwi, Hr, Ar):
    cwm = CWManager(topdir=str(tmp_path), verbose=False, parallel=False, formatter=False)
    cw_model = CWModel(cwi, cwm, dic={})
    cw_model.write_tb(Hr, Ar, f"{SEEDNAME}_tb.dat.cw")

    return tmp_path / f"{SEEDNAME}_tb.dat.cw"


# ==================================================
def read_tb_dat(filename):
    """
    read a file in the wannier90 _tb.dat layout:
    header, 3 lattice vectors, num_wann, nrpts, ndegen (15 per line),
    then for each R a blank line, R and num_wann^2 lines "m n Re Im" (m fastest),
    and the same for the position operator with 3 complex components.

    Returns:
        tuple: (lattice, ndegen, irvec, Hr, Ar), Hr[R, m, n] and Ar[a, R, m, n] as written (not divided by ndegen).
    """
    with open(filename) as fp:
        lines = fp.read().split("\n")

    lattice = np.array([lines[i].split() for i in range(1, 4)], dtype=float)
    num_wann = int(lines[4])
    nrpts = int(lines[5])
    nline = (nrpts + 14) // 15
    ndegen = np.array(" ".join(lines[6 : 6 + nline]).split(), dtype=int)
    assert len(ndegen) == nrpts

    pos = 6 + nline
    irvec = []
    Hr = np.zeros((nrpts, num_wann, num_wann), dtype=complex)
    Ar = np.zeros((3, nrpts, num_wann, num_wann), dtype=complex)
    for block in ("H", "A"):
        for ir in range(nrpts):
            assert lines[pos].strip() == "", f"line {pos + 1}: blank line expected before R block."
            R = [int(x) for x in lines[pos + 1].split()]
            if block == "H":
                irvec.append(R)
            else:
                assert R == irvec[ir]
            pos += 2
            for i in range(num_wann):
                for j in range(num_wann):
                    v = lines[pos].split()
                    assert len(v) == (4 if block == "H" else 8), f"line {pos + 1}: unexpected number of fields."
                    m, n = int(v[0]) - 1, int(v[1]) - 1
                    assert (m, n) == (j, i), f"line {pos + 1}: unexpected orbital order."
                    x = np.array(v[2:], dtype=float)
                    if block == "H":
                        Hr[ir, m, n] = x[0] + 1j * x[1]
                    else:
                        Ar[:, ir, m, n] = x[0::2] + 1j * x[1::2]
                    pos += 1

    return lattice, ndegen, np.array(irvec), Hr, Ar


# ==================================================
@pytest.mark.parametrize("nrpts", [1, 15, 16, 31])
def test_write_tb_wannier90_layout(tmp_path, nrpts):
    cwi, Hr, Ar = make_model(nrpts)
    filename = write_tb(tmp_path, cwi, Hr, Ar)

    lattice, ndegen, irvec, Hr_read, Ar_read = read_tb_dat(filename)

    np.testing.assert_allclose(lattice, cwi["unit_cell_cart"], rtol=0, atol=1e-8)
    np.testing.assert_array_equal(ndegen, cwi["ndegen"])
    np.testing.assert_array_equal(irvec, cwi["irvec"])
    np.testing.assert_allclose(Hr_read, Hr, rtol=1e-7, atol=1e-7)
    np.testing.assert_allclose(Ar_read, Ar, rtol=1e-7, atol=1e-7)


# ==================================================
def test_write_tb_rejects_tb_gauge(tmp_path):
    """
    with tb_gauge = true the phase of H(R) contains atomic positions, which is not the wannier90 convention.
    """
    cwi, Hr, Ar = make_model()
    cwi["tb_gauge"] = True

    with pytest.raises(SymCWInputError, match="tb_gauge"):
        write_tb(tmp_path, cwi, Hr, Ar)

    assert not (tmp_path / f"{SEEDNAME}_tb.dat.cw").exists()


# ==================================================
def test_cwin_rejects_write_tb_with_tb_gauge(tmp_path):
    """
    the conflict is reported when reading seedname.cwin, before any calculation.
    """
    (tmp_path / f"{SEEDNAME}.cwin").write_text("write_tb = true\ntb_gauge = true\n")

    with pytest.raises(SymCWInputError, match="tb_gauge"):
        CWin(str(tmp_path), SEEDNAME)


# ==================================================
def test_write_tb_readable_by_wannierberri(tmp_path):
    get_system_tb = pytest.importorskip("wannierberri.system.system_tb").get_system_tb

    cwi, Hr, Ar = make_model()
    filename = write_tb(tmp_path, cwi, Hr, Ar)

    system = get_system_tb(tb_file=str(filename), convention_II_to_I=False, berry=True, silent=True)

    ndegen = np.array(cwi["ndegen"])
    index = {tuple(R): ir for ir, R in enumerate(cwi["irvec"])}
    order = [index[tuple(R)] for R in system.rvec.iRvec]

    # wannierberri: Ham_R[R, m, n], AA_R[R, m, n, a], both divided by ndegen.
    np.testing.assert_allclose(system.real_lattice, cwi["unit_cell_cart"], rtol=0, atol=1e-8)
    np.testing.assert_allclose(system.Ham_R, (Hr / ndegen[:, None, None])[order], rtol=1e-6, atol=1e-7)
    AA_R = (Ar / ndegen[None, :, None, None]).transpose(1, 2, 3, 0)[order]
    np.testing.assert_allclose(system.get_R_mat("AA"), AA_R, rtol=1e-6, atol=1e-7)
