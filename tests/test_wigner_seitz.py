"""
R vectors and degeneracies of the Wigner-Seitz supercell (wigner_seitz), as wannier90's hamiltonian_wigner_seitz, and
the keywords ws_distance_tol and ws_search_size of seedname.win.
"""

import os

import numpy as np
import pytest

from symclosestwannier.cw.win import Win
from symclosestwannier.util.exceptions import SymCWInputError
from symclosestwannier.util.utility import wigner_seitz

DATA = os.path.join(os.path.dirname(__file__), "data", "wigner_seitz_w90.dat")


# ==================================================
def read_reference():
    header = [l[1:].split() for l in open(DATA) if l.startswith("#")]
    A = np.array(header[2:5], dtype=float)
    mp_grid = [int(x) for x in header[5]]
    data = np.loadtxt(DATA, dtype=int)
    return A, mp_grid, data[:, :3], data[:, 3]


# ==================================================
def test_wannier90_reference():
    """
    a hexagonal cell rounded to 7 digits: |a1| and |a2| differ by ~1e-6 Angstrom, so wannier90 does not treat the R
    vectors related by the broken symmetry as equidistant (a tolerance of 1e-4 Angstrom^2 gave 36 more R vectors).
    """
    A, mp_grid, irvec_ref, ndegen_ref = read_reference()
    irvec, ndegen = wigner_seitz(A, mp_grid)
    np.testing.assert_array_equal(irvec, irvec_ref)
    np.testing.assert_array_equal(ndegen, ndegen_ref)


# ==================================================
@pytest.mark.parametrize("mp_grid", [[1, 1, 1], [4, 4, 4], [3, 4, 5], [6, 6, 1]])
def test_cubic(mp_grid):
    """every point of the supercell once, symmetric under R -> -R, and the faces of an even mesh shared."""
    irvec, ndegen = wigner_seitz(2.5 * np.eye(3), mp_grid)
    assert np.sum(1.0 / ndegen) == pytest.approx(np.prod(mp_grid))
    pairs = {tuple(r): n for r, n in zip(irvec, ndegen)}
    assert all(pairs[tuple(-r)] == n for r, n in zip(irvec, ndegen))
    for r, n in zip(irvec, ndegen):
        # on a simple cubic supercell, R is shared by 2^(number of components on a face, |n_i| = N_i / 2)
        assert n == 2 ** sum(2 * abs(x) == N for x, N in zip(r, mp_grid))


# ==================================================
def test_tolerance():
    """a looser tolerance makes the rounded hexagonal cell hexagonal again (more R vectors on the boundary)."""
    A, mp_grid, irvec_ref, _ = read_reference()
    irvec, ndegen = wigner_seitz(A, mp_grid, ws_distance_tol=1e-2)
    assert len(irvec) > len(irvec_ref)
    assert np.sum(1.0 / ndegen) == pytest.approx(np.prod(mp_grid))


# ==================================================
@pytest.mark.parametrize("eps, tie", [(0.2e-10, True), (0.5e-10, False)])
def test_tolerance_boundary(eps, tie):
    """
    a sheared square cell: |R|^2 - |R - T|^2 = 4 eps for R = (1, 1, 0) and T = (2, 0, 0) on a 2x2x1 mesh, against
    ws_distance_tol^2 = 1e-10 Angstrom^2.
    """
    A = np.array([[1.0, 0.0, 0.0], [eps, 1.0, 0.0], [0.0, 0.0, 1.0]])
    irvec, ndegen = wigner_seitz(A, [2, 2, 1])
    degen = {tuple(r): n for r, n in zip(irvec, ndegen)}
    assert ((1, 1, 0) in degen) == tie
    assert degen[(-1, 1, 0)] == (4 if tie else 2)
    assert np.sum(1.0 / ndegen) == pytest.approx(4)


# ==================================================
def test_legacy_prec():
    """prec of earlier versions (a tolerance of the squared distances), given by position or by name."""
    A, mp_grid, irvec_ref, _ = read_reference()
    by_position, by_name = wigner_seitz(A, mp_grid, 1e-4), wigner_seitz(A, mp_grid, prec=1e-4)
    for a, b in zip(by_position, by_name):
        np.testing.assert_array_equal(a, b)
    irvec, ndegen = by_position
    assert len(irvec) == 657 and np.sum(1.0 / ndegen) == pytest.approx(np.prod(mp_grid))
    # the R vectors of wannier90 are among them, with the same order
    keep = [i for i, r in enumerate(map(tuple, irvec)) if r in set(map(tuple, irvec_ref))]
    np.testing.assert_array_equal(irvec[keep], irvec_ref)
    assert len(wigner_seitz(A, mp_grid)[0]) == len(irvec_ref) == 621


# ==================================================
def test_search_size():
    """a strongly sheared cell needs more than the neighbouring supercells; too small a search is an error."""
    A = np.array([[1.0, 0.0, 0.0], [3.7, 1.0, 0.0], [0.0, 0.0, 1.0]])
    with pytest.raises(SymCWInputError, match="ws_search_size"):
        wigner_seitz(A, [4, 4, 1], ws_search_size=1)
    irvec, ndegen = wigner_seitz(A, [4, 4, 1], ws_search_size=[3, 3, 1])
    assert np.sum(1.0 / ndegen) == pytest.approx(16)
    # the nearest image of each R is the shortest
    A_sc = A * np.array([4, 4, 1])[:, None]
    for r in irvec:
        x = r @ A
        images = [x - np.array([i, j, 0]) @ A_sc for i in range(-4, 5) for j in range(-4, 5)]
        assert x @ x <= min(y @ y for y in images) + 1e-8


# ==================================================
_cell = """
begin unit_cell_cart
ang
1 0 0
0 1 0
0 0 1
end unit_cell_cart
begin atoms_frac
A 0 0 0
end atoms_frac
"""


def write_win(tmp_path, text):
    (tmp_path / "seed.win").write_text("num_bands = 2\nnum_wann = 2\n" + text + _cell)
    return Win(str(tmp_path), "seed")


@pytest.mark.parametrize(
    "text, tol, size",
    [
        ("", 1e-5, [2, 2, 2]),
        ("ws_distance_tol = 1.d-3\nws_search_size = 3\n", 1e-3, [3, 3, 3]),
        ("ws_search_size : 1 2 3\n", 1e-5, [1, 2, 3]),
    ],
)
def test_win_keywords(tmp_path, text, tol, size):
    win = write_win(tmp_path, text)
    assert win["ws_distance_tol"] == tol
    assert win["ws_search_size"] == size


@pytest.mark.parametrize(
    "text",
    [
        "ws_search_size = 1 2\n",
        "ws_search_size = 0\n",
        "ws_search_size = 1.5\n",
        "ws_distance_tol = 0.0\n",
        "ws_distance_tol = nan\n",
        "ws_distance_tol = inf\n",
        "ws_distance_tol = -inf\n",
    ],
)
def test_win_keywords_invalid(tmp_path, text):
    with pytest.raises(SymCWInputError):
        write_win(tmp_path, text)


# ==================================================
def test_model_from_hdf5_keeps_its_r_vectors(make_case):
    """
    postcw rebuilds CWInfo from seedname.hdf5: the R vectors are those the operators were written on (for example by an
    older version with another tolerance), not new ones from wigner_seitz. Here the stored order is reversed.
    """
    from symclosestwannier.cw.cw_creator import cw_creator
    from symclosestwannier.cw.cw_info import CWInfo
    from symclosestwannier.cw.cw_model import CWModel

    seedname = "graphene_pz"
    workdir = make_case(seedname, extra="write_info_data = true\nverbose = false")
    cw_creator(seedname)

    info, _, _ = CWModel.read_info_data(str(workdir / f"{seedname}.hdf5"))
    irvec, ndegen = np.array(info["irvec"])[::-1], np.array(info["ndegen"])[::-1]
    assert not np.array_equal(irvec, info["irvec"])
    info["irvec"], info["ndegen"] = irvec.tolist(), ndegen.tolist()

    cwi = CWInfo(str(workdir), seedname, dic=info, postcw=True)
    np.testing.assert_array_equal(cwi["irvec"], irvec)
    np.testing.assert_array_equal(cwi["ndegen"], ndegen)


# ==================================================
def test_new_model_recomputes_its_r_vectors(make_case):
    """outside postcw, a dictionary with R vectors (for example to build a model again) gets them from wigner_seitz."""
    from symclosestwannier.cw.cw_creator import cw_creator
    from symclosestwannier.cw.cw_info import CWInfo
    from symclosestwannier.cw.cw_model import CWModel

    seedname = "graphene_pz"
    workdir = make_case(seedname, extra="write_info_data = true\nverbose = false")
    cw_creator(seedname)

    info, _, _ = CWModel.read_info_data(str(workdir / f"{seedname}.hdf5"))
    irvec = np.array(info["irvec"])
    info["irvec"], info["ndegen"] = irvec[::-1].tolist(), np.array(info["ndegen"])[::-1].tolist()

    cwi = CWInfo(str(workdir), seedname, dic=info)
    np.testing.assert_array_equal(cwi["irvec"], irvec)
