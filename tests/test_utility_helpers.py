"""
helpers that replace gcoreutils: grid points along a k path (grid_path), fractional coordinates shifted into the unit
cell (shift_into_cell, atoms_frac_shift of seedname.win), and the dict text files of CWManager (read_dict, write_dict).
"""

import numpy as np
import pytest

from symclosestwannier.cw.cw_manager import read_dict, write_dict
from symclosestwannier.cw.win import Win
from symclosestwannier.util.utility import grid_path, shift_into_cell


# ==================================================
def test_grid_path():
    pts = {"G": [0, 0, 0], "X": [0.5, 0, 0], "M": [0.5, 0.5, 0], "R": [0.5, 0.5, 0.5]}
    A = np.diag([2.0, 4.0, 1.0])  # lattice vectors (rows), Angstrom
    B = np.linalg.inv(A)

    grid, x, labels = grid_path(pts, "G - X-M|R-G", 4, B)

    # three segments, each with N1 + 1 points (both ends)
    assert grid.shape == (15, 3)
    np.testing.assert_allclose(grid[:5], [[0.125 * i, 0, 0] for i in range(5)])
    np.testing.assert_allclose(grid[5], pts["X"])
    np.testing.assert_allclose(grid[9], pts["M"])
    np.testing.assert_allclose(grid[10], pts["R"])
    np.testing.assert_allclose(grid[-1], pts["G"])

    # linear positions: Cartesian lengths |dk B^T|; G-X = 0.5/2, X-M = 0.5/4, R-G = |(0.25, 0.125, 0.5)|
    gx, xm, rg = 0.25, 0.125, np.linalg.norm([0.25, 0.125, 0.5])
    np.testing.assert_allclose(x[:5], np.linspace(0, gx, 5))
    np.testing.assert_allclose(x[[4, 5, 9, 10, 14]], [gx, gx, gx + xm, gx + xm, gx + xm + rg])
    assert list(labels.values()) == ["G", "X", "M|R", "G"]
    np.testing.assert_allclose(list(labels.keys()), [0, gx, gx + xm, gx + xm + rg])


# ==================================================
def test_grid_path_identity():
    grid, x, labels = grid_path({"A": [0, 0, 0], "B": [1, 0, 0]}, "A-B", 2)

    np.testing.assert_allclose(grid, [[0, 0, 0], [0.5, 0, 0], [1, 0, 0]])
    np.testing.assert_allclose(x, [0, 0.5, 1])
    assert labels == {0: "A", 1.0: "B"}


# ==================================================
@pytest.mark.parametrize(
    "r, expected",
    [
        ([0, 0.5, 0.25], [0, 0.5, 0.25]),
        ([1, -1, 2], [0, 0, 0]),
        ([-0.25, 1.75, -2.5], [0.75, 0.75, 0.5]),
    ],
)
def test_shift_into_cell(r, expected):
    s = shift_into_cell(r)
    np.testing.assert_allclose(s, expected, rtol=0, atol=1e-15)
    assert np.all((0 <= s) & (s < 1))


# ==================================================
def test_atoms_frac_shift(tmp_path):
    (tmp_path / "seed.win").write_text(
        "num_bands = 2\nnum_wann = 2\n"
        "begin unit_cell_cart\nang\n1 0 0\n0 1 0\n0 0 1\nend unit_cell_cart\n"
        "begin atoms_frac\nA -0.25 1.0 0.5\nB 0.3 0.6 -1.2\nend atoms_frac\n"
    )
    win = Win(str(tmp_path), "seed")

    np.testing.assert_allclose(win["atoms_frac_shift"][("A", 1)], [0.75, 0, 0.5], rtol=0, atol=1e-15)
    np.testing.assert_allclose(win["atoms_frac_shift"][("B", 1)], [0.3, 0.6, 0.8], rtol=0, atol=1e-15)
    assert all(type(v) is float for v in win["atoms_frac_shift"][("A", 1)])


# ==================================================
@pytest.mark.parametrize("header", [None, "dict data", "header with = and {"])
@pytest.mark.parametrize("var", [None, "info"])
def test_dict_file(tmp_path, header, var):
    d = {"a": [1, 2.5, "x=y"], "b": {"c": (1, 2)}, "kpoint": {"G": [0, 0, 0]}, "flag": True, "none": None}
    filename = str(tmp_path / "d.py")

    write_dict(filename, d, header, var)
    text = open(filename).read()

    assert text.startswith('"""' + header + '"""\n' if header is not None else (f"{var} =" if var else "{"))
    assert read_dict(filename) == d
