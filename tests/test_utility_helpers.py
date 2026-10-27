"""
helpers that replace gcoreutils: grid points along a k path (grid_path), fractional coordinates shifted into the unit
cell (shift_into_cell, atoms_frac_shift of seedname.win), and the dict text files of CWManager (read_dict, write_dict);
and the k points of a plane for Fermi-surface plots (surface_grid).
"""

import numpy as np
import pytest

from symclosestwannier.cw.cw_manager import read_dict, write_dict
from symclosestwannier.cw.win import Win
from symclosestwannier.util.utility import grid_path, shift_into_cell, surface_grid


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
def test_shift_into_cell_boundaries():
    below_0, below_1, above_1 = np.nextafter(0.0, -1.0), np.nextafter(1.0, 0.0), np.nextafter(1.0, 2.0)
    r = [-1e-17, below_0, below_1, above_1, np.nextafter(-1.0, 0.0), -1.0, 2.0 - 1e-16]

    s = shift_into_cell(r)

    assert np.all((0 <= s) & (s < 1))
    # within one rounding of r modulo 1, counting 1 as 0
    d = np.abs(s - np.mod(r, 1.0))
    np.testing.assert_allclose(np.minimum(d, 1 - d), 0, rtol=0, atol=2.3e-16)
    assert s[0] == 0.0 and s[1] == 0.0 and s[2] == below_1


# ==================================================
@pytest.mark.parametrize("r, expected", [(0.5, 0.5), (-0.25, 0.75), (-1e-17, 0.0), (np.float64(3.0), 0.0)])
def test_shift_into_cell_scalar(r, expected):
    s = shift_into_cell(r)
    assert s.shape == () and s == expected


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


# ==================================================
@pytest.mark.parametrize(
    "text",
    [
        '{"a": 1}\n{"b": 2}\n',
        'x = 1\ny = {"a": 1}\n',
        "import os\n",
        'a = b = {"a": 1}\n',
        '"""header only"""\n',
        "x = [1, 2]\n",
        "3\n",
        '"""header"""\nx = [1, 2]\n',
    ],
)
def test_read_dict_malformed(tmp_path, text):
    (tmp_path / "d.py").write_text(text)
    with pytest.raises(ValueError, match="not a dict file"):
        read_dict(str(tmp_path / "d.py"))


# ==================================================
@pytest.mark.parametrize("N1, N2", [(4, 3), (1, 1), (1, 3), (4, 1)])
def test_surface_grid(N1, N2):
    A = np.array([[2.0, 0.0, 0.0], [-1.0, 1.5, 0.0], [0.3, 0.2, 5.0]])
    V = np.dot(A[0], np.cross(A[1], A[2]))
    x1, x2, x3 = np.linspace(-1, 1, N1), np.linspace(-0.5, 0.5, N2), 0.25

    grid_2d, grid = surface_grid(A, V, x1, x2, x3)

    # the loops that surface_grid replaces (same arithmetic)
    B = np.linalg.inv(A).T
    ref_2d = np.array(
        [
            [
                np.array(
                    [
                        x1[i] / V * np.linalg.norm(np.cross(A[1], A[2])),
                        x2[j] / V * np.linalg.norm(np.cross(A[2], A[0])),
                        x3 / V * np.linalg.norm(np.cross(A[0], A[1])),
                    ]
                )
                for j in range(N2)
            ]
            for i in range(N1)
        ],
        dtype=np.float64,
    )
    ref = np.array([ref_2d[i, j] @ np.linalg.inv(B) for i in range(N1) for j in range(N2)], dtype=np.float64)

    assert grid_2d.shape == (N1, N2, 3) and grid.shape == (N1 * N2, 3)
    assert grid_2d.dtype == np.float64 and grid.dtype == np.float64
    np.testing.assert_array_equal(grid_2d, ref_2d)
    np.testing.assert_allclose(grid, ref, rtol=0, atol=1e-15)
    # fractional coordinates of the reciprocal lattice B = (A^-1)^T: k_frac = k B^-1 = k A^T
    np.testing.assert_allclose(grid, ref_2d.reshape(-1, 3) @ A.T, rtol=0, atol=1e-14)


# ==================================================
def test_band_distance_windows():
    # bands below Ef, Ef + 2 eV, Ef + 5 eV and above, with errors of 10, 20, 40 and 80 meV.
    from symclosestwannier.util.utility import band_distance

    Ak = np.eye(4, dtype=complex)[np.newaxis]
    Ek = np.array([[-1.0, 1.0, 3.0, 7.0]])
    Hk = np.diag([-0.99, 1.02, 3.04, 7.08]).astype(complex)[np.newaxis]
    eta_0, eta_0_max, eta_2, eta_2_max, eta_5, eta_5_max = band_distance(Ak, Ek, Hk, ef=0.0)
    assert eta_0 == pytest.approx(10.0) and eta_0_max == pytest.approx(10.0)
    assert eta_2 == pytest.approx(np.sqrt((10**2 + 20**2) / 2)) and eta_2_max == pytest.approx(20.0)
    # not up to Ef + 10 eV.
    assert eta_5 == pytest.approx(np.sqrt((10**2 + 20**2 + 40**2) / 3)) and eta_5_max == pytest.approx(40.0)
