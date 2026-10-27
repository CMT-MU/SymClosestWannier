"""
seedname.amn.cw written by write_windowed_amn (and write_amn, kept for compatibility):
the projections with the window function and proj_min applied, A_mn(k) = w(e_mk) <psi_mk|g_n>, not orthogonalized.
"""

import numpy as np
import pytest

from symclosestwannier.cw.amn import Amn
from symclosestwannier.cw.cw_creator import cw_creator
from symclosestwannier.cw.eig import Eig
from symclosestwannier.util.utility import weight_proj

# seedname.amn.cw is written with 12 decimal places.
ATOL = 1e-12


# ==================================================
def run(make_case, seedname, extra):
    workdir = make_case(seedname, extra=extra)
    cw_creator(seedname)

    Ak = Amn(str(workdir), seedname)["Ak"]
    Ek = np.array(Eig(str(workdir), seedname)["Ek"], dtype=float)
    filename = str(workdir / f"{seedname}.amn.cw")
    d = Amn(dic={}).read(filename)

    # one line per (band, orbital, k point) in the order of seedname.amn, with 1-based indices.
    num_k, num_bands, num_wann = Ak.shape
    index = np.loadtxt(filename, skiprows=2, usecols=(0, 1, 2), dtype=int)
    k, n, m = np.meshgrid(np.arange(num_k), np.arange(num_wann), np.arange(num_bands), indexing="ij")
    np.testing.assert_array_equal(index, np.stack([m.ravel() + 1, n.ravel() + 1, k.ravel() + 1], axis=1))

    return workdir, Ak, Ek, d


# ==================================================
@pytest.mark.parametrize("key", ["write_windowed_amn", "write_amn"])
def test_window_function(make_case, key):
    """
    graphene_pz: disentangle = true, proj_min = 0.
    """
    workdir, Ak, Ek, d = run(make_case, "graphene_pz", f"{key} = true")

    w = weight_proj(Ek, -7.5, 2.3, 1.0, 2.0, 0.0)

    assert (d["num_k"], d["num_bands"], d["num_wann"]) == Ak.shape
    np.testing.assert_allclose(d["Ak"], w[:, :, None] * Ak, rtol=0, atol=ATOL)
    # the window function is in effect: some bands are suppressed, and the result is not the input.
    assert np.min(w) < 1e-3
    assert not np.allclose(d["Ak"], Ak, rtol=0, atol=1e-6)
    assert not (workdir / "graphene_pz.amn_window.cw").exists()


# ==================================================
def test_proj_min_without_window_function(make_case):
    """
    ch4_sl: disentangle = false (w = 1), proj_min = 0.5; the bands with projectability <= proj_min are set to zero.
    """
    workdir, Ak, Ek, d = run(make_case, "ch4_sl", "write_windowed_amn = true")

    Pk = np.sum(np.abs(Ak) ** 2, axis=2)
    excluded = Pk <= 0.5
    expected = np.where(excluded[:, :, None], 0.0, Ak)

    assert np.any(excluded) and np.any(~excluded)
    np.testing.assert_allclose(d["Ak"], expected, rtol=0, atol=ATOL)
    assert not (workdir / "ch4_sl.amn_window.cw").exists()


# ==================================================
def test_window_function_and_proj_min(make_case):
    """
    graphene_pz with proj_min = 0.2: the bands are excluded by the projectability of the input projections, then weighted.
    """
    workdir, Ak, Ek, d = run(make_case, "graphene_pz", "write_windowed_amn = true\nproj_min = 0.2")

    w = weight_proj(Ek, -7.5, 2.3, 1.0, 2.0, 0.0)
    excluded = np.sum(np.abs(Ak) ** 2, axis=2) <= 0.2
    expected = np.where(excluded[:, :, None], 0.0, w[:, :, None] * Ak)

    assert np.any(excluded) and np.any(~excluded)
    np.testing.assert_allclose(d["Ak"], expected, rtol=0, atol=ATOL)


# ==================================================
def test_without_window_function(make_case):
    """
    without the window function and proj_min, the input projections are written as they are.
    """
    workdir, Ak, Ek, d = run(make_case, "ch4_sl", "write_windowed_amn = true\nproj_min = 0.0")

    np.testing.assert_allclose(d["Ak"], Ak, rtol=0, atol=ATOL)


# ==================================================
def test_not_written_by_default(make_case):
    workdir = make_case("graphene_pz")
    cw_creator("graphene_pz")

    assert not (workdir / "graphene_pz.amn.cw").exists()
    assert not (workdir / "graphene_pz.amn_window.cw").exists()
