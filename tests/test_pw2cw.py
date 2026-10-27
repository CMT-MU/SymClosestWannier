"""
regression tests for pw2cw (CW tight-binding model construction).
"""

import os

import numpy as np
import pytest

from symclosestwannier.cw.cw_creator import cw_creator
from symclosestwannier.cw.cwin import CWin

from conftest import DATA_DIR, assert_hr_equal, read_hr_dat


# ==================================================
@pytest.mark.parametrize("seedname", ["ch4_sl", "graphene_pz"])
def test_hr_matches_reference(make_case, seedname):
    workdir = make_case(seedname, extra="write_hr = true")

    cw_creator(seedname)

    assert_hr_equal(workdir / f"{seedname}_hr.dat.cw", os.path.join(DATA_DIR, f"{seedname}_hr.dat.cw"))


# ==================================================
def test_outdir_subdirectory(make_case):
    """
    inputs are read from the current directory, outputs are written to outdir.
    """
    seedname = "ch4_sl"
    workdir = make_case(seedname, outdir="./out", extra="write_hr = true\nwrite_info_data = true")

    cw_creator(seedname)

    outdir = workdir / "out"
    assert (outdir / f"{seedname}.cwout").is_file()
    assert (outdir / f"{seedname}.hdf5").is_file()
    assert not (outdir / "out").exists()
    assert_hr_equal(outdir / f"{seedname}_hr.dat.cw", os.path.join(DATA_DIR, f"{seedname}_hr.dat.cw"))


# ==================================================
@pytest.mark.parametrize("outdir, expected", [("./", "."), ("./out/", "out"), ("../other", "../other")])
def test_cwin_outdir_is_relative_to_cwin_directory(make_case, outdir, expected):
    workdir = make_case("ch4_sl", outdir=outdir)

    cwin = CWin(str(workdir), "ch4_sl")

    assert cwin["outdir"] == os.path.normpath(os.path.join(workdir, expected))
    assert cwin["mp_outdir"] == os.path.normpath(str(workdir))


# ==================================================
def test_write_sr(make_case):
    seedname = "ch4_sl"
    workdir = make_case(seedname, extra="write_sr = true")

    cw_creator(seedname)

    ndegen, irvec, Sr = read_hr_dat(workdir / f"{seedname}_sr.dat.cw")
    _, irvec_hr, _ = read_hr_dat(os.path.join(DATA_DIR, f"{seedname}_hr.dat.cw"))

    # overlap matrix: same R vectors as Hr, Hermitian, and positive diagonal at R = 0.
    np.testing.assert_array_equal(irvec, irvec_hr)
    assert np.all(np.isfinite(Sr))
    S0 = Sr[np.where(np.all(irvec == 0, axis=1))[0][0]]
    np.testing.assert_allclose(S0, S0.conj().T, atol=1e-7)
    assert np.all(np.diag(S0).real > 0)

    assert (workdir / f"{seedname}_sr_R_dep.dat.cw").stat().st_size > 0


# ==================================================
@pytest.mark.parametrize("verbose", [False, True])
def test_stdout_follows_verbose(make_case, capsys, verbose):
    workdir = make_case("ch4_sl", extra=f"verbose = {str(verbose).lower()}")

    cw_creator("ch4_sl")

    cwout = (workdir / "ch4_sl.cwout").read_text()
    assert "Occupancy" in cwout
    assert "     Sum " in cwout

    out = capsys.readouterr().out
    assert "occ_all" not in out
    assert ("Occupancy" in out) == verbose
    if not verbose:
        assert out.strip() == ""


# ==================================================
def test_write_tb_with_tb_position(make_case):
    """
    with tb_position = true, seedname_tb.dat is written without seedname.mmn: the position operator is diagonal at
    R = 0 and equal to the projection centres (here the C atoms of graphene), and H is that of seedname_hr.dat.
    """
    from test_write_tb import read_tb_dat

    seedname = "graphene_pz"
    workdir = make_case(seedname, extra="write_hr = true\nwrite_tb = true\ntb_position = true")
    assert not (workdir / f"{seedname}.mmn").exists()

    cw_creator(seedname)

    lattice, ndegen, irvec, Hr, Ar = read_tb_dat(workdir / f"{seedname}_tb.dat.cw")
    ndegen_hr, irvec_hr, Hr_hr = read_hr_dat(workdir / f"{seedname}_hr.dat.cw")

    np.testing.assert_array_equal(irvec, irvec_hr)
    np.testing.assert_array_equal(ndegen, ndegen_hr)
    np.testing.assert_allclose(Hr, Hr_hr, rtol=0, atol=1e-7)

    bohr = 0.529177249  # the lattice of graphene_pz.win is given in bohr.
    a1, a2 = np.array([4.6014827822, 0, 0]) * bohr, np.array([-2.3007413911, 3.9850009844618968, 0]) * bohr
    tau = np.array([2 / 3 * a1 + 1 / 3 * a2, 1 / 3 * a1 + 2 / 3 * a2])
    np.testing.assert_allclose(lattice[:2], [a1, a2], rtol=0, atol=1e-4)

    R0 = np.where(np.all(irvec == 0, axis=1))[0][0]
    np.testing.assert_allclose(np.diagonal(Ar[:, R0], axis1=1, axis2=2).T, tau, rtol=0, atol=1e-4)
    Ar_rest = Ar.copy()
    Ar_rest[:, R0, [0, 1], [0, 1]] = 0
    assert np.all(Ar_rest == 0)

    # wannier-berri (optional) takes the Wannier centres from this position operator.
    try:
        from wannierberri.system.system_tb import get_system_tb
    except ImportError:
        return
    system = get_system_tb(tb_file=str(workdir / f"{seedname}_tb.dat.cw"), berry=True, silent=True)
    np.testing.assert_allclose(system.wannier_centers_cart, tau, rtol=0, atol=1e-4)


# ==================================================
def test_write_rmn_with_tb_position(make_case):
    """
    the position operator from the projection centres is on-site: seedname_r.dat.cw is written, the R-dependence files
    (which would be NaN) are not.
    """
    seedname = "graphene_pz"
    workdir = make_case(seedname, extra="write_rmn = true\ntb_position = true")

    cw_creator(seedname)

    assert (workdir / f"{seedname}_r.dat.cw").stat().st_size > 0
    assert not list(workdir.glob(f"{seedname}_r*_R_dep.dat.cw"))
