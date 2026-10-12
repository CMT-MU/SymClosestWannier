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
