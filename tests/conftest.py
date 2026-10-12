"""
shared helpers for regression tests.

DFT inputs (seedname.win/amn/eig/nnkp) are taken from others/examples, and
reference outputs in tests/data were produced by pw2cw with the cwin settings below.
"""

import os
import shutil

import numpy as np
import pytest

TOPDIR = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
EXAMPLES_DIR = os.path.join(TOPDIR, "others", "examples")
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

# non-symmetrized settings (SAMB data in others/examples/*/sym require MultiPie v1).
CWIN = {
    "ch4_sl": """
restart     = cw
outdir      = {outdir}
disentangle = false
proj_min    = 0.5
verbose     = true
{extra}
""",
    "graphene_pz": """
restart       = cw
outdir        = {outdir}
disentangle   = true
proj_min      = 0.0
cwf_mu_max    = 2.3
cwf_mu_min    = -7.5
cwf_sigma_max = 2.0
cwf_sigma_min = 1.0
cwf_delta     = 0.0
svd           = true
verbose       = true
N1            = 50
{extra}
""",
}


# ==================================================
def read_hr_dat(filename):
    """
    read seedname_hr.dat.cw (wannier90 hr.dat format).

    Matrix elements are placed by the R vector and orbital indices written in each line,
    and every (R, m, n) must appear exactly once.

    Returns:
        tuple: (ndegen, irvec, Hr), Hr[R, m, n] with R in order of appearance.
    """
    with open(filename) as fp:
        lines = fp.readlines()

    num_wann = int(lines[1])
    nrpts = int(lines[2])
    nline = (nrpts + 14) // 15
    ndegen = np.array(" ".join(lines[3 : 3 + nline]).split(), dtype=int)
    assert len(ndegen) == nrpts, f"{filename}: {len(ndegen)} ndegen entries for nrpts = {nrpts}."

    data = np.loadtxt(lines[3 + nline :], ndmin=2)
    assert data.shape == (nrpts * num_wann * num_wann, 7), f"{filename}: unexpected data shape {data.shape}."

    index = data[:, :5]
    assert np.all(np.isfinite(index) & (index == np.round(index))), f"{filename}: non-integer R vector or orbital index."

    R = data[:, :3].astype(int)
    m = data[:, 3].astype(int) - 1
    n = data[:, 4].astype(int) - 1
    assert np.all((0 <= m) & (m < num_wann) & (0 <= n) & (n < num_wann)), f"{filename}: orbital index out of range."

    irvec, iR = np.unique(R, axis=0, return_index=False, return_inverse=True)
    iR = iR.reshape(-1)
    assert len(irvec) == nrpts, f"{filename}: {len(irvec)} distinct R vectors for nrpts = {nrpts}."

    count = np.zeros((nrpts, num_wann, num_wann), dtype=int)
    np.add.at(count, (iR, m, n), 1)
    assert np.all(count == 1), f"{filename}: duplicated or missing (R, m, n) entries."

    Hr = np.zeros((nrpts, num_wann, num_wann), dtype=complex)
    Hr[iR, m, n] = data[:, 5] + 1j * data[:, 6]

    # keep R vectors in order of appearance in the file.
    _, first = np.unique(iR, return_index=True)
    order = np.argsort(first)

    return ndegen, irvec[order], Hr[order]


# ==================================================
def assert_hr_equal(filename, ref_filename, atol=1e-7):
    ndegen, irvec, Hr = read_hr_dat(filename)
    ndegen_ref, irvec_ref, Hr_ref = read_hr_dat(ref_filename)

    np.testing.assert_array_equal(ndegen, ndegen_ref)
    np.testing.assert_array_equal(irvec, irvec_ref)
    np.testing.assert_allclose(Hr, Hr_ref, rtol=0, atol=atol)


# ==================================================
@pytest.fixture
def make_case(tmp_path, monkeypatch):
    """
    create a working directory with DFT inputs and seedname.cwin, and chdir into it.
    the working directory is restored by monkeypatch after the test.
    """

    def _make(seedname, outdir="./", extra="", dirname="work", band=False):
        workdir = tmp_path / dirname
        workdir.mkdir()

        exts = ("win", "amn", "eig", "nnkp") + (("band.gnu",) if band else ())
        for ext in exts:
            shutil.copy(os.path.join(EXAMPLES_DIR, seedname, f"{seedname}.{ext}"), workdir)

        cwin = CWIN[seedname].format(outdir=outdir, extra=extra)
        (workdir / f"{seedname}.cwin").write_text(cwin)

        monkeypatch.chdir(workdir)
        return workdir

    return _make
